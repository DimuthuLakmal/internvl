import os
import inspect
import json
from transformers import Trainer, GenerationConfig
import numpy as np
import pickle
import re
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import PreTrainedModel
from typing import Optional, Dict, Union, List
from transformers import TrainerCallback


class EMACallback(TrainerCallback):
    """
    Keeps an fp32 exponential moving average of all trainable parameters.

    The live weights bounce around the minimum under noisy small-batch
    gradients; their running average sits closer to its centre. Training
    still updates only the live weights; the EMA copy is saved alongside
    each checkpoint for evaluation.
    """

    def __init__(self, trainer, decay: float = 0.999):
        self.trainer = trainer
        self.decay = decay
        self.ema_params: Dict[str, torch.Tensor] = {}

    def _trainable_params(self):
        goal_model = self.trainer._get_unwrapped_goal_model()
        return [(name, param) for name, param in goal_model.named_parameters() if param.requires_grad]

    def on_train_begin(self, args, state, control, **kwargs):
        # Re-initialized from the current weights, also when resuming from a checkpoint
        self.ema_params = {
            name: param.detach().float().clone()
            for name, param in self._trainable_params()
        }

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        # Called once per optimizer step (after gradient accumulation)
        for name, param in self._trainable_params():
            self.ema_params[name].mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.ema_params.items()}


class TargetSFTTrainer(Trainer):

    INTERNVL_SUBDIR = "internvl_model"
    PREDICTOR_SUBDIR = "goal_predictor"

    PREDICTOR_WEIGHTS_NAME = "goal_predictor_weights.pt"
    EMA_WEIGHTS_NAME = "ema_trainable.pt"
    PREDICTOR_CONFIG_NAME = "goal_predictor_config.json"
    METADATA_NAME = "checkpoint_metadata.json"

    INTERNVL_ATTRIBUTE = "llm_model"
    INTERNVL_STATE_PREFIX = "llm_model."

    # Pixel-space target std (x, y) measured on train_internvl_target_pred_filtered_y.jsonl
    COORD_STD_PX = (40.0, 17.2)

    # Smoothing factor of the running pixel-error averages written to the log
    ERR_EMA_ALPHA = 0.01

    def __init__(
        self,
        vision_lr: Optional[float] = None,
        vit_layer_decay: float = 1.0,
        ema_decay: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_i = 0
        self.vision_lr = vision_lr
        self.vit_layer_decay = vit_layer_decay
        self.err_ema = None

        self.ema_callback = None
        if ema_decay > 0:
            self.ema_callback = EMACallback(self, decay=ema_decay)
            self.add_callback(self.ema_callback)

        # A DeepSpeed config with an "optimizer" section makes HF build the
        # optimizer from a flat parameter list and skip create_optimizer(),
        # which would silently drop the per-group learning rates.
        hf_ds_config = getattr(self.args, "hf_deepspeed_config", None)
        if (self._use_param_groups() and hf_ds_config is not None
                and "optimizer" in hf_ds_config.config):
            raise ValueError(
                "vision_lr / vit_layer_decay need a DeepSpeed config without an "
                "\"optimizer\" section (e.g. zero_stage1_config_paramgroups.json)."
            )

    def _use_param_groups(self) -> bool:
        return self.vision_lr is not None or self.vit_layer_decay != 1.0

    @staticmethod
    def _vit_layer_id(name: str, num_layers: int) -> int:
        """
        0 for ViT embeddings, i + 1 for encoder layer i, num_layers + 1 for
        anything after the encoder.
        """
        if ".embeddings." in name:
            return 0
        match = re.search(r"\.encoder\.layers\.(\d+)\.", name)
        if match:
            return int(match.group(1)) + 1
        return num_layers + 1

    def create_optimizer(self):
        """
        Build the optimizer with separate parameter groups: the ViT uses
        vision_lr (with optional layer-wise decay), everything else
        (mlp1, goal predictor, LLM if unfrozen) uses args.learning_rate.
        Biases, norms, layer-scales and position/class embeddings are not
        weight-decayed.
        """
        if not self._use_param_groups():
            return super().create_optimizer()
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.args.learning_rate
        vision_lr = self.vision_lr if self.vision_lr is not None else base_lr
        goal_model = self._get_unwrapped_goal_model()
        num_layers = goal_model.llm_model.config.vision_config.num_hidden_layers

        groups = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            no_decay = (param.ndim <= 1 or name.endswith(".bias")
                        or "position_embedding" in name or "class_embedding" in name)
            weight_decay = 0.0 if no_decay else self.args.weight_decay

            if ".vision_model." in name:
                layer_id = self._vit_layer_id(name, num_layers)
                lr = vision_lr * self.vit_layer_decay ** (num_layers + 1 - layer_id)
                group_name = f"vit_layer_{layer_id}"
            else:
                lr = base_lr
                group_name = "head"

            key = (group_name, weight_decay)
            if key not in groups:
                groups[key] = {"params": [], "lr": lr, "weight_decay": weight_decay,
                               "group_name": group_name, "num_params": 0}
            groups[key]["params"].append(param)
            groups[key]["num_params"] += param.numel()

        if self.is_world_process_zero():
            for (group_name, weight_decay), group in sorted(groups.items()):
                print(f"[Optimizer] {group_name:<14} lr={group['lr']:.3e} "
                      f"wd={weight_decay} params={group['num_params']:,}")

        param_groups = []
        for group in groups.values():
            group.pop("num_params")
            group.pop("group_name")
            param_groups.append(group)

        if "model" in inspect.signature(Trainer.get_optimizer_cls_and_kwargs).parameters:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        else:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def _get_unwrapped_goal_model(self) -> nn.Module:
        """
        Remove DDP, DeepSpeed, or Accelerate wrappers and return the
        original MLLMGoalPredictor instance.
        """
        return self.accelerator.unwrap_model(self.model)

    @staticmethod
    def _remove_module_prefix(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Some distributed wrappers add `module.` to all parameter names.
        Remove it before splitting the state dictionary.
        """
        if not state_dict:
            return state_dict

        if all(key.startswith("module.") for key in state_dict):
            return {
                key[len("module."):]: value
                for key, value in state_dict.items()
            }

        return state_dict

    def _split_component_state_dicts(
        self,
        state_dict: Dict[str, torch.Tensor],
    ):
        """
        Separate:

            llm_model.*  -> InternVL model
            everything else -> MLLMGoalPredictor-specific modules
        """
        state_dict = self._remove_module_prefix(state_dict)

        internvl_state_dict = {}
        predictor_state_dict = {}

        prefix = self.INTERNVL_STATE_PREFIX

        for key, value in state_dict.items():
            if key.startswith(prefix):
                internvl_key = key[len(prefix):]
                internvl_state_dict[internvl_key] = value
            else:
                predictor_state_dict[key] = value

        if not internvl_state_dict:
            available_keys = list(state_dict.keys())[:20]

            raise RuntimeError(
                f"No parameters beginning with '{prefix}' were found. "
                f"Check whether MLLMGoalPredictor stores InternVL as "
                f"`self.{self.INTERNVL_ATTRIBUTE}`.\n"
                f"Example state-dict keys: {available_keys}"
            )

        return internvl_state_dict, predictor_state_dict

    def _save_predictor_config(
        self,
        goal_model: nn.Module,
        predictor_dir: str,
    ) -> None:
        """
        Save the constructor parameters required to reconstruct
        MLLMGoalPredictor.

        The preferred approach is to define `get_init_config()` inside
        MLLMGoalPredictor.
        """
        predictor_config = None

        if hasattr(goal_model, "get_init_config"):
            predictor_config = goal_model.get_init_config()
        elif hasattr(goal_model, "init_config"):
            predictor_config = goal_model.init_config

        if predictor_config is None:
            print(
                "[Checkpoint warning] MLLMGoalPredictor does not expose "
                "`get_init_config()` or `init_config`. The predictor "
                "architecture parameters were not saved."
            )
            return

        config_path = os.path.join(
            predictor_dir,
            self.PREDICTOR_CONFIG_NAME,
        )

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(
                predictor_config,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def _save_generation_config(
        self,
        internvl_model: nn.Module,
        internvl_dir: str,
    ) -> None:
        """
        Save generation_config.json.

        Depending on the InternVL implementation, GenerationConfig may
        belong to either InternVLChatModel or its language_model.
        """
        generation_config = getattr(
            internvl_model,
            "generation_config",
            None,
        )

        language_model = getattr(
            internvl_model,
            "language_model",
            None,
        )

        if generation_config is None and language_model is not None:
            generation_config = getattr(
                language_model,
                "generation_config",
                None,
            )

        if (
            generation_config is None
            and language_model is not None
            and hasattr(language_model, "config")
        ):
            generation_config = GenerationConfig.from_model_config(
                language_model.config
            )

        if generation_config is not None:
            generation_config.save_pretrained(internvl_dir)
        else:
            print(
                "[Checkpoint warning] generation_config.json could not "
                "be created because no generation configuration was found."
            )

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Save the model using a component-based checkpoint layout.

        This method is invoked for both:
            - automatic checkpoint saves;
            - trainer.save_model().
        """
        output_dir = output_dir or self.args.output_dir

        # Prefer the state_dict supplied by Trainer. This is important
        # for distributed strategies because Trainer may already have
        # gathered or consolidated the model state.
        if state_dict is None:
            model_to_save = getattr(
                self,
                "model_wrapped",
                self.model,
            )

            state_dict = self.accelerator.get_state_dict(
                model_to_save
            )

        # Only the saving process writes files. State-dict collection
        # should happen before this check because some distributed
        # configurations require collective communication.
        if not self.args.should_save:
            return

        os.makedirs(output_dir, exist_ok=True)

        internvl_dir = os.path.join(
            output_dir,
            self.INTERNVL_SUBDIR,
        )
        predictor_dir = os.path.join(
            output_dir,
            self.PREDICTOR_SUBDIR,
        )

        os.makedirs(internvl_dir, exist_ok=True)
        os.makedirs(predictor_dir, exist_ok=True)

        goal_model = self._get_unwrapped_goal_model()

        internvl_model = getattr(
            goal_model,
            self.INTERNVL_ATTRIBUTE,
            None,
        )

        if internvl_model is None:
            raise AttributeError(
                f"MLLMGoalPredictor does not contain "
                f"`self.{self.INTERNVL_ATTRIBUTE}`."
            )

        (
            internvl_state_dict,
            predictor_state_dict,
        ) = self._split_component_state_dicts(state_dict)

        # -------------------------------------------------------------
        # 1. Save the complete InternVL component
        # -------------------------------------------------------------
        if not hasattr(internvl_model, "save_pretrained"):
            raise TypeError(
                "The nested InternVL model does not implement "
                "`save_pretrained()`."
            )

        save_kwargs = {
            "state_dict": internvl_state_dict,
            "max_shard_size": "5GB",
        }

        # Transformers 4.x supports safe_serialization. Using signature
        # inspection also avoids problems if the installed version has
        # changed this argument.
        save_signature = inspect.signature(
            internvl_model.save_pretrained
        )

        if "safe_serialization" in save_signature.parameters:
            save_kwargs["safe_serialization"] = getattr(
                self.args,
                "save_safetensors",
                True,
            )

        internvl_model.save_pretrained(
            internvl_dir,
            **save_kwargs,
        )

        # Explicitly save config.json as a safeguard for custom
        # InternVL model implementations.
        if hasattr(internvl_model, "config"):
            internvl_model.config.save_pretrained(internvl_dir)

        self._save_generation_config(
            internvl_model=internvl_model,
            internvl_dir=internvl_dir,
        )

        # -------------------------------------------------------------
        # 2. Save tokenizer or processor with InternVL
        # -------------------------------------------------------------
        processing_class = getattr(
            self,
            "processing_class",
            None,
        )

        if processing_class is None:
            # Compatibility with Transformers versions that store it as
            # self.tokenizer.
            processing_class = getattr(
                self,
                "tokenizer",
                None,
            )

        if (
            processing_class is not None
            and hasattr(processing_class, "save_pretrained")
        ):
            processing_class.save_pretrained(internvl_dir)

        # -------------------------------------------------------------
        # 3. Save only the non-InternVL predictor parameters
        # -------------------------------------------------------------
        predictor_state_dict = {
            key: value.detach().cpu()
            for key, value in predictor_state_dict.items()
        }

        predictor_weights_path = os.path.join(
            predictor_dir,
            self.PREDICTOR_WEIGHTS_NAME,
        )

        torch.save(
            predictor_state_dict,
            predictor_weights_path,
        )

        # EMA of all trainable params (predictor + any unfrozen InternVL parts),
        # keyed by full goal-model parameter name
        if self.ema_callback is not None and self.ema_callback.ema_params:
            torch.save(
                self.ema_callback.state_dict(),
                os.path.join(predictor_dir, self.EMA_WEIGHTS_NAME),
            )

        self._save_predictor_config(
            goal_model=goal_model,
            predictor_dir=predictor_dir,
        )

        # -------------------------------------------------------------
        # 4. Save metadata describing the checkpoint structure
        # -------------------------------------------------------------
        metadata = {
            "checkpoint_format": "component_checkpoint_v1",
            "goal_model_class": goal_model.__class__.__name__,
            "internvl_model_class": internvl_model.__class__.__name__,
            "internvl_attribute": self.INTERNVL_ATTRIBUTE,
            "internvl_state_prefix": self.INTERNVL_STATE_PREFIX,
            "internvl_subdirectory": self.INTERNVL_SUBDIR,
            "predictor_subdirectory": self.PREDICTOR_SUBDIR,
            "predictor_weights": self.PREDICTOR_WEIGHTS_NAME,
            "global_step": int(self.state.global_step),
        }

        metadata_path = os.path.join(
            predictor_dir,
            self.METADATA_NAME,
        )

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        # Trainer normally writes training_args.bin from its _save()
        # method. Because this implementation replaces that method,
        # save it explicitly.
        torch.save(
            self.args,
            os.path.join(output_dir, "training_args.bin"),
        )

        print(
            f"[Checkpoint] Saved component checkpoint to {output_dir}\n"
            f"  InternVL: {internvl_dir}\n"
            f"  Goal predictor: {predictor_dir}"
        )

    def _decode_input_texts(self, inputs: Dict[str, torch.Tensor], max_samples: int = 5) -> List[str]:
        """
        Decode the text prompt of the first samples in the batch for logging.
        Padding and the (many) image-context placeholder tokens are dropped.
        """
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        input_ids = inputs.get("input_ids")
        if tokenizer is None or input_ids is None:
            return []

        attention_mask = inputs.get("attention_mask")
        img_context_token_id = self._get_unwrapped_goal_model().llm_model.img_context_token_id

        texts = []
        for i in range(min(max_samples, input_ids.shape[0])):
            keep = input_ids[i] != img_context_token_id
            if attention_mask is not None:
                keep = keep & attention_mask[i].bool()
            token_ids = input_ids[i][keep].tolist()
            texts.append(tokenizer.decode(token_ids, skip_special_tokens=False))

        return texts

    def compute_y_sample_weights(self, target_y, center_y=112.0, base_weight=1.0, scale=0.15, max_weight=5.0):
        """
        target_y: [B], pixel-space y values
        """
        distance_from_center = torch.abs(target_y.float() - center_y)

        weights = base_weight + scale * distance_from_center
        weights = torch.clamp(weights, max=max_weight)

        return weights

    def _regression_loss(self, outputs, target_xy):
        """Weighted SmoothL1 in normalized + pixel space for the scalar-regression predictor."""
        target_xy_norm = target_xy / (224 - 1) * 2.0 - 1.0

        # Per-coordinate SmoothL1 loss
        loss_per_coord = F.smooth_l1_loss(
            outputs,
            target_xy_norm,
            reduction="none",
        )  # [B, 2]

        y_sample_weights = self.compute_y_sample_weights(
            target_y=target_xy[:, 1],
            center_y=112.0,
            base_weight=1.0,
            scale=0.2,
            max_weight=20.0,
        )  # [B]

        # Same tail weighting for x around the ego's x anchor, so far/fast goals aren't averaged away
        x_sample_weights = self.compute_y_sample_weights(
            target_y=target_xy[:, 0],
            center_y=56.0,
            base_weight=1.0,
            scale=0.1,
            max_weight=20.0,
        )  # [B]

        # Normalize to mean 1 per batch: keeps the relative tail emphasis but fixes the
        # per-step gradient scale, so one high-weight sample can't produce a huge step
        x_sample_weights = x_sample_weights / x_sample_weights.mean()
        y_sample_weights = y_sample_weights / y_sample_weights.mean()

        # Scale y so errors measured in each coordinate's std weigh equally (x unchanged).
        # Normalized-space errors are < 1 (quadratic smooth-L1) -> squared ratio;
        # pixel-space errors are > 1 (linear smooth-L1) -> plain ratio.
        y_scale = self.COORD_STD_PX[0] / self.COORD_STD_PX[1]

        # Weight each sample's y-loss before averaging so tail targets keep their influence
        loss_x = (x_sample_weights * loss_per_coord[:, 0]).mean()
        loss_y = y_scale ** 2 * (y_sample_weights * loss_per_coord[:, 1]).mean()

        loss_norm = loss_x + loss_y

        pred_xy_px = (outputs + 1.0) * 0.5 * (224 - 1)
        loss_px = F.smooth_l1_loss(pred_xy_px, target_xy, reduction="none")
        loss_x_px = (x_sample_weights * loss_px[:, 0]).mean()
        loss_y_px = y_scale * (y_sample_weights * loss_px[:, 1]).mean()

        loss_px = loss_x_px + loss_y_px

        loss = loss_norm + 0.01 * loss_px

        return loss

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):

        inputs["labels"] = None

        target_xy = inputs["raw_labels"].float()

        outputs = model(inputs)

        loss_parts = {}
        if isinstance(outputs, dict):
            # Heatmap predictor: goal-cell CE + offset loss; the decoded goal is only used for logging
            loss, loss_parts = self._get_unwrapped_goal_model().heatmap_loss(outputs, target_xy)
            outputs = outputs["pred_xy_norm"].float()
        else:
            outputs = outputs.float()
            loss = self._regression_loss(outputs, target_xy)

        with torch.no_grad():
            pred_xy_pixel = (outputs.detach() + 1.0) * 0.5 * (224 - 1)

            abs_error = torch.abs(pred_xy_pixel - target_xy)
            err_x = abs_error[:, 0].mean()
            err_y = abs_error[:, 1].mean()
            err_l2 = torch.norm(pred_xy_pixel - target_xy, dim=-1).mean()

            # Running average over every micro-step (kept on GPU, synced only at log time),
            # so the trend is visible instead of single 2-sample readings
            errs = torch.stack([err_x, err_y, err_l2])
            if self.err_ema is None:
                self.err_ema = errs.clone()
            else:
                self.err_ema.lerp_(errs, self.ERR_EMA_ALPHA)

        if self.batch_i % 100 == 0 and self.is_world_process_zero():
            err_x_ema, err_y_ema, err_l2_ema = self.err_ema.tolist()

            self.log({
                "goal/loss": loss.item(),
                "goal/err_x_px": err_x.item(),
                "goal/err_y_px": err_y.item(),
                "goal/err_l2_px": err_l2.item(),
                "goal/err_x_px_ema": err_x_ema,
                "goal/err_y_px_ema": err_y_ema,
                "goal/err_l2_px_ema": err_l2_ema,
                **{f"goal/loss_{name}": value.item() for name, value in loss_parts.items()},
            })

            input_texts = self._decode_input_texts(inputs, max_samples=5)
            input_text_msg = "".join(
                f"input_text[{i}]=\n{text}\n" for i, text in enumerate(input_texts)
            )

            log_msg = (
                    f"batch={self.batch_i} "
                    f"loss={loss.item():.6f} "
                    f"err_x={err_x.item():.3f}px "
                    f"err_y={err_y.item():.3f}px "
                    f"err_l2={err_l2.item():.3f}px "
                    f"ema_err_x={err_x_ema:.3f}px "
                    f"ema_err_y={err_y_ema:.3f}px "
                    f"ema_err_l2={err_l2_ema:.3f}px\n"
                    f"pred_xy={pred_xy_pixel[:5].detach().cpu().tolist()}\n"
                    f"target_xy={target_xy[:5].detach().cpu().tolist()}\n"
                    f"{input_text_msg}"
                    "--------------------------------------------------\n"
                )

            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "training_log.txt"), "a") as f:
                f.write(log_msg)
                f.flush()


        self.batch_i += 1

        return loss