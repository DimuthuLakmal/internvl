import os
import inspect
import json
from transformers import Trainer, GenerationConfig
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import PreTrainedModel
from typing import Optional, Dict, Union, List


class TargetSFTTrainer(Trainer):

    INTERNVL_SUBDIR = "internvl_model"
    PREDICTOR_SUBDIR = "goal_predictor"

    PREDICTOR_WEIGHTS_NAME = "goal_predictor_weights.pt"
    PREDICTOR_CONFIG_NAME = "goal_predictor_config.json"
    METADATA_NAME = "checkpoint_metadata.json"

    INTERNVL_ATTRIBUTE = "llm_model"
    INTERNVL_STATE_PREFIX = "llm_model."

    # Pixel-space target std (x, y) measured on train_internvl_target_pred_filtered_y.jsonl
    COORD_STD_PX = (40.0, 17.2)

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_i = 0

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

    def compute_y_sample_weights(self, target_y, center_y=112.0, base_weight=1.0, scale=0.15, max_weight=5.0):
        """
        target_y: [B], pixel-space y values
        """
        distance_from_center = torch.abs(target_y.float() - center_y)

        weights = base_weight + scale * distance_from_center
        weights = torch.clamp(weights, max=max_weight)

        return weights

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):

        inputs["labels"] = None

        target_xy = inputs["raw_labels"]
        target_xy_norm = target_xy / (224 - 1) * 2.0 - 1.0

        outputs = model(inputs)

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
            max_weight=5.0,
        )  # [B]

        # Same tail weighting for x around the ego's x anchor, so far/fast goals aren't averaged away
        x_sample_weights = self.compute_y_sample_weights(
            target_y=target_xy[:, 0],
            center_y=56.0,
            base_weight=1.0,
            scale=0.1,
            max_weight=20.0,
        )  # [B]

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

        if self.batch_i % 100 == 0 and self.is_world_process_zero():
            with torch.no_grad():
                pred_xy_pixel = (outputs.detach() + 1.0) * 0.5 * (224 - 1)

                abs_error = torch.abs(pred_xy_pixel - target_xy)
                err_x = abs_error[:, 0].mean()
                err_y = abs_error[:, 1].mean()
                err_l2 = torch.norm(pred_xy_pixel - target_xy, dim=-1).mean()

            self.log({
                "goal/loss": loss.item(),
                "goal/err_x_px": err_x.item(),
                "goal/err_y_px": err_y.item(),
                "goal/err_l2_px": err_l2.item(),
            })

            log_msg = (
                    f"batch={self.batch_i} "
                    f"loss={loss.item():.6f} "
                    f"err_x={err_x.item():.3f}px "
                    f"err_y={err_y.item():.3f}px "
                    f"err_l2={err_l2.item():.3f}px\n"
                    f"pred_xy={pred_xy_pixel[:5].detach().cpu().tolist()}\n"
                    f"target_xy={target_xy[:5].detach().cpu().tolist()}\n"
                    "--------------------------------------------------\n"
                )

            with open("training_log_kd.txt", "a") as f:
                f.write(log_msg)
                f.flush()


        self.batch_i += 1

        return loss