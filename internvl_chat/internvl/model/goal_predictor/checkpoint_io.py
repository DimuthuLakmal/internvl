import json
import os

import torch
from transformers import AutoTokenizer

from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.goal_predictor.mllm_spatial_goal_predictor import (MLLMGoalPredictor,
                                                                       MLLMHeatmapGoalPredictor)

GOAL_MODEL_CLASSES = {
    cls.__name__: cls for cls in (MLLMGoalPredictor, MLLMHeatmapGoalPredictor)
}


def read_predictor_config(checkpoint_dir: str):
    """
    Return (goal model class, constructor config) saved by TargetSFTTrainer._save.
    Checkpoints written before checkpoint_metadata.json recorded the class are MLLMGoalPredictor.
    """
    predictor_dir = os.path.join(checkpoint_dir, 'goal_predictor')
    predictor_config_path = os.path.join(predictor_dir, 'goal_predictor_config.json')
    if not os.path.exists(predictor_config_path):
        raise FileNotFoundError(f'Required checkpoint component was not found: {predictor_config_path}')

    with open(predictor_config_path, 'r', encoding='utf-8') as file:
        predictor_config = json.load(file)

    class_name = 'MLLMGoalPredictor'
    metadata_path = os.path.join(predictor_dir, 'checkpoint_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as file:
            class_name = json.load(file).get('goal_model_class', class_name)

    if class_name not in GOAL_MODEL_CLASSES:
        raise ValueError(f'Unknown goal model class in checkpoint: {class_name}')
    return GOAL_MODEL_CLASSES[class_name], predictor_config


def load_llm_checkpoint(
    checkpoint_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    system_message: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    load_tokenizer: bool = True,
    use_fast_tokenizer: bool = False,
    config=None,
):
    """
    Load the InternVL component of a component-checkpoint saved by
    TargetSFTTrainer (see internvl/train/target_trainer.py).

    Mode (train/eval) is left to the caller: this does not call .eval().
    Pass `config` to override the saved config (e.g. drop_path_rate);
    by default the config saved with the checkpoint is used.
    """
    internvl_dir = os.path.join(checkpoint_dir, 'internvl_model')

    from_pretrained_kwargs = {'config': config} if config is not None else {}
    internvl_model = InternVLChatModel.from_pretrained(
        internvl_dir,
        **from_pretrained_kwargs,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        system_message=system_message,
    )

    if not load_tokenizer:
        return internvl_model

    tokenizer = AutoTokenizer.from_pretrained(
        internvl_dir,
        trust_remote_code=True,
        use_fast=use_fast_tokenizer,
    )
    return internvl_model, tokenizer


def load_goal_predictor_checkpoint(
    checkpoint_dir: str,
    internvl_model: InternVLChatModel,
    tokenizer: AutoTokenizer,
    use_ema: bool = False,
):
    """
    Reconstruct MLLMGoalPredictor from the predictor-only weights and config
    saved by TargetSFTTrainer._save alongside the InternVL checkpoint.

    With use_ema=True, the EMA weights (ema_trainable.pt, covering all params
    that were trainable, including unfrozen InternVL parts) are overlaid on top.
    """
    predictor_dir = os.path.join(checkpoint_dir, 'goal_predictor')
    predictor_weights_path = os.path.join(predictor_dir, 'goal_predictor_weights.pt')

    if not os.path.exists(predictor_weights_path):
        raise FileNotFoundError(f'Required checkpoint component was not found: {predictor_weights_path}')

    goal_model_cls, predictor_config = read_predictor_config(checkpoint_dir)

    goal_model = goal_model_cls(
        **predictor_config,
        llm_model=internvl_model,
        llm_tokenizer=tokenizer,
    )

    try:
        predictor_state_dict = torch.load(predictor_weights_path, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases.
        predictor_state_dict = torch.load(predictor_weights_path)

    incompatible = goal_model.load_state_dict(predictor_state_dict, strict=False)

    # llm_model.* is deliberately absent from predictor_weights.pt.
    invalid_missing_keys = [
        key for key in incompatible.missing_keys if not key.startswith('llm_model.')
    ]
    if invalid_missing_keys:
        raise RuntimeError(
            'Predictor checkpoint is missing non-InternVL parameters:\n'
            + '\n'.join(invalid_missing_keys)
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            'Unexpected predictor checkpoint parameters:\n'
            + '\n'.join(incompatible.unexpected_keys)
        )

    if use_ema:
        _load_ema_weights(goal_model, predictor_dir)

    return goal_model


def _load_ema_weights(goal_model: MLLMGoalPredictor, predictor_dir: str):
    ema_weights_path = os.path.join(predictor_dir, 'ema_trainable.pt')
    if not os.path.exists(ema_weights_path):
        raise FileNotFoundError(f'EMA weights were requested but not found: {ema_weights_path}')

    try:
        ema_state_dict = torch.load(ema_weights_path, weights_only=True)
    except TypeError:
        ema_state_dict = torch.load(ema_weights_path)

    params = dict(goal_model.named_parameters())
    unknown_keys = [key for key in ema_state_dict if key not in params]
    if unknown_keys:
        raise RuntimeError('Unexpected EMA checkpoint parameters:\n' + '\n'.join(unknown_keys))

    with torch.no_grad():
        for key, value in ema_state_dict.items():
            # copy_ casts the fp32 EMA values to the param's dtype/device
            params[key].copy_(value)

    print(f'[Checkpoint] Loaded EMA weights for {len(ema_state_dict)} parameters from {ema_weights_path}')
