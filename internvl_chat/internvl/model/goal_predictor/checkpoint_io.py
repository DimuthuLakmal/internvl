import json
import os

import torch
from transformers import AutoTokenizer

from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.goal_predictor.mllm_spatial_goal_predictor import MLLMGoalPredictor


def load_llm_checkpoint(
    checkpoint_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    system_message: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    load_tokenizer: bool = True,
    use_fast_tokenizer: bool = False,
):
    """
    Load the InternVL component of a component-checkpoint saved by
    TargetSFTTrainer (see internvl/train/target_trainer.py).

    Mode (train/eval) is left to the caller: this does not call .eval().
    """
    internvl_dir = os.path.join(checkpoint_dir, 'internvl_model')

    internvl_model = InternVLChatModel.from_pretrained(
        internvl_dir,
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
):
    """
    Reconstruct MLLMGoalPredictor from the predictor-only weights and config
    saved by TargetSFTTrainer._save alongside the InternVL checkpoint.
    """
    predictor_dir = os.path.join(checkpoint_dir, 'goal_predictor')
    predictor_config_path = os.path.join(predictor_dir, 'goal_predictor_config.json')
    predictor_weights_path = os.path.join(predictor_dir, 'goal_predictor_weights.pt')

    for path in (predictor_config_path, predictor_weights_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Required checkpoint component was not found: {path}')

    with open(predictor_config_path, 'r', encoding='utf-8') as file:
        predictor_config = json.load(file)

    goal_model = MLLMGoalPredictor(
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

    return goal_model
