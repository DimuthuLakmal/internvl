import os

import torch
import torch.distributed as dist
from deepspeed.runtime.zero import GatheredParameters
from peft import PeftModel

from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.model.goal_predictor.checkpoint_io import load_llm_checkpoint


def load_and_configure_llm_model(model_args, data_args, tokenizer, num_new_tokens, img_context_token_id, logger):
    """
    Load InternVLChatModel per model_args/data_args, apply the LoRA/config
    overrides, resize position embeddings and token embeddings to match the
    tokenizer, and set up gradient checkpointing.
    """
    if model_args.model_name_or_path is None:
        logger.error('No model path provided.')
        raise ValueError('No model path provided.')

    logger.info('Loading InternVLChatModel...')

    # If loaded from local path
    config_path = model_args.model_name_or_path
    if model_args.load_checkpoint:
        config_path = os.path.join(model_args.model_name_or_path, 'internvl_model')

    config = InternVLChatConfig.from_pretrained(config_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        logger.info('Using flash_attention_2 for InternLM')
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        logger.info('Using flash_attention_2 for LLaMA')
    config.template = data_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch

    if model_args.load_checkpoint:
        llm_model = load_llm_checkpoint(model_args.model_name_or_path, load_tokenizer=False)
    else:
        llm_model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config,
            cache_dir='/data/gpfs/projects/punim2030/huggingface/cache')
        if model_args.load_lora_adapters:
            llm_model.language_model = PeftModel.from_pretrained(
                llm_model.language_model,
                model_args.model_name_or_path + '/llm_lora',
                is_trainable=True,
            )
            llm_model.vision_model = PeftModel.from_pretrained(
                llm_model.vision_model,
                model_args.model_name_or_path + '/vision_lora',
                is_trainable=True,
            )
            llm_model.train()
            llm_model.language_model.train()
            llm_model.vision_model.train()
            llm_model.language_model.enable_input_require_grads()

    llm_model.img_context_token_id = img_context_token_id

    assert llm_model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = llm_model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = llm_model.config.vision_config.patch_size
    logger.info(f'llm_model.config.force_image_size: {llm_model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'llm_model.config.vision_config.image_size: {llm_model.config.vision_config.image_size}')
    if llm_model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{llm_model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        params = [llm_model.vision_model.embeddings.position_embedding]
        with GatheredParameters(params, modifier_rank=None):
            llm_model.vision_model.resize_pos_embeddings(old_size=llm_model.config.vision_config.image_size,
                                                    new_size=data_args.force_image_size,
                                                    patch_size=patch_size)
        llm_model.config.vision_config.image_size = data_args.force_image_size
    llm_model.config.force_image_size = data_args.force_image_size
    llm_model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        llm_model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = llm_model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        llm_model.config.llm_config.vocab_size = len(tokenizer)
        llm_model.language_model.config.vocab_size = len(tokenizer)

    llm_model.language_model.config.use_cache = False
    llm_model.vision_model.gradient_checkpointing = True
    llm_model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        llm_model.language_model._set_gradient_checkpointing()

    return llm_model


def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


def freeze_model_components(llm_model, model_args, logger):
    """
    Freeze/unfreeze InternVLChatModel components (backbone, LLM, MLP, LoRA
    adapters, ViT layers) per model_args, then log the trainable parameters.
    """
    if model_args.freeze_backbone:
        _freeze_params(llm_model.vision_model)

    if model_args.freeze_llm:
        llm_model.language_model = llm_model.language_model.eval()
        _freeze_params(llm_model.language_model)

    if model_args.unfreeze_lm_head:
        llm_model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        llm_model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        llm_model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        llm_model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        llm_model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(llm_model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = llm_model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in llm_model.named_parameters():
            if param.requires_grad:
                logger.info(name)
