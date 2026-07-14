# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math

import torch
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
from peft import PeftModel


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_model_and_tokenizer(args):
    if args.auto:
        config = InternVLChatConfig.from_pretrained(args.base_path if args.load_lora else args.checkpoint)
        num_hidden_layers = config.llm_config.num_hidden_layers
        # device_map = split_model('InternVL2_5-26B')
        device_map = 'auto'
    kwargs = {
        # 'device_map': device_map,
        'system_message': args.system_message
        } if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    # model = InternVLChatModel.from_pretrained(
    #     args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    #     load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if args.load_lora:
        model = InternVLChatModel.from_pretrained(args.base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, 
        cache_dir="/data/gpfs/projects/punim2030/huggingface/cache", **kwargs)
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.checkpoint + "/llm_lora",
        )
        model.vision_model = PeftModel.from_pretrained(
            model.vision_model,
            args.checkpoint + "/vision_lora",
        )
    else:
        model = InternVLChatModel.from_pretrained(
                    args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                    load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    return model, tokenizer
