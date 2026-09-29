from torch.utils.data import Dataset, DataLoader, DistributedSampler
import argparse
import itertools
import json
import os
import re
import random
import time
from functools import partial
import jsonlines
import pandas as pd
import numpy as np
from copy import deepcopy
from datasets import load_dataset
import pickle, gzip

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5
from PIL import Image
from tqdm import tqdm
import cv2

import torch

from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)

IGNORE_INDEX = -100


system_message = """You are a Vision Language Model specialized in interpreting map images and coordinate sequences. \nYou are provided with coordinate sequences of vehicles 
in a tabular format.\nThe tabular data shows the movements of vehicles on this road segment captured in 20 time steps. Each row has both x and y coordinates recorded in each 
time step, in the format of x1,y1,x2,y2,...,x20,y20|. | represents the line break. \nAfter the line with word 'Coordinates:', the ego vehicle movement whose future movements 
are going to be predicted based on these historical observations and the road segment. The remaining rows have the movements of other vehicles on the road. \nIn some time steps, 
both x and y values are 0, and they are noises. \nYou are additionally provided with four images corresponding to steps t=14,t=16,t=18, and t=20 of the observation sequence showing 
movements of vehicles on the map.\nThe map is a 224*224 three color images of a road segment from bird eye view. Yellow lines represent white lane lines in roads. \nBlue color area 
represent pavements and red area represents the road and sometimes, parking areas where vehicle can move. Sometimes vehicles are parked in black area.\nBlue color dot represents 
the ego vehicle, and green dots represent other adjacent vehicles.\nWhen overlaying the coordinates on the map to understand movements, make sure that (0,0) coordinate starts from 
the top left corner. \nConsidering these information, You need to estimate the x and y coordinates of the ego vehicle at t=72"""


def load_image(image_path):
    merged_image = Image.open(image_path).convert('RGB')
     # Get full merged image size
    merged_width, merged_height = merged_image.size

    # Since it is a 1 x 4 image
    single_width = merged_width // 4
    single_height = merged_height

    # Crop back into 4 original images
    images_t = []

    for t in range(4):
        left = t * single_width
        upper = 0
        right = (t + 1) * single_width
        lower = single_height

        image_t = merged_image.crop((left, upper, right, lower))
        images_t.append(image_t)

    return images_t


def get_pixel_values(transform, input_images, config):
        images, num_tiles = [], []
        num_image = len(input_images)

        for image in input_images:
            # image_path = config.image_root + image_path
            # image = Image.open(image_path).convert("RGB")

            if config.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, image_size=config.input_size,
                                           use_thumbnail=config.use_thumbnail,
                                           max_num=config.max_num)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values


def create_conversation(question):
    return [
            {
                "from": "system",
                "value": system_message
            },
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": "NA"
            }
        ]
    

class TrafficInferenceDataset(Dataset):
    def __init__(self, hf_dataset, args, transform, tokenizer):
        self.ds = hf_dataset
        self.args = args
        self.transform = transform
        self.tokenizer = tokenizer
        self.num_image_token = args.num_image_token
        self.group_by_length = args.group_by_length
        self.use_packed_ds = args.use_packed_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        image_name = row["image"][0]
        # image_idx = int(image_name.split(".")[0])
        image_path = f"/data/gpfs/projects/punim2030/mllm/data/internvl_maps/{image_name}"
        human_value = row["conversations"][1]["value"]
        movement_data_str = human_value.split("Coordinates:\n")[-1]

        images = load_image(image_path)
        pixel_values = get_pixel_values(self.transform, images, self.args)
        num_patches = pixel_values.size(0)

        question = (
            't14:<image>\n t16:<image>\n t18:<image>\n t20:<image>\n'
            '\nCoordinates:\n'
            + movement_data_str
        )

        return {
            # "image_id": image_idx,
            "pixel_values": pixel_values,
            "question": question
        }


def collate_fn(inputs):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    # image_ids = [_['image_id'] for _ in inputs]
    questions = [_['question'] for _ in inputs]
    # input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, questions
    

@torch.no_grad()
def evaluate_hidden_states(args, model, tokenizer, transform):
    file_dict = {
        "train": "/data/gpfs/projects/punim2030/mllm/data/train_internvl_target_pred_filtered_y.jsonl",
    }

    ds = load_dataset(
        "json",
        data_files=file_dict,
        cache_dir="/data/gpfs/projects/punim2030/huggingface/cache",
    )

    dataset = TrafficInferenceDataset(ds["train"], args, transform, tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=False,
    )

    loader = DataLoader(
        dataset,
        # batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=1,
    )

    rank = torch.distributed.get_rank()
    results = {}

    model.eval()

    b_last_hidden_states = {}
    for i, batch in enumerate(tqdm(loader)):
        if i >= 100:
            break

        pixel_values = batch[0].to(torch.bfloat16).to(model.device)
        questions = batch[1]
        # image_ids = batch[2]

        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            return_dict_in_generate=True,
            output_logits=False,
            output_scores=False,
            output_hidden_states=True
        )

        with torch.inference_mode():
            pred, b_hidden_states, b_seq_end_lengths = model.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                questions=questions,
                generation_config=generation_config,
                # save_logits=True
            )

            for i in range(len(questions)):
                q_hidden_states = b_hidden_states[:b_seq_end_lengths[i]]
                q_last_hidden_states = torch.stack([w_hidden_states[-1][i, -1, :] for w_hidden_states in q_hidden_states]).to(torch.float16).cpu().numpy()
                # b_last_hidden_states[image_ids[i]] = q_last_hidden_states

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged, b_last_hidden_states)

    if torch.distributed.get_rank() == 0:
        out_file = os.path.join(args.out_dir, "hidden_states_0_100.pkl.gz")
        with gzip.open(out_file, "wb", compresslevel=9) as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Saved evaluation results to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--down_sample_ratio', type=int, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--group_by_length', action='store_true')
    parser.add_argument('--use_packed_ds', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    args.system_message = system_message
    args.load_lora = False
    args.dynamic_image_size = args.dynamic
    args.input_size = args.image_size

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.to('cuda')

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    args.use_thumbnail = model.config.use_thumbnail
    args.num_image_token = int((image_size // model.config.vision_config.patch_size) ** 2 * (args.down_sample_ratio ** 2))

    transform = build_transform(is_train=False, input_size=image_size)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_hidden_states(args, model, tokenizer, transform)