from torch.utils.data import Dataset, DataLoader, DistributedSampler
import argparse
import itertools
import json
import os
import re
import random
import time
import traceback
from contextlib import nullcontext
from functools import partial
import jsonlines
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle, gzip

import torch
import torch.profiler
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5_target
from internvl.patch import concat_pad_data_collator
from PIL import Image
from tqdm import tqdm
import cv2

from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)

IGNORE_INDEX = -100
from eval.traffic.utils.load_checkpoints import load_llm_checkpoint, load_goal_predictor_checkpoint


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


class TrafficGoalPredictorDataset(Dataset):
    """
    Inference-only dataset for the goal predictor. Mirrors the tensor layout
    produced by TargetGoalDataset.multi_modal_multi_image_get_item (see
    internvl/train/dataset.py) during training (input_ids/labels/attention_mask/
    position_ids/raw_labels/pixel_values/image_flags), since
    MLLMGoalPredictor.forward feeds that dict straight into the InternVL
    model's forward() rather than model.chat().

    Unlike the training dataset, this does a single, un-sharded pass over the
    annotation file (rank/world-size splitting is left to the DataLoader's
    DistributedSampler) and never substitutes a random sample on failure --
    a bad row is logged and skipped so predictions stay attributable to a
    known sample index.
    """

    def __init__(self, annotation_path, image_root, args, transform, tokenizer, template_name):
        with open(annotation_path, 'r') as f:
            self.raw_data = f.readlines()
        self.image_root = image_root
        self.args = args
        self.transform = transform
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = args.num_image_token

    def __len__(self):
        return len(self.raw_data)

    def _build_item(self, data_item):
        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for i, image_path in enumerate(data_item['image']):
            full_path = self.image_root + image_path

            # full_path='/home/dkariyawasan/projects/digi-twin/CTG/merged.png'

            image = load_image(full_path)[i]
            if self.args.dynamic_image_size:
                tiles = dynamic_preprocess(image, image_size=self.args.input_size,
                                           use_thumbnail=self.args.use_thumbnail,
                                           max_num=self.args.max_num)
                images += tiles
                num_tiles.append(len(tiles))
            else:
                images.append(image)
                num_tiles.append(1)

        pixel_values = torch.stack([self.transform(image) for image in images])
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        # data_item['conversations'][1]['value'] = 't14:<image>\n t16:<image>\n t18:<image>\n t20:<image>\nCoordinates:\n30,112,31,112,32,112,33,112,35,112,36,112,38,112,39,112,41,112,42,112,43,112,45,112,46,112,48,112,49,112,50,112,52,112,53,112,54,112,56,112|34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132,34,132|29,133,29,133,29,133,29,133,29,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133,28,133|24,133,24,133,24,133,24,133,24,133,24,133,24,133,24,133,24,133,24,133,23,133,23,133,23,133,23,133,23,133,23,133,23,133,23,133,23,133,23,133|13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132,13,132|1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131,1,131|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131,141,131|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,147,132,147,132|152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131,152,131|157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132,157,132|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,132,168,132|0,0,0,0,0,0,175,131,175,131,175,131,175,131,175,131,174,131,174,131,174,131,174,131,174,131,174,131,174,131,174,131,174,131,174,131,174,131,174,131|180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,131,180,130,180,130,180,130,180,130,180,130,180,130|0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0|'

        ret = preprocess_internvl2_5_target(
            self.template_name, [deepcopy(data_item['conversations'])],
            self.tokenizer, num_image_tokens,
            group_by_length=False, use_packed_ds=False,
            ds_name='traffic_eval', num_image=num_image,
        )

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, \
            'image tokens are truncated for the traffic_eval dataset'

        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        # print(data_item['conversations'])

        return dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            raw_labels=ret['raw_labels'][0],
            conversation=data_item['conversations'][1]['value'].split("Coordinates")[-1].split("|")[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
        )

    def __getitem__(self, i):
        try:
            data_item = json.loads(self.raw_data[i])
            item = self._build_item(data_item)
            item['index'] = i
            return item
        except Exception as e:
            print(f'[traffic_eval] skipping sample {i}: {e}', flush=True)
            traceback.print_exc()
            return None


def collate_fn(features):
    features = [f for f in features if f is not None]
    if not features:
        return None
    pad_content = concat_pad_data_collator(features)
    pad_content['conversation'] = [f["conversation"] for f in features]
    return pad_content


def build_profiler(args):
    """Torch profiler with a wait/warmup/active schedule, exporting traces in
    TensorBoard format (one subdir per rank, since inference runs under
    torch.distributed)."""
    rank = torch.distributed.get_rank()
    trace_dir = os.path.join(args.profile_dir, f'rank{rank}')
    os.makedirs(trace_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=args.profile_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


@torch.no_grad()
def run_goal_predictor_inference(args, goal_model, tokenizer, transform):
    annotation_path = args.jsonl or "/data/gpfs/projects/punim2030/mllm/data/train_internvl_target_pred.jsonl"
    image_root = args.image_root or "/data/gpfs/projects/punim2030/mllm/data/internvl_maps/"
    template_name = goal_model.llm_model.config.template

    dataset = TrafficGoalPredictorDataset(annotation_path, image_root, args, transform, tokenizer, template_name)
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    goal_model.eval()

    # index -> predicted (x, y) in pixel space
    predictions = {}
    device = goal_model.llm_model.device
    sum_abs_error = torch.zeros(2, dtype=torch.float64, device=device)
    sum_l2_error = torch.zeros(1, dtype=torch.float64, device=device)
    num_samples = torch.zeros(1, dtype=torch.float64, device=device)

    rank = torch.distributed.get_rank()
    log_path = os.path.join(args.out_dir, f'predictions_rank{rank}.jsonl')
    log_file = open(log_path, 'a')

    profiler_ctx = build_profiler(args) if args.profile else nullcontext()
    with profiler_ctx as prof:
        for batch in tqdm(loader):
            if batch is None:
                continue

            indices = batch.pop('index').tolist()
            target_xy = batch['raw_labels']
            batch.pop('labels', None)
            batch.pop('raw_labels', None)
            conversation = batch.pop('conversation', None)

            inputs = {k: v.to(goal_model.llm_model.device) for k, v in batch.items()}

            with torch.inference_mode(), torch.profiler.record_function('forward'):
                pred_xy_norm = goal_model(inputs)

            pred_xy_px = (pred_xy_norm + 1.0) * 0.5 * (224 - 1)
            target_xy = target_xy.to(pred_xy_px.device).float()

            abs_error = torch.abs(pred_xy_px - target_xy)  # [B, 2], L1 per axis
            l2_error = torch.norm(pred_xy_px - target_xy, dim=-1)  # [B], L2 per sample

            sum_abs_error += abs_error.sum(dim=0).double()
            sum_l2_error += l2_error.sum().double()
            num_samples += abs_error.size(0)

            pred_xy_cpu = pred_xy_px.detach().cpu()
            target_xy_cpu = target_xy.detach().cpu()
            for j, idx in enumerate(indices):
                log_file.write(json.dumps({
                    'index': idx,
                    'pred_xy': pred_xy_cpu[j].tolist(),
                    'target_xy': target_xy_cpu[j].tolist(),
                    'conversation': conversation[j] if conversation is not None else None,
                }) + '\n')
                predictions[idx] = pred_xy_px[j].float().cpu()
            log_file.flush()

            if prof is not None:
                prof.step()

    log_file.close()

    if torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(sum_abs_error, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sum_l2_error, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_samples, op=torch.distributed.ReduceOp.SUM)

    if rank == 0 and num_samples.item() > 0:
        mean_abs_error = (sum_abs_error / num_samples).cpu().tolist()
        mean_l1 = sum(mean_abs_error) / 2.0
        mean_l2 = (sum_l2_error / num_samples).item()
        metrics = {
            'num_samples': int(num_samples.item()),
            'l1_x_px': mean_abs_error[0],
            'l1_y_px': mean_abs_error[1],
            'l1_px': mean_l1,
            'l2_px': mean_l2,
        }
        print(f"[goal_predictor eval] {metrics}")
        with open(os.path.join(args.out_dir, 'error_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    return predictions


system_message = """You are a Vision Language Model specialized in interpreting map images and coordinate sequences. \nYou are provided with coordinate sequences of vehicles 
in a tabular format.\nThe tabular data shows the movements of vehicles on this road segment captured in 20 time steps. Each row has both x and y coordinates recorded in each 
time step, in the format of x1,y1,x2,y2,...,x20,y20|. | represents the line break. \nAfter the line with word 'Coordinates:', the ego vehicle movement whose future movements 
are going to be predicted based on these historical observations and the road segment. The remaining rows have the movements of other vehicles on the road. \nIn some time steps, 
both x and y values are 0, and they are noises. \nYou are additionally provided with four images corresponding to steps t=14,t=16,t=18, and t=20 of the observation sequence showing 
movements of vehicles on the map.\nThe map is a 224*224 three color images of a road segment from bird eye view. Yellow lines represent white lane lines in roads. \nBlue color area 
represent pavements and red area represents the road and sometimes, parking areas where vehicle can move. Sometimes vehicles are parked in black area.\nBlue color dot represents 
the ego vehicle, and green dots represent other adjacent vehicles.\nWhen overlaying the coordinates on the map to understand movements, make sure that (0,0) coordinate starts from 
the top left corner. \nConsidering these information, You need to estimate the x and y coordinates of the ego vehicle at t=72"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--logits-path', type=str, default=None)
    parser.add_argument('--jsonl', type=str, default=None)
    parser.add_argument('--jsonl_gt', type=str, default=None)
    parser.add_argument('--image_root', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--down_sample_ratio', type=int, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--group_by_length', action='store_true')
    parser.add_argument('--use_packed_ds', action='store_true')
    parser.add_argument('--profile', action='store_true', help='enable torch profiler, traces viewable in TensorBoard')
    parser.add_argument('--profile-dir', type=str, default='eval/traffic/profiler_logs')
    parser.add_argument('--profile-wait', type=int, default=5)
    parser.add_argument('--profile-warmup', type=int, default=5)
    parser.add_argument('--profile-active', type=int, default=5)
    parser.add_argument('--profile-repeat', type=int, default=1)
    args = parser.parse_args()

    args.system_message = system_message
    args.dynamic_image_size = args.dynamic
    args.input_size = args.image_size
    args.load_lora = False

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # llm_model, tokenizer = load_model_and_tokenizer(args)
    llm_model, tokenizer = load_llm_checkpoint(args.checkpoint, system_message=system_message)
    image_size = llm_model.config.force_image_size or llm_model.config.vision_config.image_size
    use_thumbnail = llm_model.config.use_thumbnail

    llm_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    llm_model.to('cuda')

    image_size = llm_model.config.force_image_size or llm_model.config.vision_config.image_size
    use_thumbnail = llm_model.config.use_thumbnail
    args.use_thumbnail = llm_model.config.use_thumbnail
    args.num_image_token = int((image_size // llm_model.config.vision_config.patch_size) ** 2 * (args.down_sample_ratio ** 2))

    transform = build_transform(is_train=False, input_size=image_size)

    total_params = sum(p.numel() for p in llm_model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {llm_model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    goal_model = load_goal_predictor_checkpoint(args.checkpoint, internvl_model=llm_model, tokenizer=tokenizer)
    goal_model.to('cuda')

    run_goal_predictor_inference(args, goal_model, tokenizer, transform)
