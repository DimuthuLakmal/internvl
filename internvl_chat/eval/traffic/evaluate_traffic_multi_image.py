import argparse
import itertools
import json
import os
import re
import random
import time
from functools import partial
import jsonlines

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm


class TrafficDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        jsonl_path,
        image_root,
        input_size=224,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num=6,
    ):
        self.samples = list(jsonlines.open(jsonl_path))
        self.image_root = image_root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.samples)

    def multi_modal_multi_image_get_item(self, data_item):
        images, num_tiles = [], []
        num_image = len(data_item['image'])

        image_id = int(data_item["image"][0].split("/")[-1].split('.')[0].split("_")[0])

        for image_path in data_item['image']:
            # Merge the image path
            # image_path = self.get_image_path(image_path)
            image_path = self.image_root + image_path
            # Load the image using tcs_loader if available, otherwise use PIL
            image = Image.open(image_path).convert("RGB")
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, image_size=self.input_size,
                                           use_thumbnail=self.use_thumbnail,
                                           max_num=self.max_num)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values, image_id

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pixel_values, image_id = self.multi_modal_multi_image_get_item(sample)

        # image_path = os.path.join(self.image_root, sample["image"])
        # image = Image.open(image_path).convert("RGB")
        # image_id = int(sample["image"].split("/")[-1].split('.')[0])

        # if self.dynamic_image_size:
        #     images = dynamic_preprocess(image, image_size=self.input_size,
        #                                 use_thumbnail=self.use_thumbnail,
        #                                 max_num=self.max_num)
        # else:
        #     images = [image]
        # pixel_values = [self.transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)

        conversations = sample["conversations"]
        human_msg = conversations[-2]["value"]

        gt = conversations[-1]["value"]

        return {
            "image_id": image_id,
            "pixel_values": pixel_values,
            "question": human_msg,
            "gt": gt,
        }


def collate_fn(inputs):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['question'] for _ in inputs]
    gts = [_['gt'] for _ in inputs]
    # input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, input_texts, gts, image_ids


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def read_jsonl(file_path):
    """
    Reads a JSONL file and yields each line as a Python dictionary.
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip whitespace and check if line is not empty
            line = line.strip()
            if line:
                try:
                    # Parse the JSON string on the line into a Python object
                    json_object = json.loads(line)
                    data_list.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    return data_list


def extract_idx_from_images(images):
    """
    images: list of strings like ["/648779/648779_t0.png", ...]
    returns int idx = 648779
    """
    if not images:
        return None
    m = re.search(r"/(\d+)/", images[0])
    return int(m.group(1)) if m else None


def evaluate_chat_model(args, model, tokenizer):
    # Reading Ground Truth Data from ChatGPT
    idx_to_prediction={}
    data = read_jsonl(args.jsonl_gt)

    for obj in data:
        images = obj.get("image", [])
        idx = extract_idx_from_images(images)

        for msg in obj["conversations"]:
            if msg["from"] == "gpt":
                idx_to_prediction[idx] = msg["value"]


    random.seed(args.seed)
    summaries = []

    dataset = TrafficDataset(
        jsonl_path=args.jsonl,
        image_root=args.image_root,
        input_size=args.image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=model.config.use_thumbnail,
        max_num=args.max_num,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    results = []

    for i, (pixel_values, questions, gts, idxs) in tqdm(enumerate(dataloader)):
        # if i < 3001:
        #     continue

        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            output_hidden_states=True
        )

        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=questions[0],
            generation_config=generation_config,
            save_logits=True
        )

        # print(f"Prediction {idxs[0]}: " +  pred)
        # print(f"GT {idxs[0]}: " + gts[0])
        # print(f"ChatGPT Output: {idxs[0]}: " + idx_to_prediction[idxs[0]])
        # print('--------------')

        results.append({
            "idx": idxs[0],
            "prediction": pred[0],
            "ground_truth": gts[0],
            "chatgpt": idx_to_prediction[idxs[0]],
            # "logits": pred[1]
        })

        if i==50:
            break

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged, results)
    merged = [x for xs in merged for x in xs]

    if torch.distributed.get_rank() == 0:
        out_file = os.path.join(args.logits_path)
        json.dump(merged, open(out_file, "w"), indent=2)
        print(f"Saved evaluation results to {out_file}")


system_message = """You are a Vision Language Model specialized in interpreting map images and coordinate sequences. 
You are provided with coordinate sequences of vehicles in a tabular format.
The tabular data shows the movements of vehicles on this road segment captured on 8 time steps in a fixed frequency. Each row has both x and y coordinates recorded in each time step, in the format of x1,y1,x2,y2,...,x8,y8|. | represents the line break. 
The first line of the tabular data has the column names and second line has the ego vehicle movement whose future movements are going to be predicted based on these historical observations and the road segment. The remaining rows have the movements of other vehicles on the road. 
In some time steps, both x and y values are 0, and they are noises. 
You are additionally provided with four images corresponding to steps t=4,t=8,t=12, and t=16 of the observation sequence showing movements of vehicles on the map.
The map is a 56*56 three color images of a road segment from bird eye view. Yellow lines represent white lane lines in roads. 
Blue color area represent pavements and red area represents the road and sometimes, parking areas where vehicle can move. Sometimes vehicles are parked in black area.
Blue color dot represents the ego vehicle, and green dots represent other adjacent vehicles.
Don't consider them in the analysis. When overlaying the coordinates on the map to understand movements, make sure that (0,0) coordinate starts from the top left corner. 
Be very accurate when mention the directions and relative positions of vehicles. x values increase along the East, and y values increase along the South. You have to determine other directions accordingly.
Considering these information, You need to generate a text describing the movement of the vehicles on the road, context of covering constraints and prominent road features like intersections, bends"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--logits-path', type=str, default=None)
    parser.add_argument('--jsonl', type=str, default=None)
    parser.add_argument('--jsonl_gt', type=str, default=None)
    parser.add_argument('--image_root', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    args.system_message = system_message
    args.load_lora = False

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
    model.to('cuda')
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

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

    evaluate_chat_model(args, model, tokenizer)
