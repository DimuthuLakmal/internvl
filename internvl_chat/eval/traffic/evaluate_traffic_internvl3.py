import os
import json
import math
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

def load_image_tiles(image_path, input_size=448, max_num=12):
    img = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values  # [num_tiles, 3, H, W]

def build_question(conversations):
    # conversations: [system, human, gpt]
    system = conversations[0]["value"].strip()
    human  = conversations[1]["value"].strip()
    # Keep system+human; InternVL3 can accept long prompts, but you may later trim system if needed.
    return system + "\n\n" + human

@torch.inference_mode()
def main(
    in_jsonl: str,
    out_jsonl: str,
    image_root: str,
    model_path: str = "OpenGVLab/InternVL3-8B",
    input_size: int = 448,
    max_tiles: int = 12,
    max_new_tokens: int = 512,
):
    # device_map = split_model(model_path)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map='auto',
        cache_dir="/data/gpfs/projects/punim2030/huggingface/cache"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, cache_dir="/data/gpfs/projects/punim2030/huggingface/cache")

    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,          # deterministic for dataset building
        temperature=0.0,
    )

    with open(in_jsonl, "r") as f_in, open(out_jsonl, "w") as f_out:
        for idx, line in enumerate(f_in):
            obj = json.loads(line)

            # --- images (4) ---
            img_paths = obj["image"]
            full_paths = [os.path.join(image_root, p.lstrip("/")) for p in img_paths]

            pixel_values_list = []
            num_patches_list = []
            for p in full_paths:
                pv = load_image_tiles(p, input_size=input_size, max_num=max_tiles)
                num_patches_list.append(pv.size(0))
                pixel_values_list.append(pv)

            pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()

            # --- prompt ---
            question = build_question(obj["conversations"])

            # --- inference ---
            resp = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
            )

            # existing label (your current gpt output)
            label = obj["conversations"][2]["value"].strip()

            # --- DPO output ---
            dpo_obj = {
                "image": obj["image"],
                "question": question,
                "chosen": label,
                "rejected": resp.strip(),
            }
            f_out.write(json.dumps(dpo_obj, ensure_ascii=False) + "\n")

            if idx % 20 == 0:
                print(f"[{idx}] wrote sample. patches={num_patches_list} gen_len={len(resp)}", flush=True)

if __name__ == "__main__":
    # Example usage:
    # python build_dpo_from_internvl3.py \
    #   --in_jsonl train.jsonl --out_jsonl dpo.jsonl --image_root /data/maps

    main(
        in_jsonl="/data/gpfs/projects/punim2030/mllm/data/train_internvl_1b.jsonl",
        out_jsonl="/data/gpfs/projects/punim2030/mllm/data/train_internvl_3_8b.jsonl",
        image_root="/data/gpfs/projects/punim2030/mllm/data/internvl_maps",
        model_path="OpenGVLab/InternVL3-8B-Instruct",
        input_size=56,
        max_tiles=1,
        max_new_tokens=16384,
    )
