import os
import json
import re
from glob import glob
import random

# -------- paths --------
jsonl_path = "/data/gpfs/projects/punim2030/mllm/data/all_data_internvl.jsonl"
pred_dir  = "eval/traffic/results"
output_path = "eval/traffic/results/all_internvl3_8b.jsonl"
# missing_data_path = "eval/traffic/results/missing.json"

# -------- helper to extract idx from image paths --------
def extract_idx_from_images(images):
    """
    images: list of strings like ["/648779/648779_t0.png", ...]
    returns int idx = 648779
    """
    if not images:
        return None
    m = re.search(r"/(\d+)/", images[0])
    return int(m.group(1)) if m else None

questions = {}
missing = []
matched = 0

with open(jsonl_path, "r") as fin:
    for line in fin:
        obj = json.loads(line)

        images = obj.get("image", [])
        idx = extract_idx_from_images(images)

        question_elements = {}
        for msg in obj["conversations"]:
            if msg["from"] == "system":
                question_elements["system"] = msg["value"].encode().decode("unicode_escape").encode("latin1").decode("utf8")
            if msg["from"] == "human":
                question_elements["human"] = msg["value"].encode().decode("unicode_escape").encode("latin1").decode("utf8")

        questions[idx] = question_elements

# -------- load second JSON into lookup dict --------
idx_to_prediction = []
total_items = 0
duplicate_idxs = 0
data_obj_list = []

json_files = sorted(glob(os.path.join(pred_dir, "*.json")))
if not json_files:
    raise FileNotFoundError(f"No .json files found in: {pred_dir}")

for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)

    # Expect: list[{"idx":..., "prediction":...}, ...]
    if isinstance(data, dict):
        # If a file stores a dict wrapper, try common keys
        for key in ("data", "items", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {jf}: expected a list (or wrapper containing a list).")

    for item in data:
        total_items += 1
        if "idx" not in item or "prediction" not in item:
            continue
        idx = item["idx"]
        rejected = item["prediction"]
        chosen = item["chatgpt"]
        idx_str = str(idx)
        images = ["/" + idx_str + "/" + idx_str + "_t0.png", 
                    "/" + idx_str + "/" + idx_str + "_t1.png", 
                    "/" + idx_str + "/" + idx_str + "_t2.png", 
                    "/" + idx_str + "/" + idx_str + "_t3.png"
                ]
        question = questions[idx]["system"] + "\n" + questions[idx]["human"]
        if idx in idx_to_prediction:
            duplicate_idxs += 1  # keep the first by default
            continue
        idx_to_prediction.append(idx)

        data_obj = {
            "image": images,
            "question": question,
            "chosen": chosen,
            "rejected": rejected

        }

        data_obj_list.append(data_obj)

print(f"Loaded {len(idx_to_prediction)} unique predictions from {len(json_files)} files "
      f"({total_items} total records, {duplicate_idxs} duplicate idxs ignored).")

# # -------- write dump missing data --------
# with open(missing_data_path, "w", encoding="utf-8") as fout:
#     fout.write(json.dumps(missing))

# -------- write new jsonl --------
with open(output_path, "w", encoding="utf-8") as fout:
    for row in data_obj_list:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


train_path = "/data/gpfs/projects/punim2030/mllm/data/train_internvl3_8b.jsonl"
test_path  = "/data/gpfs/projects/punim2030/mllm/data/val_internvl3_8b.jsonl"
train_ratio = 0.9
seed = 42

random.seed(seed)

# -------- load all jsonl --------
data = []
with open(output_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# -------- shuffle + split --------
random.shuffle(data)
n_train = int(len(data) * train_ratio)

train_data = data[:n_train]
test_data  = data[n_train:]

# -------- save --------
with open(train_path, "w", encoding="utf-8") as f:
    for obj in train_data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with open(test_path, "w", encoding="utf-8") as f:
    for obj in test_data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Total: {len(data)}")
print(f"Train: {len(train_data)}")
print(f"Test:  {len(test_data)}")        

print(f"Done!")
print(f"Matched & replaced: {matched}")
print(f"No matching prediction found: {missing}")
print(f"Saved to: {output_path}")
