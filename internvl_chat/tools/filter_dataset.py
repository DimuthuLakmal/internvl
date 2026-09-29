from pathlib import Path
import random

input_path = Path("/data/gpfs/projects/punim2030/mllm/data/train_internvl_target_pred.jsonl")
output_path = Path("/data/gpfs/projects/punim2030/mllm/data/train_internvl_target_pred_filtered_y.jsonl")

random.seed(42)

target_text = (
    "Coordinates:\\n"
    "56,112,56,112,56,112,56,112,56,112,56,112,56,112,56,112,"
    "56,112,56,112,56,112,56,112,56,112,56,112,56,112,56,112,"
    "56,112,56,112,56,112,56,112|"
)

matching_seen = 0
matching_kept = 0
non_matching_kept = 0
total_count = 0

with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        y_target = int(line.split("\"")[-2].split(",")[-1])
        if y_target > 109 and y_target < 116 and target_text not in line:
            continue
        if target_text in line:
            matching_seen += 1

            if random.random() < 0.05:
                fout.write(line)
                matching_kept += 1
        else:
            fout.write(line)
            non_matching_kept += 1
        total_count += 1

print(f"Total lines found: {total_count}")
print(f"Matching lines found: {matching_seen}")
print(f"Matching lines kept: {matching_kept}")
print(f"Non-matching lines kept: {non_matching_kept}")
print(f"Total lines written: {matching_kept + non_matching_kept}")
print(f"Saved to: {output_path}")