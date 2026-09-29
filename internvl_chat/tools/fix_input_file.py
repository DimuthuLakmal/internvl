import json
from pathlib import Path


def filter_jsonl_by_image(
    input_file: str,
    output_file: str,
    target: str = "merged.png",
) -> None:
    """
    Keep records where:
      1. The "image" value is a non-empty list.
      2. Every element is a string.
      3. Every string contains the target text.

    Records are read and written one at a time to minimise memory usage.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    total_count = 0
    kept_count = 0
    invalid_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as source, \
            output_path.open("w", encoding="utf-8") as destination:

        for line_number, line in enumerate(source, start=1):
            line = line.strip()

            if not line:
                continue

            total_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                invalid_count += 1
                print(
                    f"[WARNING] Invalid JSON at line {line_number}: {error}"
                )
                continue

            images = record.get("image")

            is_valid = (
                isinstance(images, list)
                and len(images) > 0
                and all(
                    isinstance(image, str) and target in image
                    for image in images
                )
            )

            if is_valid:
                destination.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                kept_count += 1

            if total_count % 100_000 == 0:
                print(
                    f"Processed: {total_count:,} | "
                    f"Kept: {kept_count:,} | "
                    f"Invalid: {invalid_count:,}"
                )

    print("\nFiltering completed.")
    print(f"Processed records : {total_count:,}")
    print(f"Kept records      : {kept_count:,}")
    print(f"Invalid JSON lines: {invalid_count:,}")
    print(f"Output file       : {output_path}")


if __name__ == "__main__":
    filter_jsonl_by_image(
        input_file="/data/gpfs/projects/punim2030/mllm/data/train_internvl_target_pred.jsonl",
        output_file="/data/gpfs/projects/punim2030/mllm/data/train_internvl_target_pred_fixed.jsonl",
    )