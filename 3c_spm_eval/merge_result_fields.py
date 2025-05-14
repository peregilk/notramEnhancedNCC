#!/usr/bin/env python3
import argparse
import json
from tqdm import tqdm


def load_jsonlines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_file", required=True, help="JSONL file with original result field")
    parser.add_argument("--eval_file", required=True, help="JSONL file with eval results")
    parser.add_argument("--output_file", required=True, help="Path to write merged JSONL")
    parser.add_argument("--id_prefix", default="train_edu2_ling1_no_clean_spm_merged", help="Prefix for new ID")
    args = parser.parse_args()

    original_data = load_jsonlines(args.original_file)
    eval_data = load_jsonlines(args.eval_file)

    # Build lookup from text → result, only first occurrence is used
    text_to_result = {}
    for entry in original_data:
        text = entry["text"]
        if text not in text_to_result:
            text_to_result[text] = entry["result"]

    unmatched = 0
    merged = []
    for idx, eval_entry in enumerate(tqdm(eval_data, desc="Merging on text")):
        text = eval_entry["text"]
        if text not in text_to_result:
            unmatched += 1
            continue
        new_entry = dict(eval_entry)  # Copy to avoid modifying in-place
        new_entry["old_result"] = text_to_result[text]
        new_entry["id"] = f"{args.id_prefix}{idx}"
        merged.append(new_entry)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Merged {len(merged)} entries. Skipped {unmatched} unmatched eval entries.")


if __name__ == "__main__":
    main()
