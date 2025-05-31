#!/usr/bin/env python3
import argparse
import jsonlines
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download abisee/cnn_dailymail (3.0.0) and save train split as JSONL (fields trimmed).")
    parser.add_argument('--output_file', required=True, help='Output JSONL file')
    args = parser.parse_args()

    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    with jsonlines.open(args.output_file, 'w') as writer:
        for item in tqdm(dataset, desc="Writing JSONL", unit="records"):
            writer.write({
                "id": item["id"].strip() if item["id"] else "",
                "article": item["article"].strip() if item["article"] else "",
                "highlights": item["highlights"].strip() if item["highlights"] else ""
            })

if __name__ == "__main__":
    main()
