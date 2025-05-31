#!/usr/bin/env python3
import argparse
import jsonlines
from datasets import load_dataset
from tqdm import tqdm

def is_single_sentence(text):
    return text.count('.') == 1

def main():
    parser = argparse.ArgumentParser(description="Download pere/wiki_paragraphs_norwegian and save single-sentence texts as JSONL.")
    parser.add_argument('--output_file', required=True, help='Output JSONL file')
    args = parser.parse_args()

    dataset = load_dataset("pere/wiki_paragraphs_norwegian", split="train")
    with jsonlines.open(args.output_file, 'w') as writer:
        for item in tqdm(dataset, desc="Filtering and writing", unit="records"):
            text = item.get("text", "")
            if is_single_sentence(text):
                writer.write({"text": text.strip()})

if __name__ == "__main__":
    main()
