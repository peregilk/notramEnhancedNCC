#!/usr/bin/env python3

import argparse
import json
import fasttext
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import re

# Regular expressions for tag-based extraction
USER_PATTERN = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL
)
ASSISTANT_PATTERN = re.compile(
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL
)

def extract_clean_text(text: str) -> str:
    if "<|start_header_id|>user<|end_header_id|>" in text and "<|start_header_id|>assistant<|end_header_id|>" in text:
        user_match = USER_PATTERN.search(text)
        assistant_match = ASSISTANT_PATTERN.search(text)
        user_text = user_match.group(1).strip() if user_match else ""
        assistant_text = assistant_match.group(1).strip() if assistant_match else ""
        return f"{user_text}\n{assistant_text}".strip()
    else:
        return text.strip()

def clean_text(text: str) -> str:
    return ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())

def detect_languages(model, texts):
    cleaned_texts = [clean_text(extract_clean_text(text)) for text in texts]
    preds, confs = model.predict(cleaned_texts)
    return [
        (label[0].replace("__label__", ""), float(conf[0]))
        for label, conf in zip(preds, confs)
    ]

def process_file(input_file, output_file, model, batch_size=100):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    records = [json.loads(line) for line in lines]
    all_texts = [rec.get("text", "") for rec in records]

    output_lines = []

    for i in tqdm(range(0, len(records), batch_size), desc="Processing lines"):
        batch = all_texts[i:i + batch_size]
        batch_preds = detect_languages(model, batch)
        for rec, (lang, conf) in zip(records[i:i + batch_size], batch_preds):
            rec["language"] = lang
            rec["language_confidence"] = conf
            output_lines.append(json.dumps(rec, ensure_ascii=False))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(output_lines) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Annotate a JSONL file with GlotLID language predictions.")
    parser.add_argument('--input_file', required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_file', required=True, help="Path to the output JSONL file.")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for fastText predictions.")
    args = parser.parse_args()

    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    model = fasttext.load_model(model_path)

    process_file(args.input_file, args.output_file, model, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
