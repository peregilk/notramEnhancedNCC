#!/usr/bin/env python3

import argparse
import json
import fasttext
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from huggingface_hub import hf_hub_download
from pathlib import Path
import re

model = None  # global for worker

USER_PATTERN = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL
)
ASSISTANT_PATTERN = re.compile(
    r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL
)

def init_model():
    global model
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    model = fasttext.load_model(model_path)

def extract_clean_text(text: str) -> str:
    if "<|start_header_id|>user<|end_header_id|>" in text and "<|start_header_id|>assistant<|end_header_id|>" in text:
        user_match = USER_PATTERN.search(text)
        assistant_match = ASSISTANT_PATTERN.search(text)
        user_text = user_match.group(1).strip() if user_match else ""
        assistant_text = assistant_match.group(1).strip() if assistant_match else ""
        return f"{user_text}\n{assistant_text}".strip()
    else:
        return text.strip()

def clean_text(text):
    return ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())

def process_batch(lines):
    global model
    records = [json.loads(line) for line in lines]

    texts = [clean_text(extract_clean_text(rec.get("text", ""))) for rec in records]
    preds, confs = model.predict(texts)

    for rec, label, conf in zip(records, preds, confs):
        if "text" in rec:
            rec["language"] = label[0].replace("__label__", "")
            rec["language_confidence"] = float(conf[0])
    return [json.dumps(rec, ensure_ascii=False) for rec in records]

def chunked_iterable(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def process_file_parallel(input_path: Path, output_path: Path, batch_size=100):
    with input_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with Pool(processes=cpu_count(), initializer=init_model) as pool:
        chunks = list(chunked_iterable(lines, batch_size))
        with output_path.open("w", encoding="utf-8") as outfile:
            for processed_batch in tqdm(pool.imap(process_batch, chunks), total=len(chunks), desc=f"Processing {input_path.name}"):
                outfile.write('\n'.join(processed_batch) + '\n')

def process_all_files(input_dir: Path, output_dir: Path, batch_size=100):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    for in_file in jsonl_files:
        out_file = output_dir / in_file.name
        process_file_parallel(in_file, out_file, batch_size=batch_size)

def main():
    parser = argparse.ArgumentParser(description="Annotate all JSONL files in a directory with GlotLID language info.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .jsonl files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .jsonl files.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for fastText predictions.")
    args = parser.parse_args()

    process_all_files(Path(args.input_dir), Path(args.output_dir), batch_size=args.batch_size)

if __name__ == "__main__":
    main()
