#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract first annotated line from each JSONL file into a CSV.")
    parser.add_argument("--input_dir", required=True, help="Directory with .jsonl files to scan.")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file.")
    parser.add_argument("--debug", action="store_true", help="Print debug info.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    assert input_dir.exists() and input_dir.is_dir(), f"Not a valid input dir: {input_dir}"

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    print(f"Matched {len(jsonl_files)} files:")
    for f in jsonl_files:
        print(f" - {f}")

    total_written = 0

    with open(args.output_csv, 'w', encoding='utf-8', newline='') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=["filename", "text", "language", "language_confidence"])
        writer.writeheader()

        for filepath in tqdm(jsonl_files, desc="Processing files"):
            matched = False
            line_no = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_no += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if args.debug:
                            print(f"[WARN] Invalid JSON in {filepath.name} line {line_no}")
                        continue

                    if "language_confidence" in record:
                        writer.writerow({
                            "filename": filepath.name,
                            "text": record.get("text", "").strip().replace("\n", " "),
                            "language": record.get("language", ""),
                            "language_confidence": record["language_confidence"]
                        })
                        total_written += 1
                        matched = True
                        if args.debug:
                            print(f"[DEBUG] Matched line {line_no} in {filepath.name}")
                        break

            if not matched and args.debug:
                print(f"[INFO] No match found in {filepath.name}")

    print(f"Done. Wrote {total_written} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
