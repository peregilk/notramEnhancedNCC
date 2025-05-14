#!/usr/bin/env python3
import argparse
import json
import logging
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Add a 'text' field combining source and target")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_file", help="Path to output JSONL file (defaults to stdout)")
    return parser.parse_args()

def setup_logging():
    # Only show warnings and errors; suppress debug/info on success
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

def process_line(line):
    record = json.loads(line)
    src = record.get("source", "")
    tgt = record.get("target", "")
    record["text"] = f"source_text: {src}\\ntranslated_text: {tgt}"
    return record

def main():
    args = parse_args()
    setup_logging()
    infile = open(args.input_file, "r", encoding="utf-8")
    out = open(args.output_file, "w", encoding="utf-8") if args.output_file else sys.stdout

    for line in infile:
        line = line.strip()
        if not line:
            continue
        try:
            updated = process_line(line)
            json.dump(updated, out, ensure_ascii=False)
            out.write("\n")
        except Exception as e:
            logging.error(f"Failed to process line: {e}")

    if args.output_file:
        out.close()
    infile.close()

if __name__ == "__main__":
    main()
