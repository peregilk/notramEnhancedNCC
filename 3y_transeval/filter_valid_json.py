#!/usr/bin/env python3

import json
import argparse

def filter_valid_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                obj = json.loads(line)
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write('\n')
            except Exception:
                continue  # skip invalid lines

def main():
    parser = argparse.ArgumentParser(description='Filter only valid JSON lines from a JSONL file.')
    parser.add_argument('--input_file', required=True, help='Input JSONL file')
    parser.add_argument('--output_file', required=True, help='Output JSONL file with only valid JSON objects')
    args = parser.parse_args()

    filter_valid_jsonl(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
