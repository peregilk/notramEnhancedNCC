#!/usr/bin/env python3
import argparse
import json
import logging
import statistics

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL dialogues: count multiturns and compute discussion length stats."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the JSONL file containing one dialogue per line."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    lengths = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping line {i}: invalid JSON ({e})")
                continue

            conv = obj.get("conversations", [])
            length = len(conv)
            lengths.append(length)
            logging.debug(f"Line {i}: {length} messages")

    if not lengths:
        logging.info("No valid dialogues found in the file.")
        return

    total = len(lengths)
    multiturn = sum(1 for l in lengths if l > 2)

    print(f"Total dialogues: {total}")
    print(f"Multiturn dialogues (>2 messages): {multiturn} ({multiturn/total*100:.2f}%)")
    print("\nDiscussion length statistics (in messages):")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Mean:   {statistics.mean(lengths):.2f}")
    print(f"  Median: {statistics.median(lengths)}")

if __name__ == "__main__":
    main()
