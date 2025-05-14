#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from tqdm import tqdm


def extract_result_json(result_str):
    """
    Extract and parse the JSON block from the 'result' field.
    Assumes it's wrapped in triple backticks and possibly followed by explanations.
    Returns a dict or None.
    """
    if not isinstance(result_str, str):
        return None

    match = re.search(r"```(?:json)?\n(.*?)\n```", result_str, re.DOTALL)
    if not match:
        return None

    json_content = match.group(1)
    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for error_freeness and coherence.")
    parser.add_argument("--input_file", required=True, help="Input JSONL file with result fields")
    args = parser.parse_args()

    ef_counts = Counter()
    coh_counts = Counter()
    total_parsed = 0
    total_skipped = 0
    total_invalid_result = 0

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing lines"), start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                total_skipped += 1
                continue

            parsed = extract_result_json(entry.get("result"))
            if parsed is None:
                total_invalid_result += 1
                continue

            ef_counts[parsed.get("error_freeness")] += 1
            coh_counts[parsed.get("coherence")] += 1
            total_parsed += 1

    print("\n--- Result Statistics ---")
    print(f"Total lines processed     : {line_num}")
    print(f"Valid JSON entries        : {total_parsed}")
    print(f"Invalid JSON lines skipped: {total_skipped}")
    print(f"Valid JSON but missing/invalid result field: {total_invalid_result}\n")

    print("Error Freeness:")
    for key in sorted(ef_counts):
        print(f"  {key}: {ef_counts[key]}")

    print("\nCoherence:")
    for key in sorted(coh_counts):
        print(f"  {key}: {coh_counts[key]}")


if __name__ == "__main__":
    main()
