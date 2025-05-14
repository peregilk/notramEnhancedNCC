#!/usr/bin/env python3
import argparse
import json
import re
from tqdm import tqdm


def extract_result_field(result_str):
    """
    Parse the stringified JSON inside the 'result' field.
    Assumes it is wrapped in markdown-style triple backticks and optional 'json' tag.
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


def is_valid_result(parsed_result):
    if not isinstance(parsed_result, dict):
        return False
    return (
        parsed_result.get("error_freeness") == 5
        and parsed_result.get("answerability") == 5
        and parsed_result.get("general_knowledge_fit") is True
    )


def main():
    parser = argparse.ArgumentParser(description="Filter and deduplicate best results from JSONL file.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_file", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    seen_texts = set()
    filtered_unique = []
    total_matched = 0
    duplicates_skipped = 0

    for entry in tqdm(lines, desc="Filtering"):
        parsed = extract_result_field(entry.get("result"))
        if is_valid_result(parsed):
            total_matched += 1
            text = entry.get("text")
            if text not in seen_texts:
                seen_texts.add(text)
                filtered_unique.append(entry)
            else:
                duplicates_skipped += 1

    with open(args.output_file, "w", encoding="utf-8") as f:
        for entry in filtered_unique:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Total input lines         : {len(lines)}")
    print(f"Matched filtered criteria : {total_matched}")
    print(f"Unique text entries kept  : {len(filtered_unique)}")
    print(f"Duplicates skipped        : {duplicates_skipped}")


if __name__ == "__main__":
    main()
