#!/usr/bin/env python3
import argparse
import json
import re
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


def is_valid(parsed_result):
    """
    Returns True if both error_freeness and coherence >= 4
    """
    if not isinstance(parsed_result, dict):
        return False
    return (
        isinstance(parsed_result.get("error_freeness"), int)
        and isinstance(parsed_result.get("coherence"), int)
        and parsed_result["error_freeness"] >= 4
        and parsed_result["coherence"] >= 4
    )


def main():
    parser = argparse.ArgumentParser(description="Filter and deduplicate by text.")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    args = parser.parse_args()

    kept = []
    seen_texts = set()

    total_lines = 0
    skipped_json_error = 0
    skipped_result_error = 0
    skipped_short_text = 0
    skipped_duplicate = 0
    matched = 0

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Filtering"):
            total_lines += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped_json_error += 1
                continue

            text = entry.get("text", "")
            if len(text) < 50:
                skipped_short_text += 1
                continue

            parsed_result = extract_result_json(entry.get("result"))
            if not is_valid(parsed_result):
                skipped_result_error += 1
                continue

            if text in seen_texts:
                skipped_duplicate += 1
                continue

            seen_texts.add(text)
            matched += 1
            kept.append({
                "id": entry.get("id"),
                "text": text
            })

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for entry in kept:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n--- Filter Summary ---")
    print(f"Total lines read               : {total_lines}")
    print(f"Valid and matched entries kept : {matched}")
    print(f"Skipped due to JSON errors     : {skipped_json_error}")
    print(f"Skipped due to short text      : {skipped_short_text}")
    print(f"Skipped due to filter criteria : {skipped_result_error}")
    print(f"Skipped due to duplicates      : {skipped_duplicate}")


if __name__ == "__main__":
    main()
