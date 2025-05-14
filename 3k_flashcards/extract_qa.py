#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
from typing import List, Dict, Tuple, Any
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform JSONL with embedded Q&A into flat JSONL records."
    )
    parser.add_argument("--input_file", "-i", required=True)
    parser.add_argument("--output_file", "-o", required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def sanitize_quotes(s: str) -> str:
    """
    Replace typographic quotes with plain ASCII quotes to ensure JSON compatibility.
    """
    return (
        s.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
    )


def extract_qa_array(result_field: str) -> List[Dict[str, Any]]:
    """
    Extract JSON Q&A list from result_field. Supports:
    - Markdown code block with ```json [ ... ] ```
    - Raw string containing JSON list
    - Automatic sanitization of smart quotes
    """
    result_field = sanitize_quotes(result_field).strip()

    # Remove markdown block if present
    if result_field.startswith("```json"):
        result_field = result_field[len("```json"):].strip()
    if result_field.endswith("```"):
        result_field = result_field[:-3].strip()

    try:
        return json.loads(result_field)
    except json.JSONDecodeError:
        # Try repair: remove trailing comma and ensure closing bracket
        repaired = result_field.rstrip().rstrip(",")
        if not repaired.endswith("]"):
            repaired += "]"
        return json.loads(repaired)


def process_record(record: Dict[str, Any], debug: bool = False) -> Tuple[List[Dict[str, str]], List[str], int, str]:
    errors: List[str] = []
    outputs: List[Dict[str, str]] = []

    base_id = record.get("id")
    if not isinstance(base_id, str):
        errors.append("Missing or invalid 'id'")
        return outputs, errors, 0, ""

    result = record.get("result")
    if not isinstance(result, str):
        errors.append(f"Record {base_id}: Missing or invalid 'result' field")
        return outputs, errors, 0, ""

    try:
        qa_list = extract_qa_array(result)
    except Exception as e:
        errors.append(f"Record {base_id}: {e}")
        return outputs, errors, 0, result  # Return full result field for debug

    count = 0
    for idx, qa in enumerate(qa_list, start=1):
        q = qa.get("question")
        a = qa.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            errors.append(f"Record {base_id}_{idx}: Missing question or answer")
            continue
        new_id = f"{base_id}_{idx}"
        text = f"Spørsmål: {q}\nSvar: {a}\n"
        outputs.append({"id": new_id, "question": q, "answer": a, "text": text})
        count += 1

    return outputs, errors, count, ""


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    total_input = 0
    total_output = 0
    error_list: List[Tuple[int, str, str, str]] = []  # (line_num, id, error, raw_result)
    extraction_counts = Counter()

    try:
        with open(args.input_file, "r", encoding="utf-8") as infile, \
             open(args.output_file, "w", encoding="utf-8") as outfile:
            for line_num, line in enumerate(infile, 1):
                total_input += 1
                line = line.strip()
                if not line:
                    extraction_counts[0] += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    error_list.append((line_num, "N/A", f"JSON decode error: {e}", ""))
                    extraction_counts[0] += 1
                    continue

                record_id = record.get("id", "UNKNOWN")
                outputs, errors, count, raw_result = process_record(record, args.debug)
                extraction_counts[count] += 1

                for out in outputs:
                    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                    total_output += 1

                for err in errors:
                    error_list.append((line_num, record_id, err, raw_result))

    except IOError as e:
        logging.error(f"File error: {e}")
        sys.exit(1)

    # Summary
    print(f"\nProcessed {total_input} input lines.")
    print(f"Generated {total_output} output lines.")
    print("\nExtraction count distribution (how many Q&A items were extracted per input line):")
    for n in range(5, -1, -1):
        print(f"  {n} items: {extraction_counts[n]}")

    print(f"\nEncountered {len(error_list)} errors.")
    if args.debug and error_list:
        print("\n--- First 10 detailed debug errors ---")
        for i, (line_num, record_id, err, raw_result) in enumerate(error_list[:10]):
            print(f"\nRecord #{line_num} ID: {record_id}")
            print(f"Error: {err}")
            if raw_result:
                print("Full 'result' field:")
                print(raw_result)

if __name__ == "__main__":
    main()
