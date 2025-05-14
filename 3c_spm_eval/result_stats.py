#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter, defaultdict
from tqdm import tqdm


def parse_result_field(result_str, debug=False):
    # Remove surrounding ```json ... ``` markers
    if result_str.startswith("```json") and result_str.endswith("```"):
        inner = result_str[len("```json"):].strip("`\n ")
    elif result_str.startswith("```") and result_str.endswith("```"):
        inner = result_str.strip("`\n ")
    else:
        if debug:
            print("No valid triple backtick JSON block found.")
        return None

    try:
        return json.loads(inner)
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON decoding failed: {e}")
        return None


def compute_stats(data, debug=False, collect_examples=False):
    ef_counter = Counter()
    ans_counter = Counter()
    gkf_counter = Counter()
    total = 0

    ef_examples = {}
    ans_examples = {}
    gkf_examples = {}

    for row in tqdm(data, desc="Processing"):
        try:
            obj = json.loads(row)
            parsed = parse_result_field(obj.get("result", ""), debug)
            if parsed is None:
                continue

            ef = parsed.get("error_freeness")
            ans = parsed.get("answerability")
            gkf = parsed.get("general_knowledge_fit")

            if isinstance(ef, int):
                ef_counter[ef] += 1
                if collect_examples and ef not in ef_examples:
                    ef_examples[ef] = row.strip()
            else:
                if debug:
                    print(f"Invalid error_freeness value: {ef}")

            if isinstance(ans, int):
                ans_counter[ans] += 1
                if collect_examples and ans not in ans_examples:
                    ans_examples[ans] = row.strip()
            else:
                if debug:
                    print(f"Invalid answerability value: {ans}")

            if isinstance(gkf, bool):
                gkf_counter[gkf] += 1
                if collect_examples and gkf not in gkf_examples:
                    gkf_examples[gkf] = row.strip()
            else:
                if debug:
                    print(f"Invalid general_knowledge_fit value: {gkf}")

            total += 1
        except Exception as e:
            if debug:
                print(f"Skipping row due to exception: {e}")
            continue

    return (ef_counter, ans_counter, gkf_counter, total,
            ef_examples, ans_examples, gkf_examples)


def print_histogram(counter, total, title):
    print(f"\n{title}")
    if not counter:
        print("  (no data)")
        return
    for key in sorted(counter):
        percent = 100 * counter[key] / total if total else 0
        print(f"  {key}: {counter[key]} ({percent:.1f}%)")


def print_verbose_examples(name, examples):
    print(f"\nFirst examples for each category in {name}:")
    for key in sorted(examples):
        print(f"\n{name} == {key}")
        print(examples[key])


def main():
    parser = argparse.ArgumentParser(description="Compute histogram stats from JSONL result field")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .jsonl file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for failed parses")
    parser.add_argument("--verbose", action="store_true", help="Print one example for each value category")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    (ef_counter, ans_counter, gkf_counter, total,
     ef_examples, ans_examples, gkf_examples) = compute_stats(
        lines, debug=args.debug, collect_examples=args.verbose)

    print("\nSummary statistics for parsed results:")
    print_histogram(ef_counter, total, "Error Freeness (1-5)")
    print_histogram(ans_counter, total, "Answerability (1-5)")
    print_histogram(gkf_counter, total, "General Knowledge Fit (bool)")

    if args.verbose:
        print_verbose_examples("error_freeness", ef_examples)
        print_verbose_examples("answerability", ans_examples)
        print_verbose_examples("general_knowledge_fit", gkf_examples)


if __name__ == "__main__":
    main()

