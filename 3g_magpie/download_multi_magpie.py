#!/usr/bin/env python3
"""
magpie_to_chat.py
=================
Stream Magpie-Align/Magpie-Pro-MT-300K-v0.1 from HF and convert to
instruction-chat JSONL (id + text) using a chat template.

Usage:
  python magpie_to_chat.py \
      --output_file magpie_chat.jsonl \
      --chat_template meta-llama/Meta-Llama-3-8B-Instruct \
      [--limit N]

Fields:
  - input1, output1, input2, output2, …  define the turns.
  - No --input_file: dataset ID is hard-coded.
"""

import argparse
import json
import re
import sys
from itertools import zip_longest

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Hard-coded HF dataset ID
DATASET_ID = "Magpie-Align/Magpie-Pro-MT-300K-v0.1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", required=True,
        help="Where to write the resulting JSONL"
    )
    parser.add_argument(
        "--chat_template", required=True,
        help="HF model ID with built-in chat template"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of examples"
    )
    args = parser.parse_args()

    # Load tokenizer + verify chat template support
    tok = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
    if not hasattr(tok, "apply_chat_template"):
        sys.exit("Error: tokenizer lacks `apply_chat_template`; pick a chat/instruction model.")
    system_prompt = getattr(tok, "default_system_prompt", "You are a helpful assistant.")

    # Stream the dataset
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    it = iter(ds)

    # Peek first record to discover fields
    try:
        first = next(it)
    except StopIteration:
        sys.exit("Dataset appears empty.")
    # Identify all input/output keys, sorted by their numeric suffix
    all_keys = list(first.keys())
    input_keys = sorted([k for k in all_keys if re.match(r"^input\d+$", k)],
                        key=lambda x: int(x.replace("input", "")))
    output_keys = sorted([k for k in all_keys if re.match(r"^output\d+$", k)],
                         key=lambda x: int(x.replace("output", "")))
    if not input_keys or len(input_keys) != len(output_keys):
        sys.exit(f"Unexpected columns: inputs={input_keys}, outputs={output_keys}")

    # Rewind streaming iterator by re-creating
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    writer = open(args.output_file, "w", encoding="utf-8")
    count = 0

    for row in tqdm(ds, desc="Converting", unit="ex"):
        if args.limit and count >= args.limit:
            break

        # Build the chat turns
        messages = [{"role": "system", "content": system_prompt}]
        for inp, out in zip(input_keys, output_keys):
            # skip if missing or empty
            user_txt = row.get(inp)
            bot_txt  = row.get(out)
            if not user_txt or not bot_txt:
                continue
            messages.append({"role": "user",      "content": user_txt.strip()})
            messages.append({"role": "assistant", "content": bot_txt.strip()})

        if len(messages) < 3:
            # no valid turns beyond system→user→assistant
            continue

        # Apply chat template
        chat_text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        # Generate a stable ID (use existing uuid if present)
        example_id = row.get("uuid") or f"magpie_{count}"
        writer.write(json.dumps({"id": example_id, "text": chat_text},
                                ensure_ascii=False) + "\n")
        count += 1

    writer.close()
    print(f"Finished. Wrote {count} examples to {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
