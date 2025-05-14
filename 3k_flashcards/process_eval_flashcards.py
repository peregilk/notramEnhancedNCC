#!/usr/bin/env python3
"""
process_eval.py
===============

Filtrerer og konverterer et JSONL-evalsett med feltene:
  id, question, answer, text, result

Kun eksempler med:
  • error_freeness == 5
  • answerability   == 5
  • general_knowledge_fit is True
beholdes.

Output: JSONL med kun { "id", "text" }.

Uten --chat_template:
  text = "{question}\n{answer}"

Med --chat_template MODEL:
  text fra tokenizer.apply_chat_template() med:
    system: modellen.default_system_prompt
    user:   question
    assistant: answer

Skriptet rapporterer antall linjer prosessert, skrevet og slettet.
"""

import argparse
import json
import re
import sys

from tqdm import tqdm
from transformers import AutoTokenizer

START_FENCE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

def parse_result(fenced: str) -> dict:
    m = START_FENCE.search(fenced)
    if not m:
        raise ValueError("Missing fenced JSON")
    return json.loads(m.group(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file",  required=True,
                   help="Input JSONL med eval-resultater")
    p.add_argument("--output_file", required=True,
                   help="Output JSONL med id+text")
    p.add_argument("--chat_template",
                   help="HF-modell med innebygget chat-template")
    args = p.parse_args()

    # Chat-modus?
    use_chat = bool(args.chat_template)
    if use_chat:
        tok = AutoTokenizer.from_pretrained(
            args.chat_template, trust_remote_code=True
        )
        if not hasattr(tok, "apply_chat_template"):
            sys.exit("Tokenizer mangler apply_chat_template; velg chatmodell")
        system_prompt = getattr(tok, "default_system_prompt",
                                "You are a helpful assistant.")

    total = kept = dropped = 0

    with open(args.input_file, encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout, \
         tqdm(desc="Prosessering", unit="linje") as bar:

        for line in fin:
            bar.update(1)
            total += 1
            try:
                rec = json.loads(line)
                # parse result-felt
                res = parse_result(rec.get("result", ""))
                if not (
                    res.get("error_freeness") == 5 and
                    res.get("answerability")   == 5 and
                    res.get("general_knowledge_fit") is True
                ):
                    dropped += 1
                    continue

                q = rec["question"].strip()
                a = rec["answer"].strip()

                if use_chat:
                    msgs = [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": q},
                        {"role": "assistant", "content": a},
                    ]
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                else:
                    text = f"{q}\n{a}"

                out = {"id": rec["id"], "text": text}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1

            except Exception:
                dropped += 1
                continue

    print(f"\nTotal: {total}  skrevet: {kept}  slettet: {dropped}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
