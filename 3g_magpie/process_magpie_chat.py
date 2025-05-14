#!/usr/bin/env python3
"""
jsonl_to_chat.py
================
Konverterer et JSON-Lines-datasett til et instruksjons­chat-format
(id + text).

Input pr. linje inneholder:
  • «askLLMresult»  (norsk dialog)
  • «conversations» (engelsk dialog)

Med flagget  --english  brukes «conversations» og id-en prefikses med
«en_».  Uten flagget brukes «askLLMresult».

Validerer at dialogen:
  • starter med «human»  • annenhver human/gpt  • slutter med «gpt»

Outputskrives med tokenizer.apply_chat_template().  
Statistikk (linjer, skrevet, droppet, turfordeling) vises til stderr.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

LABEL_TO_ROLE = {"human": "user", "gpt": "assistant"}


# ---------- validering ----------------------------------------------------
def valid_chat(msgs: list[dict]) -> bool:
    if len(msgs) < 2:
        return False
    expected = "human"
    for m in msgs:
        if m.get("from") != expected or "value" not in m:
            return False
        expected = "gpt" if expected == "human" else "human"
    return msgs[-1]["from"] == "gpt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--chat_template", required=True,
                    help="HF-modell med innebygget chat-template")
    ap.add_argument("--english", action="store_true",
                    help="bruk conversations og prefiks id med en_")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.chat_template, trust_remote_code=True)
    if not hasattr(tokenizer, "apply_chat_template"):
        sys.exit("Valgt tokenizer mangler apply_chat_template (ikke chatmodell)")

    system_prompt = getattr(
        tokenizer, "default_system_prompt", "You are a helpful assistant.")

    processed = written = skipped = 0
    turns_stats: Counter[int] = Counter()

    key = "conversations" if args.english else "askLLMresult"
    id_prefix = "en_" if args.english else ""

    with open(args.input_file, encoding="utf-8") as fin, \
            open(args.output_file, "w", encoding="utf-8") as fout, \
            tqdm(desc="Lines", unit="l") as bar:

        for line in fin:
            bar.update(1)
            processed += 1
            try:
                obj = json.loads(line)
                msgs = obj.get(key)
                if not msgs or not valid_chat(msgs):
                    raise ValueError
                # bygg chat-meldinger
                chat = [{"role": "system", "content": system_prompt}]
                for m in msgs:
                    chat.append(
                        {"role": LABEL_TO_ROLE[m["from"]],
                         "content": m["value"].strip()}
                    )

                prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=False)

                out_id = id_prefix + obj.get("uuid", f"row{written}")
                json.dump({"id": out_id, "text": prompt},
                          fout, ensure_ascii=False)
                fout.write("\n")

                written += 1
                turns_stats[len(msgs)//2] += 1  # par human/gpt

            except Exception:
                skipped += 1

    # ---------- sammendrag ----------
    print(f"\nFerdig. linjer={processed}  skrevet={written}  droppet={skipped}",
          file=sys.stderr)
    print("Tur-fordeling:", file=sys.stderr)
    for k in sorted(turns_stats):
        print(f"  {k}-turn: {turns_stats[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
