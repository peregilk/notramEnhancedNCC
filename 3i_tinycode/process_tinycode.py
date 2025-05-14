#!/usr/bin/env python3
"""
tiny_codes_to_chat.py
---------------------
Konverterer datasettet "nampdn-ai/tiny-codes" til et
instruction-chat-format (id + text).

* Én JSONL-linje per eksempel.
* Rollen «user» får feltet  prompt
* Rollen «assistant» får feltet  response
* Første turn er en system-prompt hentet fra tokenizer.default_system_prompt
  (eller «You are a helpful assistant.» hvis modellen ikke har en).
* Chat-prompten bygges med tokenizer.apply_chat_template().

Bruk:
python tiny_codes_to_chat.py \
    --output_file tiny_codes_chat.jsonl \
    --chat_template meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import sys

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

LABELS = {"user": "prompt", "assistant": "response"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_file", required=True,
                    help="sti til JSONL-utfil")
    ap.add_argument("--chat_template", required=True,
                    help="HF-modell id med chat-template")
    args = ap.parse_args()

    print("Laster tokenizer …")
    tok = AutoTokenizer.from_pretrained(args.chat_template,
                                        trust_remote_code=True)
    if not hasattr(tok, "apply_chat_template"):
        sys.exit("Tokenizeren mangler apply_chat_template – velg chatmodell")
    system_prompt = getattr(
        tok, "default_system_prompt", "You are a helpful assistant.")

    print("Laster dataset (tiny-codes) …")
    ds = load_dataset("nampdn-ai/tiny-codes", split="train")
    print("Antall eksempler:", len(ds))

    with open(args.output_file, "w", encoding="utf-8") as fout, \
            tqdm(total=len(ds), desc="Eksempler") as bar:
        for idx, row in enumerate(ds):
            bar.update(1)
            try:
                prompt = row["prompt"].strip()
                response = row["response"].strip()
            except KeyError as e:
                continue  # hopp over ufullstendige rader

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

            json.dump({"id": f"tiny_codes_{idx}", "text": text},
                      fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    main()
