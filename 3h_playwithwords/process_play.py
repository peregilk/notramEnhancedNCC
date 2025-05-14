#!/usr/bin/env python3
"""
replace_system_prompt.py
------------------------
Bytter system-prompten i et chat-formatert JSONL (felt id + text).

Ny prompt-rekkefølge:
    1. --system_prompt  (hvis gitt)
    2. tokenizer.default_system_prompt
    3. "You are a helpful assistant."  (reserve)

Alle andre deler er uendret.
"""

import argparse, json, sys
from tqdm import tqdm
from transformers import AutoTokenizer

START_TAG = "<|start_header_id|>system<|end_header_id|>"
EOT_TAG   = "<|eot_id|>"


def swap_prompt(text: str, new_prompt: str | None) -> str | None:
    start = text.find(START_TAG)
    if start == -1:
        return None
    start += len(START_TAG)
    end = text.find(EOT_TAG, start)
    if end == -1:
        return None
    if new_prompt is None:          # ingen endring
        return text
    return text[:start] + "\n\n" + new_prompt.strip() + text[end:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--chat_template", required=True)
    ap.add_argument("--system_prompt",
                    help="overstyr modellens default-prompt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.chat_template,
                                        trust_remote_code=True)

    # henter standard-prompt fra tokenizeren
    default_prompt = getattr(tok, "default_system_prompt",
                             "You are a helpful assistant.")
    new_prompt = args.system_prompt or default_prompt

    tot = ok = bad = 0
    with open(args.input_file, encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout, \
         tqdm(desc="Lines", unit="l") as bar:
        for line in fin:
            bar.update(1); tot += 1
            try:
                row = json.loads(line)
                text2 = swap_prompt(row["text"], new_prompt)
                if text2 is None:
                    raise ValueError
                json.dump({"id": row["id"], "text": text2},
                          fout, ensure_ascii=False)
                fout.write("\n"); ok += 1
            except Exception:
                bad += 1

    print(f"Total {tot}  skrevet {ok}  droppet {bad}", file=sys.stderr)


if __name__ == "__main__":
    main()
