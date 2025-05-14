#!/usr/bin/env python3
"""
fast_augment_mcq.py
-------------------
MCQ ➜ plain prompts  |  HF chat prompts
"""

import argparse, json, os, random, re, sys
from typing import List
from tqdm import tqdm

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
OR_WORD = "eller"   # overridden by --english

# ------------ templates ---------------------------------------------------
def load_templates():
    path = os.path.join(os.path.dirname(__file__), "prompts.txt")
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")]
TEMPLATES = load_templates()
MAX_PID   = len(TEMPLATES) - 1

# ------------ HF chat -----------------------------------------------------
def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(tok, "apply_chat_template"):
        sys.exit("tokenizer lacks chat template")
    return tok

# ------------ helpers -----------------------------------------------------
def fenced_json(txt): return json.loads(re.search(r"```json\s*(.*?)\s*```", txt, re.S).group(1))
def trim(s): return s.strip()
def commas(ch):  return ", ".join(trim(c) for c in ch[:-1]) + f" {OR_WORD} {trim(ch[-1])}"
def bullets_labeled(ch): return "\n".join(f"- {l}: {trim(t)}" for l, t in zip(LABELS, ch))

def norm_choices(raw):
    def txt(x): return (x["text"] if isinstance(x, dict) else x).strip()
    lst = [txt(c) for c in raw]
    if re.match(r"^[A-ZÆØÅ]\s*[\.\:\)]", lst[0]): raise ValueError
    return lst

def validate(e):
    e["choices"] = norm_choices(e["choices"])
    if e["answer"] not in LABELS[: len(e["choices"])]:
        raise ValueError

def answer_text(e): return e["choices"][LABELS.index(e["answer"])]

def shuffle_choices(e):
    right = answer_text(e)
    random.shuffle(e["choices"])
    e["answer"] = LABELS[e["choices"].index(right)]

# ------------ main --------------------------------------------------------
def main():
    global OR_WORD
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_file", required=True)
    pa.add_argument("--output_file")
    pa.add_argument("--chat_template")
    pa.add_argument("--prompt_id", type=int)
    pa.add_argument("--shuffle_alternatives", action="store_true")
    pa.add_argument("--english", action="store_true", help="use 'or' instead of 'eller'")
    pa.add_argument("--keep_keys", action="store_true")
    pa.add_argument("--backup_result", action="store_true")
    pa.add_argument("--limit", type=int)
    args = pa.parse_args()

    if args.english:
        OR_WORD = "or"

    chat = bool(args.chat_template)
    if chat:
        tok = load_tokenizer(args.chat_template)
        sys_prompt = getattr(tok, "default_system_prompt", "You are a helpful assistant.")
        args.prompt_id = None  # ignore prompt choice in chat mode

    outfh = open(args.output_file, "w", encoding="utf-8") if args.output_file else None
    wr = outfh or sys.stdout
    produced = 0
    suffix_chat = f"_{args.chat_template.split('/')[-1]}" if chat else ""

    for ln in tqdm(open(args.input_file, encoding="utf-8"), desc="Processing", unit="line"):
        if args.limit and produced >= args.limit:
            break
        try:
            outer = json.loads(ln)
            e = fenced_json(outer["old_result"])
            validate(e)
        except Exception:
            continue

        if args.shuffle_alternatives and not chat:
            shuffle_choices(e)

        # -------- chat mode --------
        if chat:
            bullet = random.random() < 0.5
            if bullet:
                user_msg = f"{trim(e['question'])}\n{bullets_labeled(e['choices'])}"
                assistant = e["answer"]
            else:
                user_msg = f"{trim(e['question'])}\n{commas(e['choices'])}"
                assistant = answer_text(e)

            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant},
            ]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

            rec = outer.copy() if args.keep_keys else {}
            if args.backup_result: rec["old_result"] = outer.get("old_result")
            rec.update({"id": f"{outer.get('id', f'id{produced}')}{suffix_chat}", "text": text})
            json.dump(rec, wr); wr.write("\n")
            produced += 1
            continue

        # -------- plain mode --------
        pid = 0 if args.prompt_id is None else args.prompt_id
        tpl = TEMPLATES[pid]
        prompt = tpl.format(
            question=trim(e["question"]),
            letters="\n".join(f"{l}: {trim(t)}" for l, t in zip(LABELS, e["choices"])),
            letters_bullets=bullets_labeled(e["choices"]),
            letters_dashes="\n".join(f"– {l}: {trim(t)}" for l, t in zip(LABELS, e["choices"])),
            plain_bullets=bullets_labeled(e["choices"]),
            plain_commas=commas(e["choices"]),
            options=", ".join(LABELS[: len(e["choices"])]),
            answer=e["answer"],
            answer_text=answer_text(e),
            full_text_answer=answer_text(e),
            answer_combo=f"{e['answer']} – {answer_text(e)}",
        )
        rec = outer.copy() if args.keep_keys else {}
        if args.backup_result: rec["old_result"] = outer.get("old_result")
        rec.update({"id": f"{outer.get('id', f'id{produced}')}_prompt{pid}", "text": prompt})
        json.dump(rec, wr); wr.write("\n")
        produced += 1

    if outfh:
        outfh.close()
    print(f"Prompts generated: {produced}", file=sys.stderr)

if __name__ == "__main__":
    main()
