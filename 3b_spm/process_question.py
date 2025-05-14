#!/usr/bin/env python3
"""
fast_augment_mcq.py
===================

Streaming MCQ-to-prompt converter (Norwegian + extra languages).

Options
-------
--shuffle_alternatives   Shuffle choice order, keep correct answer
--prompt_id N            0-53 fixed template
                         -k   |k| random templates/item   (1 ≤ k ≤ 54)
                         -99  all templates
--n_shots K              Prepend K shots; shots & target share one template
                         (incompatible with --prompt_id)
--limit N                Stop after N generated prompts
--keep_keys              Copy all original JSON keys; overwrite only id/text
--backup_result          Duplicate original 'result' ➜ 'old_result'
--output_file FILE       Write prompts as JSONL (stdout if omitted)

Fast: single-pass, ≤1000-item reservoir for few-shot sampling.
"""

import argparse
import copy
import json
import random
import re
import sys
from typing import Dict, List

from tqdm import tqdm

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SHOT_BUF = 1_000                       # reservoir size for few-shot

# -------------------------------------------------------------------------
# PROMPT TEMPLATES
# -------------------------------------------------------------------------
PROMPTS: Dict[int, str] = {
    0:  "Spørsmål: {question}\n\nSvar: {full_text_answer}",
    1:  "Oppgave: {question}\n\nFasit: {full_text_answer}",
    2:  "Spørsmål: {question}\n\nRiktig svar: {full_text_answer}",
    3:  "Problemstilling:\n{question}\n\nKorrekt svar: {full_text_answer}",
    4:  "Les spørsmålet nøye:\n{question}\n\nSvar: {full_text_answer}",

    5:  "Spørsmål: {question}\nHvilket alternativ er riktig?\n"
        "{letters}\n\nSvar: {answer}",
    6:  "Oppgave: {question}\nVelg riktig alternativ blant punktene under:\n"
        "{letters_bullets}\n\nSvar: {answer}",
    # updated 7
    7:  "Problem: {question}\nLøsning:\n"
        "{letters_dashes}\n\nFasit: {answer}",

    8:  "Problemstilling: {question}\nMulige svar:\n"
        "{letters_dashes}\n\nFasit: {answer}",
    9:  "Spørsmål: {question}\n{letters}\n\nEr det riktige svaret {options}?\nSvar: {answer}",
    10: "Spørsmål: {question}\nAlternativene er:\n"
         "{letters_bullets}\n\nRiktig bokstav: {answer}",
    11: "Oppgave: {question}\nVelg én av disse:\n"
         "{letters_dashes}\n\nFasit: {answer}",
    12: "Spørsmål: {question}\nHva er riktig?\n"
         "{letters}\n\nSvar: {answer}",
    13: "Oppgave: {question}\nHvilket alternativ stemmer?\n"
         "{letters_bullets}\n\nSvarbokstav: {answer}",
    14: "Spørsmål: {question}\nMarker det riktige svaret.\n"
         "{letters}\n\nSvar: {answer}",
    15: "Oppgave: {question}\nMulige svar:\n"
         "{letters_bullets}\n\nDet riktige svaret er: {answer}",
    16: "Problem: {question}\nSe alternativene nedenfor:\n"
         "{letters_dashes}\n\nRiktig svar: {answer}",
    17: "Spørsmål: {question}\nVelg den korrekte påstanden.\n"
         "{letters}\n\nSvar: {answer}",
    18: "Oppgave: {question}\nKun ett alternativ er riktig.\n"
         "{letters_bullets}\n\nSvar: {answer}",
    19: "Spørsmål: {question}\n{letters}\n\nHva velger du?\nSvar: {answer}",

    # bullet / comma templates – answer_text only
    20: "Oppgave: {question}\n{plain_bullets}\n\nKorrekt svar: {answer_text}",
    21: "Spørsmål: {question}\n{plain_commas}.\n\nRiktig svar: {answer_text}",
    22: "Spørsmål: {question}\n{plain_commas}.\n\nSvar: {answer_text}",
    23: "Oppgave: {question}\n{plain_bullets}\n\nSvar: {answer_text}",
    24: "Spørsmål: {question}\n{plain_commas}.\n\nFasit: {answer_text}",
    25: "Oppgave: {question}\n{plain_commas}.\n\nKorrekt svar: {answer_text}",
    26: "Problem: {question}\n{plain_bullets}\n\nRiktig alternativ: {answer_text}",
    27: "Spørsmål: {question}\n{plain_commas}.\n\nSvar: {answer_text}",
    28: "Oppgave: {question}\n{plain_commas}.\n\nRiktig svar: {answer_text}",
    29: "{question}\n{plain_bullets}\n\nHva er korrekt?\n\nSvar: {answer_text}",
    30: "{question}\n{plain_commas}.\n\nSvar (tekst): {answer_text}",
    31: "Quiz:\n{question}\n{plain_bullets}\n\nRiktig svar: {answer_text}",
    32: "{question}\n{plain_commas}.\n\nSvar: {answer_text}",
    33: "Oppgave: {question}\n{plain_bullets}\n\nDet riktige svaret er: {answer_text}",
    34: "{question}\n{plain_commas}.\n\nSvar: {answer_text}",
    35: "Spørsmål: {question}\n{plain_commas}.\n\nFasit: {answer_text}",
    36: "Problem: {question}\n{plain_bullets}\n\nRiktig svar: {answer_text}",
    37: "{question}\n{plain_commas}.\n\nSvar: {answer_text}",
    38: "Oppgave: {question}\n{plain_commas}.\n\nKorrekt svar: {answer_text}",
    39: "Spørsmål: {question}\n{plain_bullets}\n\nSvar: {answer_text}",
    40: "Problem: {question}\n{plain_commas}.\n\nRiktig svar: {answer_text}",
    41: "{question}\n{plain_commas}.\n\nSvar: {answer_text}",
    42: "Utfordring: {question}\n{plain_bullets}\n\nHva er riktig?\n\nSvar: {answer_text}",
    43: "Oppgave: {question}\n{plain_commas}.\n\nLøsning: {answer_text}",
    44: "{question}\n{plain_commas}.\n\nRiktig alternativ: {answer_text}",
    45: "Quiz-spørsmål: {question}\n{plain_bullets}\n\nFasit: {answer_text}",
    46: "Spørsmål: {question}\n{plain_commas}.\n\nSvar: {answer_text}",
    47: "Oppgave: {question}\n{plain_commas}.\n\nKorrekt svar: {answer_text}",
    48: "Utfordring: {question}\n{plain_bullets}\n\nSvar: {answer_text}",
    49: "{question}\n{plain_commas}.\n\nRiktig svar: {answer_text}",

    # Additional language variants
    50: "Problem: {question}\nSolution:\n"
        "{letters_dashes}\n\nAnswer: {answer}",              # English
    51: "Problem: {question}\nLøsning:\n"
        "{letters_dashes}\n\nFacit: {answer}",               # Danish
    52: "Problem: {question}\nLösning:\n"
        "{letters_dashes}\n\nFacit: {answer}",               # Swedish
    53: "Problem: {question}\nLøysing:\n"
        "{letters_dashes}\n\nFasit: {answer}",               # Nynorsk
}

MAX_PID = max(PROMPTS)  # 53

# -------------------------------------------------------------------------
def fenced_json(text: str) -> dict:
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not m:
        raise ValueError
    return json.loads(m.group(1))

def norm_choices(raw) -> List[str]:
    if not isinstance(raw, list):
        raise ValueError
    def t(x): return (x["text"] if isinstance(x, dict) else x).strip()
    ch = [t(c) for c in raw]
    if re.match(r"^[A-ZÆØÅ]\s*[\.\:\)]", ch[0]):
        raise ValueError
    return ch

def validate(e: dict):
    e["choices"] = norm_choices(e["choices"])
    if len(e["choices"]) < 2:
        raise ValueError
    a = e.get("answer")
    if not (isinstance(a, str) and len(a) == 1):
        raise ValueError
    if a not in LABELS[: len(e["choices"])]: raise ValueError

def answer_text(e): return e["choices"][LABELS.index(e["answer"])]
def shuffle(e):
    cor = answer_text(e)
    random.shuffle(e["choices"])
    e["answer"] = LABELS[e["choices"].index(cor)]

def choose_pids(arg: int) -> List[int]:
    if arg is None:   return [random.randint(0, MAX_PID)]
    if arg >= 0:      return [arg] if arg <= MAX_PID else sys.exit("prompt_id out of range")
    if arg == -99:    return list(range(MAX_PID + 1))
    return random.sample(range(MAX_PID + 1), k=min(abs(arg), MAX_PID + 1))

def render(e, pid):
    q, ch, ans = e["question"].strip(), e["choices"], e["answer"]
    letters         = "\n".join(f"{l}: {t}" for l, t in zip(LABELS, ch))
    letters_bullets = "\n".join(f"- {l}: {t}" for l, t in zip(LABELS, ch))
    letters_dashes  = "\n".join(f"– {l}: {t}" for l, t in zip(LABELS, ch))
    plain_bullets   = "\n".join(f"- {t}" for t in ch)
    plain_commas    = ", ".join(ch[:-1])+f" eller {ch[-1]}" if len(ch)>1 else ch[0]
    opts            = ", ".join(LABELS[: len(ch)])
    return PROMPTS[pid].format(
        question=q, letters=letters, letters_bullets=letters_bullets,
        letters_dashes=letters_dashes, plain_bullets=plain_bullets,
        plain_commas=plain_commas, options=opts, answer=ans,
        answer_text=answer_text(e),
        answer_combo=f"{ans} – {answer_text(e)}",
        full_text_answer=answer_text(e))

SHOT_TEMPLATE = "Spørsmål: {question}\n\nSvar: {answer}"
def render_shot(e): return SHOT_TEMPLATE.format(
        question=e["question"].strip(), answer=answer_text(e))

# -------------------------------------------------------------------------
def main() -> None:
    pr = argparse.ArgumentParser()
    pr.add_argument("--input_file", required=True)
    pr.add_argument("--output_file")
    pr.add_argument("--prompt_id", type=int)
    pr.add_argument("--shuffle_alternatives", action="store_true")
    pr.add_argument("--limit", type=int)
    pr.add_argument("--n_shots", type=int, default=0)
    pr.add_argument("--keep_keys", action="store_true")
    pr.add_argument("--backup_result", action="store_true")
    args = pr.parse_args()

    if args.n_shots and args.prompt_id is not None:
        sys.exit("--n_shots cannot be combined with --prompt_id")

    outfh = open(args.output_file, "w", encoding="utf-8") if args.output_file else None
    wr = outfh if outfh else sys.stdout

    reservoir: List[dict] = []
    produced = 0
    shot_suffix = f"_{args.n_shots}shot" if args.n_shots else ""

    for raw in tqdm(open(args.input_file, encoding="utf-8"),
                    desc="Processing", unit="line"):
        if args.limit and produced >= args.limit:
            break
        try:
            outer = json.loads(raw)
            itm   = fenced_json(outer["result"])
            itm["_base"] = outer.get("id", f"id{produced}")
            validate(itm)
        except Exception:
            continue

        main_e = copy.deepcopy(itm) if args.shuffle_alternatives else itm
        if args.shuffle_alternatives:
            shuffle(main_e)

        # FEW-SHOT
        if args.n_shots:
            if len(reservoir) < args.n_shots:
                if len(reservoir) < SHOT_BUF:
                    reservoir.append(itm)
                continue

            pid = random.randint(0, MAX_PID)
            shots = random.sample(reservoir, k=args.n_shots)
            prompt_text = "\n\n".join(render(s, pid) for s in shots) + \
                          "\n\n" + render(main_e, pid)
            rec = outer.copy() if args.keep_keys else {}
            if args.backup_result:
                rec["old_result"] = outer.get("result")
            rec.update({"id": f"{itm['_base']}{shot_suffix}_prompt{pid}",
                        "text": prompt_text})
            json.dump(rec, wr); wr.write("\n")
            produced += 1
        # REGULAR
        else:
            for pid in choose_pids(args.prompt_id):
                if args.limit and produced >= args.limit:
                    break
                txt = render(main_e, pid)
                rec = outer.copy() if args.keep_keys else {}
                if args.backup_result:
                    rec["old_result"] = outer.get("result")
                rec.update({"id": f"{itm['_base']}_prompt{pid}", "text": txt})
                json.dump(rec, wr); wr.write("\n")
                produced += 1

        # update reservoir
        if args.n_shots:
            if len(reservoir) < SHOT_BUF:
                reservoir.append(itm)
            else:
                j = random.randint(0, produced-1)
                if j < SHOT_BUF:
                    reservoir[j] = itm

    if outfh:
        outfh.close()
    print(f"Prompts generated: {produced}", file=sys.stderr)

if __name__ == "__main__":
    main()
