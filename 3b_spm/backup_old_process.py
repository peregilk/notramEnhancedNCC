#!/usr/bin/env python3
"""
fast_augment_mcq.py
===================

Streaming MCQ-to-prompt converter (Norwegian) — fast and memory-light.

Key options
-----------
--shuffle_alternatives   Shuffle choice order, keep correct answer
--prompt_id N            0-49 fixed template   |  -k random  |  -99 all
--n_shots K              Prepend K shots (shared template); incompatible with --prompt_id
--limit N                Stop after N prompts
--keep_keys              Copy *all* original JSONL keys; overwrite only 'id' and 'text'
--output_file FILE       Write prompts as JSONL to FILE (stdout if omitted)
"""

import argparse, copy, json, random, re, sys
from typing import Dict, List
from tqdm import tqdm

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SHOT_BUF = 1_000            # reservoir size for few-shot sampling

# -------------------------------------------------------------------------
# 50 PROMPT TEMPLATES (bullet & comma templates use answer_text)
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
    7:  "Problem: {question}\nMulige svar:\n"
        "{letters_dashes}\n\nFasit: {answer}",
    8:  "Spørsmål: {question}\n{letters}\n\nEr det riktige svaret {options}?\nSvar: {answer}",
    9:  "Oppgave:\n{question}\n\nAlternativer:\n"
        "{letters}\n\nRiktig svar: {answer}",
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
}

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def fenced_json(text: str) -> dict:
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not m: raise ValueError
    return json.loads(m.group(1))

def norm_choices(raw) -> List[str]:
    if not isinstance(raw, list): raise ValueError
    def t(x): return (x["text"] if isinstance(x, dict) else x).strip()
    ch = [t(c) for c in raw]
    if re.match(r"^[A-ZÆØÅ]\s*[\.\:\)]", ch[0]): raise ValueError
    return ch

def validate(e: dict):
    e["choices"] = norm_choices(e["choices"])
    if len(e["choices"]) < 2: raise ValueError
    a = e.get("answer")
    if not (isinstance(a, str) and len(a) == 1): raise ValueError
    if a not in LABELS[: len(e["choices"])]: raise ValueError

def answer_text(e): return e["choices"][LABELS.index(e["answer"])]

def shuffle(e):
    cor = answer_text(e)
    random.shuffle(e["choices"])
    e["answer"] = LABELS[e["choices"].index(cor)]

def choose_pids(arg: int) -> List[int]:
    if arg is None:   return [random.randint(0, 49)]
    if arg >= 0:      return [arg]
    if arg == -99:    return list(range(50))
    return random.sample(range(50), k=min(abs(arg), 50))

def render(e, pid):
    q, ch, ans = e["question"].strip(), e["choices"], e["answer"]
    letters         = "\n".join(f"{l}: {t}" for l,t in zip(LABELS,ch))
    letters_bullets = "\n".join(f"- {l}: {t}" for l,t in zip(LABELS,ch))
    letters_dashes  = "\n".join(f"– {l}: {t}" for l,t in zip(LABELS,ch))
    plain_bullets   = "\n".join(f"- {t}" for t in ch)
    plain_commas    = ", ".join(ch[:-1])+f" eller {ch[-1]}" if len(ch)>1 else ch[0]
    opts  = ", ".join(LABELS[: len(ch)])
    return PROMPTS[pid].format(
        question=q, letters=letters, letters_bullets=letters_bullets,
        letters_dashes=letters_dashes, plain_bullets=plain_bullets,
        plain_commas=plain_commas, options=opts, answer=ans,
        answer_text=answer_text(e), answer_combo=f"{ans} – {answer_text(e)}",
        full_text_answer=answer_text(e))

SHOT_TEMPLATE = "Spørsmål: {question}\n\nSvar: {answer}"
def render_shot(e): return SHOT_TEMPLATE.format(
        question=e["question"].strip(), answer=answer_text(e))

# -------------------------------------------------------------------------
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_file", required=True)
    pa.add_argument("--output_file")
    pa.add_argument("--prompt_id", type=int)
    pa.add_argument("--shuffle_alternatives", action="store_true")
    pa.add_argument("--limit", type=int)
    pa.add_argument("--n_shots", type=int, default=0)
    pa.add_argument("--keep_keys", action="store_true",
                    help="Preserve all original keys (overwrite id & text)")
    args = pa.parse_args()

    if args.n_shots and args.prompt_id is not None:
        sys.exit("--n_shots cannot be combined with --prompt_id")

    outfh = open(args.output_file, "w", encoding="utf-8") if args.output_file else None
    wrt = outfh if outfh else sys.stdout

    reservoir: List[dict] = []
    produced = 0
    shot_suffix = f"_{args.n_shots}shot" if args.n_shots else ""

    for raw in tqdm(open(args.input_file, encoding="utf-8"),
                    desc="Processing", unit="line"):
        if args.limit and produced >= args.limit:
            break
        try:
            outer = json.loads(raw)
            item  = fenced_json(outer["result"])
            item["_base"] = outer.get("id", f"id{produced}")
            validate(item)
        except Exception:
            continue

        main_e = copy.deepcopy(item) if args.shuffle_alternatives else item
        if args.shuffle_alternatives:
            shuffle(main_e)

        if args.n_shots:
            if len(reservoir) < args.n_shots:
                # fill buffer first
                if len(reservoir) < SHOT_BUF:
                    reservoir.append(item)
                continue

            pid = random.randint(0, 49)
            shots = random.sample(reservoir, k=args.n_shots)
            prompt_txt = "\n\n".join(render(s, pid) for s in shots) + \
                         "\n\n" + render(main_e, pid)
            new_id = f"{item['_base']}{shot_suffix}_prompt{pid}"
            rec = outer.copy() if args.keep_keys else {}
            rec.update({"id": new_id, "text": prompt_txt})
            json.dump(rec, wrt); wrt.write("\n")
            produced += 1
        else:
            for pid in choose_pids(args.prompt_id):
                if args.limit and produced >= args.limit: break
                txt = render(main_e, pid)
                new_id = f"{item['_base']}_prompt{pid}"
                rec = outer.copy() if args.keep_keys else {}
                rec.update({"id": new_id, "text": txt})
                json.dump(rec, wrt); wrt.write("\n")
                produced += 1

        # update reservoir
        if args.n_shots:
            if len(reservoir) < SHOT_BUF:
                reservoir.append(item)
            else:
                j = random.randint(0, produced-1)
                if j < SHOT_BUF:
                    reservoir[j] = item

    if outfh: outfh.close()
    print(f"Prompts generated: {produced}", file=sys.stderr)

if __name__ == "__main__":
    main()
