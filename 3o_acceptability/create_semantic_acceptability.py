#!/usr/bin/env python3

import argparse
import json
import random
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# 20 Norwegian prompt templates, all specifying the options ja, nei
NB_PROMPTS = [
    "Følgende er setninger og hvorvidt de er grammatisk korrekte. Mulige svar: {labels_str}.",
    "Er denne setningen grammatisk korrekt? Svar med {labels_str}.",
    "Klassifiser om teksten under er grammatisk korrekt ({labels_str}).",
    "Vurder setningen og angi om den er korrekt grammatisk ({labels_str}):",
    "Er setningen riktig skrevet? Kun {labels_str} tillatt.",
    "Angi korrekthet. Gyldige svar: {labels_str}.",
    "Bestem om denne setningen er grammatisk korrekt eller ikke. ({labels_str})",
    "Svar kun med ett av: {labels_str}.",
    "Er teksten grammatisk riktig? Svar {labels_str}.",
    "Setning: {{text}}\nEr denne grammatisk korrekt? ({labels_str})",
    "Språkvurdering: Velg {labels_str}.",
    "Er dette god grammatikk? Mulige svar: {labels_str}.",
    "Gi kun ett av følgende som svar: {labels_str}.",
    "Vurder grammatisk korrekthet. Kun {labels_str} tillatt.",
    "Korrekt setning? Svar med {labels_str}.",
    "Språkanalyse: er setningen korrekt? ({labels_str})",
    "Oppgi om setningen er grammatisk korrekt. ({labels_str})",
    "Vurder setningen og svar {labels_str}.",
    "Er denne teksten grammatisk riktig? Mulige svar: {labels_str}.",
    "Er grammatikk riktig i setningen? Svar {labels_str}.",
]

LABELS_STR = "ja, nei"
PROMPT_TEMPLATE = "Setning: {text}\nGrammatisk korrekt: {label}"
INSTRUCTION_TEMPLATE = (
    "Setning: {text}\n\nBestem om setningen er grammatisk korrekt eller ikke. Svar med {labels_str}, og ikke noe annet."
)

def swap_words(sentence):
    """Swap 2-4 random words (not first) in the sentence."""
    words = sentence.strip().split()
    if len(words) < 4:
        # Can't meaningfully swap, return as is
        return sentence
    num_to_swap = random.randint(2, min(4, len(words)-1))
    idxs = list(range(1, len(words)))  # Never swap the first word
    swap_idxs = random.sample(idxs, num_to_swap)
    swapped_words = words[:]
    swap_order = swap_idxs[:]
    random.shuffle(swap_order)
    for i, j in zip(swap_idxs, swap_order):
        swapped_words[i], swapped_words[j] = swapped_words[j], swapped_words[i]
    return " ".join(swapped_words)

def process_file(input_file, output_file, use_chat_template, chat_template_model):
    if use_chat_template:
        if AutoTokenizer is None:
            raise ImportError("transformers not installed. Required for --chat_template.")
        tokenizer = AutoTokenizer.from_pretrained(chat_template_model, trust_remote_code=True)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(f"Model '{chat_template_model}' does not support apply_chat_template.")

    with open(input_file, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        lines = [json.loads(l) for l in fin if l.strip()]
        pbar = tqdm(total=len(lines), desc="Processing sentences")
        for i, rec in enumerate(lines):
            orig_text = rec["text"].strip()
            if not orig_text:
                pbar.update(1)
                continue
            # Randomly choose correct or incorrect for each line (or alternate)
            make_incorrect = (i % 2 == 1)
            if make_incorrect:
                sent = swap_words(orig_text)
                label = "nei"
            else:
                sent = orig_text
                label = "ja"
            prompt_prefix = random.choice(NB_PROMPTS).format(labels_str=LABELS_STR)
            if use_chat_template:
                chat_message = [
                    {"role": "system", "content": prompt_prefix},
                    {"role": "user", "content": INSTRUCTION_TEMPLATE.format(text=sent, labels_str=LABELS_STR)},
                    {"role": "assistant", "content": label}
                ]
                formatted = tokenizer.apply_chat_template(
                    [chat_message], tokenize=False, add_generation_prompt=False
                )[0]
                record = {
                    "language": "no",
                    "text": formatted,
                    "prompt": prompt_prefix,
                    "label": label,
                }
            else:
                record = {
                    "language": "no",
                    "text": prompt_prefix + "\n\n" + PROMPT_TEMPLATE.format(text=sent, label=label),
                    "prompt": prompt_prefix,
                    "label": label,
                }
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a Norwegian grammatical correctness dataset with prompt augmentation.')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file (field: "text")')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSONL')
    parser.add_argument('--chat_template', type=str, default=None, help='HF model for chat template formatting (optional)')
    args = parser.parse_args()
    process_file(
        args.input_file,
        args.output_file,
        use_chat_template=bool(args.chat_template),
        chat_template_model=args.chat_template
    )
