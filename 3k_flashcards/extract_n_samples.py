#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
import random
from typing import List, Dict, Tuple, Any
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

AUGMENTATION_TEMPLATES = [
    "Begrens svaret til maksimalt {n} ord.",
    "Svaret ditt skal ikke ha mer enn {n} ord.",
    "Hold deg til maks {n} ord i svaret.",
    "Maksimalt {n} ord for svaret ditt.",
    "Svar med høyst {n} ord.",
    "Ikke bruk mer enn {n} ord i svaret.",
    "Svaret kan ikke inneholde mer enn {n} ord.",
    "Responsen din bør ha maksimalt {n} ord.",
    "Gi et svar på høyst {n} ord.",
    "Svaret må ikke overstige {n} ord.",
    "Sørg for at svaret ikke er lengre enn {n} ord.",
    "Pass på at svaret ikke overstiger {n} ord.",
    "Svaret skal ikke være på mer enn {n} ord.",
    "Ikke la svaret bli lengre enn {n} ord.",
    "Unngå å bruke flere enn {n} ord.",
    "Antall ord i svaret må ikke overskride {n}.",
    "Svaret ditt skal ikke gå utover {n} ord.",
    "Ikke skriv mer enn {n} ord.",
    "Svarets lengde skal ikke overstige {n} ord.",
    "Vær nøye med å ikke bruke mer enn {n} ord.",
    "Svaret må være på maksimalt {n} ord.",
    "En øvre grense på {n} ord gjelder for svaret.",
    "Svar innenfor en ramme på høyst {n} ord.",
    "{n} ord er den maksimale lengden på svaret.",
    "Maksimum {n} ord i din besvarelse.",
    "Bruk høyst {n} ord til å svare.",
    "Svarets omfang: maksimalt {n} ord.",
    "Forventet lengde: høyst {n} ord.",
    "Ikke overskrid en grense på {n} ord.",
    "Svar på maks {n} ord.",
    "Hold svaret innenfor {n} ord.",
    "Svaret må være {n} ord eller færre.",
    "Svar innenfor en grense på {n} ord.",
    "Formuler svaret med {n} ord eller mindre.",
    "Svaret skal holdes innenfor {n} ord.",
    "Ikke gå over en grense på {n} ord.",
    "Sørg for at svaret er på {n} ord eller kortere.",
    "Hold deg innenfor {n}-ordsgrensen.",
    "Svaret ditt skal være {n} ord eller under.",
    "Forbli innenfor {n} ord i svaret.",
    "Du må begrense svaret til {n} ord.",
    "Svaret skal være på maks {n} ord.",
    "Det er viktig at svaret ikke overskrider {n} ord.",
    "Sørg for at ordantallet ikke overstiger {n}.",
    "Svaret må ikke være på flere enn {n} ord.",
    "Husk: maksimalt {n} ord i svaret.",
    "Din oppgave: svar med maks {n} ord.",
    "Følg instruksen: høyst {n} ord.",
    "Du bes begrense svaret til {n} ord.",
    "Svaret skal ikke inneholde flere enn {n} ord.",
    "Maks {n} ord.",
    "Høyst {n} ord.",
    "Ikke over {n} ord.",
    "Svar: maks {n} ord.",
    "{n} ord maks.",
    "Grense: {n} ord.",
    "Inntil {n} ord.",
    "Svarlengde: maks {n} ord.",
    "Ikke flere enn {n} ord.",
    "{n} ord eller mindre.",
    "Vennligst begrens svaret til maksimalt {n} ord.",
    "Vi ber om at svaret ikke overstiger {n} ord.",
    "Kan du holde svaret på høyst {n} ord?",
    "Vennligst svar med ikke mer enn {n} ord.",
    "Svaret bes holdes innenfor {n} ord.",
    "Vi anmoder om et svar på maksimalt {n} ord.",
    "Vennligst sørg for at svaret ikke inneholder mer enn {n} ord.",
    "Det forventes et svar på høyst {n} ord.",
    "Vær vennlig å begrense responsen til {n} ord.",
    "Vi setter pris på om svaret holdes under {n} ord (eller til {n} ord).",
    "Ordtellingen for svaret skal ikke overstige {n}.",
    "Hold ordantallet i svaret på maks {n}.",
    "Svarets ordantall må ikke være mer enn {n}.",
    "Maksimalt ordantall for svar er {n}.",
    "Ordlengden på svaret er begrenset til {n}.",
    "Respekter ordgrensen på {n} ord.",
    "Svaret ditt må overholde en ordgrense på {n}.",
    "Antall ord i svaret må holdes til {n} eller færre.",
    "En ordgrense på {n} gjelder for dette svaret.",
    "Ikke overskrid det tillatte ordantallet på {n}.",
    "For svaret ditt gjelder en maksgrense på {n} ord.",
    "Svar kortfattet, innenfor {n} ord.",
    "Svaret skal ikke være unødvendig langt, maks {n} ord.",
    "{n} ord er det meste du kan bruke i svaret.",
    "Svar med {n} ord som øvre grense.",
    "Ditt svar: ikke lengre enn {n} ord.",
    "Bruk maksimalt {n} ord i din respons.",
    "Svarets lengde må ikke overgå {n} ord.",
    "Svar, men ikke med flere enn {n} ord.",
    "Hold deg til {n} ord, eller færre, i svaret.",
    "Svar presist og ikke utover {n} ord.",
    "Formuler deg innenfor en ramme på {n} ord.",
    "Svaret ditt skal ikke omfatte mer enn {n} ord.",
    "En begrensning på {n} ord er satt for svaret.",
    "Vennligst hold deg til {n} ord eller mindre i besvarelsen.",
    "Svar med et ordantall som ikke er høyere enn {n}.",
    "Svar så konsist som mulig, maks {n} ord.",
    "Den absolutte grensen for ord i svaret er {n}.",
    "Ikke skriv et svar som er lengre enn {n} ord.",
    "Svaret må være begrenset til {n} ord."
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ekstraher kun svar med 2–6 ord og lag augmentert JSONL."
    )
    parser.add_argument("--input_file", "-i", required=True)
    parser.add_argument("--output_file", "-o", required=True)
    parser.add_argument("--min_words", type=int, default=2)
    parser.add_argument("--max_words", type=int, default=6)
    parser.add_argument("--chat_template", help="HF-modell med innebygget chat-template")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def extract_question_answer(text: str) -> Tuple[str, str]:
    lines = text.strip().split("\n", 1)
    if len(lines) != 2:
        return "", ""
    return lines[0].strip(), lines[1].strip()

def count_words(answer: str) -> int:
    return len(re.findall(r"\b\w+\b", answer, flags=re.UNICODE))

def process_record(
    record: Dict[str, Any],
    min_words: int,
    max_words: int,
    debug: bool = False,
    chat_template: str = None,
    tokenizer=None,
    system_prompt: str = None
) -> Tuple[List[Dict[str, str]], List[str], int, str, int]:
    errors: List[str] = []
    outputs: List[Dict[str, str]] = []

    base_id = record.get("id")
    if not isinstance(base_id, str):
        errors.append("Missing or invalid 'id'")
        return outputs, errors, 0, "", 0

    text = record.get("text")
    if not isinstance(text, str):
        errors.append(f"Record {base_id}: Missing or invalid 'text' field")
        return outputs, errors, 0, "", 0

    question, answer = extract_question_answer(text)
    if not question or not answer:
        errors.append(f"Record {base_id}: Could not extract question/answer")
        return outputs, errors, 0, text, 0

    num_words = count_words(answer)
    if debug:
        print(f"DEBUG: answer='{answer}' | num_words={num_words}")

    if not (min_words <= num_words <= max_words):
        return outputs, errors, 0, "", num_words

    augmentation = random.choice(AUGMENTATION_TEMPLATES).format(n=num_words)
    full_question = f"{question} {augmentation}"

    # Use chat template if enabled
    if chat_template and tokenizer:
        msgs = [
            {"role": "system",    "content": system_prompt or "You are a helpful assistant."},
            {"role": "user",      "content": full_question},
            {"role": "assistant", "content": answer},
        ]
        text_out = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    else:
        text_out = f"{full_question}\n{answer}"

    outputs.append({
        "id": base_id,
        "question": question,
        "augmentation": augmentation,
        "answer": answer,
        "text": text_out
    })
    return outputs, errors, 1, "", num_words

def count_lines(filename: str) -> int:
    with open(filename, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f, 1):
            pass
    return i if 'i' in locals() else 0

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    # Load tokenizer if chat_template is used
    tokenizer = None
    system_prompt = None
    if args.chat_template:
        if AutoTokenizer is None:
            sys.exit("transformers må være installert for --chat_template")
        tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template, trust_remote_code=True
        )
        if not hasattr(tokenizer, "apply_chat_template"):
            sys.exit("Tokenizer mangler apply_chat_template; velg chatmodell")
        system_prompt = getattr(tokenizer, "default_system_prompt", "You are a helpful assistant.")

    total_input = 0
    total_output = 0
    error_list: List[Tuple[int, str, str, str]] = []
    extraction_counts = Counter()
    word_stats = Counter()

    try:
        num_lines = count_lines(args.input_file)
        with open(args.input_file, "r", encoding="utf-8") as infile, \
             open(args.output_file, "w", encoding="utf-8") as outfile, \
             tqdm(total=num_lines, desc="Processing lines", unit="lines") as pbar:
            for line_num, line in enumerate(infile, 1):
                total_input += 1
                line = line.strip()
                if not line:
                    extraction_counts[0] += 1
                    pbar.update(1)
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    error_list.append((line_num, "N/A", f"JSON decode error: {e}", ""))
                    extraction_counts[0] += 1
                    pbar.update(1)
                    continue

                record_id = record.get("id", "UNKNOWN")
                outputs, errors, count, raw_text, num_words = process_record(
                    record,
                    args.min_words,
                    args.max_words,
                    args.debug,
                    args.chat_template,
                    tokenizer,
                    system_prompt,
                )
                extraction_counts[count] += 1

                if count == 1:
                    word_stats[num_words] += 1

                for out in outputs:
                    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                    total_output += 1

                for err in errors:
                    error_list.append((line_num, record_id, err, raw_text))
                pbar.update(1)

    except IOError as e:
        logging.error(f"File error: {e}")
        sys.exit(1)

    print(f"\nProcessed {total_input} input lines.")
    print(f"Generated {total_output} output lines.")
    print("\nExtraction count distribution (lines yielding 1 relevant record vs others):")
    for n in range(args.max_words + 1):
        print(f"  {n} item(s): {extraction_counts[n]}")

    print(f"\nWord count distribution in answers kept:")
    for n in range(args.min_words, args.max_words + 1):
        print(f"  {n} ord: {word_stats[n]}")

    print(f"\nEncountered {len(error_list)} errors.")
    if args.debug and error_list:
        print("\n--- First 10 detailed debug errors ---")
        for i, (line_num, record_id, err, raw_text) in enumerate(error_list[:10]):
            print(f"\nRecord #{line_num} ID: {record_id}")
            print(f"Error: {err}")
            if raw_text:
                print("Full 'text' field:")
                print(raw_text)

if __name__ == "__main__":
    main()
