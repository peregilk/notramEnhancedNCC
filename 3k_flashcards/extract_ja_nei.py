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

AUGMENTATION_PHRASES = [
    "Svar med ja eller nei, og ikke noe annet.",
    "Kun ja eller nei som svar, ingenting mer.",
    "Ja eller nei, det er alt.",
    "Besvar med ja eller nei, og kun det.",
    "Svaret skal være ja eller nei, uten tillegg.",
    "Bare ja eller nei, takk.",
    "Ett ord: ja eller nei. Ikke mer.",
    "Svar kun \"ja\" eller \"nei\".",
    "Enten ja eller nei, ingenting utover det.",
    "Hold deg til ja eller nei.",
    "Vennligst svar med ja eller nei, og ingenting annet.",
    "Vi ber om svar med ja eller nei, og kun det.",
    "Kan du svare ja eller nei, uten ytterligere kommentarer?",
    "Svaret bes være ja eller nei, og ikke noe mer.",
    "Vær så snill, svar kun ja eller nei.",
    "Din respons bør være ja eller nei, og ikke noe utover det.",
    "Vennligst begrens svaret til ja eller nei.",
    "Et ja eller nei er tilstrekkelig, ikke noe mer.",
    "Vennligst oppgi ja eller nei, og intet annet.",
    "Vi forventer et svar som er enten ja eller nei, uten tillegg.",
    "Svar utelukkende med ja eller nei.",
    "Ikke noe annet enn ja eller nei aksepteres.",
    "Kun svaralternativene ja eller nei er gyldige.",
    "Svaret må begrenses til ja eller nei.",
    "Absolutt kun ja eller nei.",
    "Ingen andre ord enn ja eller nei.",
    "Ja eller nei. Punktum.",
    "Forventer kun ja eller nei, uten forklaringer.",
    "Svaret ditt skal kun inneholde \"ja\" eller \"nei\".",
    "Unngå alt annet enn ja eller nei i svaret.",
    "Du må svare ja eller nei, og ikke noe annet.",
    "Sørg for å svare kun med ja eller nei.",
    "Husk: kun ja eller nei som svar.",
    "Det er viktig at du svarer ja eller nei, og ikke noe mer.",
    "Svaret må være et enkelt ja eller nei.",
    "Gi et svar bestående av ja eller nei, og bare det.",
    "Svarformat: Ja/Nei. Ingenting annet.",
    "For dette spørsmålet: svar ja eller nei, og hold deg til det.",
    "Din oppgave er å svare ja eller nei, uten tilleggsinformasjon.",
    "Svar kortfattet: ja eller nei.",
    "Ja eller nei er de eneste akseptable svarene.",
    "Ingenting annet enn ja eller nei skal oppgis.",
    "Svar enten \"ja\" eller \"nei\", uten andre ord.",
    "Besvarelsen skal være ja eller nei, og kun det.",
    "Gi kun ja eller nei, intet mer.",
    "Ditt svar: ja eller nei. Ikke noe ekstra.",
    "Bekreft med ja eller nei, uten videre utdypning.",
    "Kun \"ja\" eller \"nei\" er tillatt som respons.",
    "Svar med ett av de to: ja, eller nei. Ikke noe annet.",
    "Hold svaret til ja eller nei.",
    "Ja/Nei. Kun det.",
    "Svar: Ja/Nei. Ikke mer.",
    "Kun ja/nei.",
    "Ja eller nei. Stopp der.",
    "Bare ja, eller bare nei.",
    "Ja eller nei. Ikke noe tillegg.",
    "Svar ja. Eller nei. Ikke noe annet.",
    "\"Ja\" eller \"Nei\". Det er alt.",
    "Kun ja eller nei.",
    "Enkelt svar: ja/nei.",
    "Svar ja eller nei. Ingen ytterligere detaljer.",
    "Ja eller nei, og utelukk alt annet.",
    "Svar med ja eller nei. Ingen kommentarer utover det.",
    "Bare ja eller nei, unngå all annen tekst.",
    "Ja eller nei. Ikke en stavelse mer.",
    "Svar ja eller nei. Ikke noe snikksnakk.",
    "Kun ja eller nei skal skrives.",
    "Svar ja eller nei, uten noen form for utbrodering.",
    "Ja eller nei. Ikke noe ved siden av.",
    "Svar med ja eller nei, og det alene.",
    "Si ja eller nei, og ferdig med det.",
    "Gi meg et ja eller nei, ikke noe dilldall.",
    "Ja eller nei. Ikke noe mer å si.",
    "Kom igjen, ja eller nei. Ikke noe annet.",
    "Bare gi et ja eller nei.",
    "Ja eller nei, takk. Ikke noe ekstra.",
    "Svarer du ja eller nei? Kun det.",
    "Ok, svar ja eller nei. Ikke noe mer.",
    "Rett på sak: ja eller nei.",
    "Hold det enkelt: ja eller nei.",
    "Svaret ditt må være enten ja eller nei, og ingenting utover dette.",
    "Vennligst gi kun et \"ja\" eller \"nei\" som tilbakemelding.",
    "Det forventes et svar som er ja eller nei, uten noen andre elementer.",
    "Du bes svare ja eller nei, og kun ja eller nei.",
    "Responsen din skal være ja eller nei, og ikke noe annet.",
    "Svar bekreftende (ja) eller avkreftende (nei), uten tillegg.",
    "Et enkelt \"ja\" eller \"nei\" er svaret vi ser etter, ingenting mer.",
    "Oppgi kun \"ja\" eller \"nei\" som svar.",
    "Vennligst hold deg strengt til ja eller nei.",
    "Svaret skal kun bestå av ordet \"ja\" eller ordet \"nei\".",
    "Forventet respons: kun ja eller nei.",
    "Svar med \"ja\" eller \"nei\". Ikke noe annet er nødvendig.",
    "Kun et ja eller et nei vil bli akseptert.",
    "Ja eller nei, uten noen form for kvalifisering.",
    "Vær vennlig å svare med enten ja eller nei, og ikke noe annet.",
    "Svaret ditt begrenses til ja eller nei. Ikke noe mer.",
    "Gi et klart ja eller nei, og ingenting i tillegg.",
    "Svaralternativer: Ja, Nei. Velg ett, og kun ett.",
    "Ditt bidrag: ja eller nei. Ikke noe mer enn det.",
    "Vennligst svar kun med \"ja\" eller \"nei\", og utelat all annen informasjon."
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and reformat JSONL records with binary (ja/nei) answers."
    )
    parser.add_argument("--input_file", "-i", required=True)
    parser.add_argument("--output_file", "-o", required=True)
    parser.add_argument("--chat_template", help="HF-modell med innebygget chat-template")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def extract_question_answer(text: str) -> Tuple[str, str]:
    lines = text.strip().split("\n", 1)
    if len(lines) != 2:
        return "", ""
    return lines[0].strip(), lines[1].strip()

def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower()
    if re.match(r"^ja([\s,.!?:;»”'\-]|$)", answer):
        return "ja"
    elif re.match(r"^nei([\s,.!?:;»”'\-]|$)", answer):
        return "nei"
    return ""

def process_record(
    record: Dict[str, Any],
    debug: bool = False,
    chat_template: str = None,
    tokenizer=None,
    system_prompt: str = None
) -> Tuple[List[Dict[str, str]], List[str], int, str]:
    errors: List[str] = []
    outputs: List[Dict[str, str]] = []

    base_id = record.get("id")
    if not isinstance(base_id, str):
        errors.append("Missing or invalid 'id'")
        return outputs, errors, 0, ""

    text = record.get("text")
    if not isinstance(text, str):
        errors.append(f"Record {base_id}: Missing or invalid 'text' field")
        return outputs, errors, 0, ""

    question, answer = extract_question_answer(text)
    if not question or not answer:
        errors.append(f"Record {base_id}: Could not extract question/answer")
        return outputs, errors, 0, text

    binary = normalize_answer(answer)
    if not binary:
        return outputs, errors, 0, ""  # Not an error, just not relevant

    augmentation = random.choice(AUGMENTATION_PHRASES)
    full_question = f"{question} {augmentation}"

    # Use chat template if enabled
    if chat_template and tokenizer:
        msgs = [
            {"role": "system",    "content": system_prompt or "You are a helpful assistant."},
            {"role": "user",      "content": full_question},
            {"role": "assistant", "content": binary},
        ]
        text_out = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    else:
        text_out = f"{full_question}\n{binary}"

    outputs.append({
        "id": base_id,
        "question": question,
        "augmentation": augmentation,
        "answer": binary,
        "text": text_out
    })
    return outputs, errors, 1, ""

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
                outputs, errors, count, raw_text = process_record(
                    record,
                    args.debug,
                    args.chat_template,
                    tokenizer,
                    system_prompt
                )
                extraction_counts[count] += 1

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
    for n in [1, 0]:
        print(f"  {n} item(s): {extraction_counts[n]}")

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
