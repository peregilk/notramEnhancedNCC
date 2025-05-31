#!/usr/bin/env python3
# create_bokmal_nynorsk.py

import argparse
import json
import logging
import sys
import random
from typing import Dict, Any, List
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

NB_LABELS = ["Bokmål:", "Norwegian Bokmål:", "Norsk bokmål:", "NB:"]
NN_LABELS = ["Nynorsk:", "Norwegian Nynorsk:", "NN:"]

NB_TO_NN_PROMPTS = [
    # Nynorsk prompts (Bokmål -> Nynorsk)
    "Omset denne teksta frå bokmål til nynorsk:",
    "Ver venleg og omset følgjande tekst frå bokmål til nynorsk:",
    "Kan du omsetja setninga under (på bokmål) til nynorsk?",
    "Omset frå bokmål til nynorsk:",
    "Skriv denne bokmålsteksten om til nynorsk.",
    "Lag ein nynorsk versjon av denne bokmålsteksten.",
    "Gje ei nynorsk omsetjing av dette (på bokmål):",
    "Eg treng denne teksta omsett frå bokmål til nynorsk:",
    "Ver venleg å skriv om teksten frå bokmål til nynorsk:",
    "Til nynorsk, omset denne bokmålsteksten:",
    "Kjeldespråk: Bokmål. Målspråk: Nynorsk. Omset teksten:",
    "Gje innhaldet på nynorsk (teksta er på bokmål):",
    "Formuler denne bokmålsteksten på nynorsk.",
    "Overfør tydinga i denne teksta frå bokmål til nynorsk.",
    "Lag ei nynorsk gjengiving av teksten under (på bokmål).",
    "Hjelp meg å omsetje dette frå bokmål til nynorsk.",
    "Ver snill og gjer bokmålsteksten om til nynorsk:",
    "Tekst på bokmål for nynorsk omsetjing:",
    "Omset denne passasjen frå bokmål til nynorsk.",
    # Bokmål/English prompts
    "Oversett denne teksten fra bokmål til nynorsk:",
    "Vennligst oversett følgende tekst fra bokmål til nynorsk:",
    "Kan du oversette teksten nedenfor (som er på bokmål) til nynorsk?",
    "Oversett fra bokmål til nynorsk:",
    "Gjengi denne bokmål-teksten på nynorsk.",
    "Konverter dette tekstutdraget fra bokmål til nynorsk.",
    "Oversettelse fra bokmål til nynorsk ønskes for:",
    "Jeg trenger denne teksten oversatt fra bokmål til nynorsk:",
    "Oversett følgende til nynorsk (teksten er på bokmål):",
    "Bokmål-tekst for oversettelse til nynorsk:",
    "Oversett denne teksten (skrevet på bokmål) til nynorsk:",
    "Vennligst gjør om følgende tekst, som er på bokmål, til nynorsk:",
    "Kan du gjengi teksten under, som er på bokmål, på nynorsk?",
    "Translate this text from Norwegian Bokmål to Norwegian Nynorsk:",
    "Please translate the following passage from Norwegian Bokmål into Norwegian Nynorsk.",
    "Could you convert this Norwegian Bokmål text to Norwegian Nynorsk?",
    "Render the text below from Norwegian Bokmål to Norwegian Nynorsk.",
    "Norwegian Bokmål to Norwegian Nynorsk translation task:",
    "Translate from Norwegian Bokmål to Norwegian Nynorsk:",
    "I need a Norwegian Nynorsk translation of the following Norwegian Bokmål text:",
    "Provide a Norwegian Nynorsk version of the text (originally in Norwegian Bokmål):",
    "Source: Norwegian Bokmål. Target: Norwegian Nynorsk. Translate the provided text:",
    "Translate this Norwegian Bokmål text to Norwegian Nynorsk:",
    "Please convert the following text, which is in Norwegian Bokmål, into Norwegian Nynorsk.",
    "Could you render the passage below (in Norwegian Bokmål) in Norwegian Nynorsk?",
    "To Norwegian Nynorsk, please translate this Norwegian Bokmål text:",
    "Here is a text in Norwegian Bokmål. Please provide the Norwegian Nynorsk translation:",
    "Task: Translate the following from its original language (Norwegian Bokmål) to Norwegian Nynorsk.",
    "Rephrase this Norwegian Bokmål text in Norwegian Nynorsk.",
    "Convert the following from Norwegian Bokmål into Norwegian Nynorsk:",
    "Your task is to translate the upcoming Norwegian Bokmål text into Norwegian Nynorsk.",
    "Norwegian Nynorsk rendition needed for this Norwegian Bokmål text:",
    "Process the following Norwegian Bokmål text and output its Norwegian Nynorsk equivalent.",
    "Norwegian Bokmål -> Norwegian Nynorsk: Translate.",
    "Assist me by translating this from Norwegian Bokmål to Norwegian Nynorsk.",
    "Kindly transform the Norwegian Bokmål input into Norwegian Nynorsk.",
    "Provide the Norwegian Nynorsk translation for the Norwegian Bokmål content below:",
    "Transcribe this passage from Norwegian Bokmål into Norwegian Nynorsk."
]

NN_TO_NB_PROMPTS = [
    # Bokmål prompts (Nynorsk -> Bokmål)
    "Oversett denne teksten fra nynorsk til bokmål:",
    "Vennligst oversett følgende tekst fra nynorsk til bokmål:",
    "Kan du oversette teksten nedenfor (som er på nynorsk) til bokmål?",
    "Oversett fra nynorsk til bokmål:",
    "Gjengi denne nynorsk-teksten på bokmål.",
    "Konverter dette tekstutdraget fra nynorsk til bokmål.",
    "Oversettelse fra nynorsk til bokmål ønskes for:",
    "Jeg trenger denne teksten oversatt fra nynorsk til bokmål:",
    "Oversett følgende til bokmål (teksten er på nynorsk):",
    "Nynorsk-tekst for oversettelse til bokmål:",
    "Oversett denne teksten (skrevet på nynorsk) til bokmål:",
    "Vennligst gjør om følgende tekst, som er på nynorsk, til bokmål:",
    "Kan du gjengi teksten under, som er på nynorsk, på bokmål?",
    # Nynorsk prompts (Nynorsk -> Bokmål)
    "Omset denne teksta frå nynorsk til bokmål:",
    "Ver venleg og omset følgjande tekst frå nynorsk til bokmål:",
    "Kan du omsetja setninga under (på nynorsk) til bokmål?",
    "Omset frå nynorsk til bokmål:",
    "Skriv denne nynorskteksten om til bokmål.",
    "Lag ein bokmålsversjon av denne nynorskteksten.",
    "Gje ei bokmålsomsetjing av dette (på nynorsk):",
    "Eg treng denne teksta omsett frå nynorsk til bokmål:",
    "Ver venleg å skriv om teksten frå nynorsk til bokmål:",
    "Til bokmål, omset denne nynorskteksten:",
    "Kjeldespråk: Nynorsk. Målspråk: Bokmål. Omset teksten:",
    "Gje innhaldet på bokmål (teksta er på nynorsk):",
    "Formuler denne nynorskteksten på bokmål.",
    "Overfør tydinga i denne teksta frå nynorsk til bokmål.",
    "Lag ei bokmåls gjengiving av teksten under (på nynorsk).",
    "Hjelp meg å omsetje dette frå nynorsk til bokmål.",
    "Ver snill og gjer nynorskteksten om til bokmål:",
    "Tekst på nynorsk for bokmålsomsetjing:",
    "Omset denne passasjen frå nynorsk til bokmål.",
    # English
    "Translate this text from Norwegian Nynorsk to Norwegian Bokmål:",
    "Please translate the following passage from Norwegian Nynorsk into Norwegian Bokmål.",
    "Could you convert this Norwegian Nynorsk text to Norwegian Bokmål?",
    "Render the text below from Norwegian Nynorsk to Norwegian Bokmål.",
    "Norwegian Nynorsk to Norwegian Bokmål translation task:",
    "Translate from Norwegian Nynorsk to Norwegian Bokmål:",
    "I need a Norwegian Bokmål translation of the following Norwegian Nynorsk text:",
    "Provide a Norwegian Bokmål version of the text (originally in Norwegian Nynorsk):",
    "Source: Norwegian Nynorsk. Target: Norwegian Bokmål. Translate the provided text:",
    "Translate this Norwegian Nynorsk text to Norwegian Bokmål:",
    "Please convert the following text, which is in Norwegian Nynorsk, into Norwegian Bokmål.",
    "Could you render the passage below (in Norwegian Nynorsk) in Norwegian Bokmål?",
    "To Norwegian Bokmål, please translate this Norwegian Nynorsk text:",
    "Here is a text in Norwegian Nynorsk. Please provide the Norwegian Bokmål translation:",
    "Task: Translate the following from its original language (Norwegian Nynorsk) to Norwegian Bokmål.",
    "Rephrase this Norwegian Nynorsk text in Norwegian Bokmål.",
    "Convert the following from Norwegian Nynorsk into Norwegian Bokmål:",
    "Your task is to translate the upcoming Norwegian Nynorsk text into Norwegian Bokmål.",
    "Norwegian Bokmål rendition needed for this Norwegian Nynorsk text:",
    "Process the following Norwegian Nynorsk text and output its Norwegian Bokmål equivalent.",
    "Norwegian Nynorsk -> Norwegian Bokmål: Translate.",
    "Assist me by translating this from Norwegian Nynorsk to Norwegian Bokmål.",
    "Kindly transform the Norwegian Nynorsk input into Norwegian Bokmål.",
    "Provide the Norwegian Bokmål translation for the Norwegian Nynorsk content below:",
    "Transcribe this passage from Norwegian Nynorsk into Norwegian Bokmål."
]

DEFAULT_BATCH_SIZE = 5000

def count_lines(filename: str) -> int:
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def create_chat_messages(record: Dict[str, Any], swap: bool, system_prompt_str: str):
    nb, nn = record.get("nb", ""), record.get("nn", "")
    if not nb or not nn:
        return None, None
    if not swap:
        augmentation = random.choice(NB_TO_NN_PROMPTS)
        user_content = f"{augmentation}\n{nb}"
        assistant_content = nn
    else:
        augmentation = random.choice(NN_TO_NB_PROMPTS)
        user_content = f"{augmentation}\n{nn}"
        assistant_content = nb
    return [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ], augmentation

def process_record_standard(record: Dict[str, Any], swap: bool) -> Dict[str, Any]:
    nb, nn = record.get("nb", ""), record.get("nn", "")
    if not nb or not nn: return None
    out = dict(record)
    nb_label, nn_label = random.choice(NB_LABELS), random.choice(NN_LABELS)
    out["text"] = f"{nb_label} {nb}\n{nn_label} {nn}" if not swap else f"{nn_label} {nn}\n{nb_label} {nb}"
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Format and augment a parallel Bokmål–Nynorsk translation dataset."
    )
    parser.add_argument("--input_file", "-i", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", "-o", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--chat_template", help="HF model identifier with a built-in chat template.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for chat template processing (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--no_count_lines", action="store_true", help="Skip initial line count for faster startup.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")

    tokenizer = None
    system_prompt_for_chat = "Du er ein hjelpsam språkassistent."
    process_as_chat = False

    if args.chat_template:
        if AutoTokenizer is None:
            logging.error("transformers not installed, cannot use --chat_template")
            sys.exit(1)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
            if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
                process_as_chat = True
                if "Llama-3" in args.chat_template:
                    system_prompt_for_chat = "You are a helpful AI assistant."
        except Exception as e:
            logging.error(f"Error loading tokenizer '{args.chat_template}': {e}", exc_info=args.debug)

    total_input = 0
    total_output = 0
    error_list = []

    num_lines = 0
    if not args.no_count_lines:
        try:
            num_lines = count_lines(args.input_file)
        except Exception:
            pass

    with open(args.input_file, "r", encoding="utf-8") as infile, \
         open(args.output_file, "w", encoding="utf-8") as outfile, \
         tqdm(desc="Processing lines", unit="lines", total=(num_lines or None)) as pbar:

        record_batch_originals = []
        messages_batch_for_template = []
        augmentation_batch_info = []
        output_lines_buffer = []

        for line_num, line in enumerate(infile, 1):
            total_input += 1
            line = line.strip()
            if not line:
                pbar.update(1)
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                error_list.append((line_num, f"JSON load: {e}. L: '{line[:100]}...'"))
                pbar.update(1)
                continue

            swap = bool(random.getrandbits(1))
            if process_as_chat:
                messages, augmentation = create_chat_messages(record, swap, system_prompt_for_chat)
                if messages:
                    record_batch_originals.append(dict(record))
                    messages_batch_for_template.append(messages)
                    augmentation_batch_info.append(augmentation)
                if len(messages_batch_for_template) >= args.batch_size:
                    try:
                        formatted_texts = tokenizer.apply_chat_template(
                            messages_batch_for_template, tokenize=False, add_generation_prompt=False
                        )
                        for i, original_rec in enumerate(record_batch_originals):
                            out_record = original_rec
                            out_record["augmentation"] = augmentation_batch_info[i]
                            out_record["text"] = formatted_texts[i]
                            output_lines_buffer.append(json.dumps(out_record, ensure_ascii=False))
                            total_output += 1
                        outfile.write("\n".join(output_lines_buffer) + "\n")
                    except Exception as e:
                        logging.error(f"Error applying chat template batch: {e}", exc_info=args.debug)
                        for i_err, _ in enumerate(record_batch_originals):
                            error_list.append((f"Batch item {i_err} in failing batch", f"Failed batch: {e}"))
                    finally:
                        record_batch_originals.clear()
                        messages_batch_for_template.clear()
                        augmentation_batch_info.clear()
                        output_lines_buffer.clear()
            else:
                out_record = process_record_standard(record, swap)
                if out_record:
                    outfile.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    total_output += 1
            pbar.update(1)

        # Final batch processing for chat mode
        if process_as_chat and messages_batch_for_template:
            try:
                formatted_texts = tokenizer.apply_chat_template(
                    messages_batch_for_template, tokenize=False, add_generation_prompt=False
                )
                for i, original_rec in enumerate(record_batch_originals):
                    out_record = original_rec
                    out_record["augmentation"] = augmentation_batch_info[i]
                    out_record["text"] = formatted_texts[i]
                    output_lines_buffer.append(json.dumps(out_record, ensure_ascii=False))
                    total_output += 1
                if output_lines_buffer:
                    outfile.write("\n".join(output_lines_buffer) + "\n")
            except Exception as e:
                logging.error(f"Error applying chat template final batch: {e}", exc_info=args.debug)
                for i_err, _ in enumerate(record_batch_originals):
                    error_list.append((f"Final batch item {i_err}", f"Failed final batch: {e}"))
            finally:
                record_batch_originals.clear()
                messages_batch_for_template.clear()
                augmentation_batch_info.clear()
                output_lines_buffer.clear()

    print(f"\n--- Processing Summary ---")
    print(f"Total input lines read: {total_input}")
    print(f"Total output lines generated: {total_output}")
    if error_list:
        print(f"\nEncountered {len(error_list)} errors.")
        max_errors_to_show = 10
        for i, (line_info, err) in enumerate(error_list[:max_errors_to_show]):
            print(f"  Err {i+1} (Line/Info: {line_info}): {err}")
        if len(error_list) > max_errors_to_show:
            print(f"  ... and {len(error_list) - max_errors_to_show} more errors.")
    else:
        print("No errors encountered.")

if __name__ == "__main__":
    main()
