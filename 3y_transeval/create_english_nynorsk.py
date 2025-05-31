#!/usr/bin/env python3
# create_english_nynorsk.py (Corrected Logic - Efficient Loop Condition, Nynorsk Edition)

import argparse
import logging
import sys
import random
from typing import Dict, Any, List

# Using orjson if available
try:
    import orjson
    JSON_LIB_USED = "orjson"
    def json_loads(s: str) -> Any: return orjson.loads(s)
    def json_dumps(obj: Any) -> str: return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json as std_json
    JSON_LIB_USED = "standard json"
    def json_loads(s: str) -> Any: return std_json.loads(s)
    def json_dumps(obj: Any) -> str: return std_json.dumps(obj, ensure_ascii=False)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm"); sys.exit(1)
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None # Will be checked later

NN_LABELS = ["Nynorsk:"]
EN_LABELS = ["English:", "Engelsk:"]

# Prompt-sets, ca. halvparten på nynorsk, halvparten på bokmål/norsk
NN_PROMPTS_TO_EN = [
    # Nynorsk -> English (nynorsk/engelsk)
    "Omset denne teksta frå nynorsk til engelsk:",
    "Ver venleg og omset denne teksten frå nynorsk til engelsk:",
    "Kan du omsetja setninga under (på nynorsk) til engelsk?",
    "Omset frå nynorsk til engelsk:",
    "Skriv denne nynorskteksta på engelsk.",
    "Lag ein engelsk versjon av denne nynorskteksten.",
    "Gje ei engelsk omsetjing av dette (på nynorsk):",
    "Eg treng denne teksta omsett frå nynorsk til engelsk:",
    "Ver venleg å skriv om teksten frå nynorsk til engelsk:",
    "Til engelsk, omset denne nynorskteksten:",
    "Kjeldespråk: Nynorsk. Målspråk: Engelsk. Omset teksten:",
    "Gje innhaldet på engelsk (teksta er på nynorsk):",
    "Formuler denne nynorskteksten på engelsk.",
    "Overfør tydinga i denne teksta frå nynorsk til engelsk.",
    "Lag ei engelsk gjengiving av teksten under (på nynorsk).",
    "Hjelp meg å omsetje dette frå nynorsk til engelsk.",
    "Ver snill og gjer nynorskteksten om til engelsk:",
    "Tekst på nynorsk for engelsk omsetjing:",
    "Omset denne passasjen frå nynorsk til engelsk.",
    # Bokmål-aktig for variasjon
    "Oversett denne teksten fra nynorsk til engelsk:",
    "Vennligst oversett følgende tekst fra nynorsk til engelsk:",
    "Kan du oversette teksten nedenfor (som er på nynorsk) til engelsk?",
    "Oversett fra nynorsk til engelsk:",
    "Gjengi denne nynorsk-teksten på engelsk.",
    "Konverter dette tekstutdraget fra nynorsk til engelsk.",
    "Oversettelse fra nynorsk til engelsk ønskes for:",
    "Jeg trenger denne teksten oversatt fra nynorsk til engelsk:",
    "Oversett følgende til engelsk (teksten er på nynorsk):",
    "Nynorsk-tekst for oversettelse til engelsk:",
    "Oversett denne teksten (skrevet på nynorsk) til engelsk:",
    "Vennligst gjør om følgende tekst, som er på nynorsk, til engelsk:",
    "Kan du gjengi teksten under, som er på nynorsk, på engelsk?",
    # English
    "Translate this text from Norwegian Nynorsk to English:",
    "Please translate the following passage from Norwegian Nynorsk into English.",
    "Could you convert this Norwegian Nynorsk text to English?",
    "Render the text below from Norwegian Nynorsk to English.",
    "Norwegian Nynorsk to English translation task:",
    "Translate from Norwegian Nynorsk to English:",
    "I need an English translation of the following Norwegian Nynorsk text:",
    "Provide an English version of the text (originally in Norwegian Nynorsk):",
    "Source: Norwegian Nynorsk. Target: English. Translate the provided text:",
    "Translate this Norwegian Nynorsk text to English:",
    "Please convert the following text, which is in Norwegian Nynorsk, into English.",
    "Could you render the passage below (in Norwegian Nynorsk) in English?",
    "To English, please translate this Norwegian Nynorsk text:",
    "Here is a text in Norwegian Nynorsk. Please provide the English translation:",
    "Task: Translate the following from its original language (Norwegian Nynorsk) to English.",
    "Rephrase this Norwegian Nynorsk text in English.",
    "Convert the following from Norwegian Nynorsk into English:",
    "Your task is to translate the upcoming Norwegian Nynorsk text into English.",
    "English rendition needed for this Norwegian Nynorsk text:",
    "Process the following Norwegian Nynorsk text and output its English equivalent.",
    "Norwegian Nynorsk -> English: Translate.",
    "Assist me by translating this from Norwegian Nynorsk to English.",
    "Kindly transform the Norwegian Nynorsk input into English.",
    "Provide the English translation for the Norwegian Nynorsk content below:",
    "Transcribe this passage from Norwegian Nynorsk into English."
]

EN_PROMPTS_TO_NN = [
    # English -> Nynorsk (nynorsk/engelsk)
    "Omset denne teksta frå engelsk til nynorsk:",
    "Ver venleg og omset følgjande tekst frå engelsk til nynorsk:",
    "Kan du omsetja setninga under (på engelsk) til nynorsk?",
    "Omset frå engelsk til nynorsk:",
    "Skriv denne engelske teksten om til nynorsk.",
    "Lag ein nynorsk versjon av denne engelske teksten.",
    "Gje ei nynorsk omsetjing av dette (på engelsk):",
    "Eg treng denne teksta omsett frå engelsk til nynorsk:",
    "Ver venleg å skriv om teksten frå engelsk til nynorsk:",
    "Til nynorsk, omset denne engelske teksten:",
    "Kjeldespråk: Engelsk. Målspråk: Nynorsk. Omset teksten:",
    "Gje innhaldet på nynorsk (teksta er på engelsk):",
    "Formuler denne engelske teksten på nynorsk.",
    "Overfør tydinga i denne teksta frå engelsk til nynorsk.",
    "Lag ei nynorsk gjengiving av teksten under (på engelsk).",
    "Hjelp meg å omsetje dette frå engelsk til nynorsk.",
    "Ver snill og gjer om den engelske teksten til nynorsk:",
    "Tekst på engelsk for nynorsk omsetjing:",
    "Omset denne passasjen frå engelsk til nynorsk.",
    # Bokmål-aktig for variasjon
    "Oversett denne teksten fra engelsk til nynorsk:",
    "Vennligst oversett følgende tekst fra engelsk til nynorsk:",
    "Kan du oversette teksten nedenfor fra engelsk til nynorsk?",
    "Oversett fra engelsk til nynorsk:",
    "Gjengi denne engelske teksten på nynorsk.",
    "Konverter dette tekstutdraget fra engelsk til nynorsk.",
    "Oversettelse fra engelsk til nynorsk ønskes for:",
    "Jeg trenger denne teksten oversatt fra engelsk til nynorsk:",
    "Oversett følgende til nynorsk (teksten er på engelsk):",
    "Engelsk tekst for oversettelse til nynorsk:",
    "Oversett denne teksten (fra engelsk) til nynorsk:",
    "Vennligst gjør om følgende engelske tekst til nynorsk:",
    "Kan du gjengi teksten under (som er på engelsk) på nynorsk?",
    # English
    "Translate this text from English to Norwegian Nynorsk:",
    "Please translate the following passage from English into Norwegian Nynorsk.",
    "Could you convert this English text to Norwegian Nynorsk?",
    "Render the text below from English to Norwegian Nynorsk.",
    "English to Norwegian Nynorsk translation task:",
    "Translate from English to Norwegian Nynorsk:",
    "I need a Norwegian Nynorsk translation of the following English text:",
    "Provide a Norwegian Nynorsk version of the text (originally in English):",
    "Source: English. Target: Norwegian Nynorsk. Translate the provided text:",
    "Translate this English text to Norwegian Nynorsk:",
    "Please convert the following text into Norwegian Nynorsk.",
    "Could you render the passage below in Norwegian Nynorsk?",
    "To Norwegian Nynorsk, please translate the text:",
    "Here is a text in English. Please provide the Norwegian Nynorsk translation:",
    "Task: Translate the following from its original language (English) to Norwegian Nynorsk.",
    "Rephrase this English text in Norwegian Nynorsk.",
    "Convert the following from English into Norwegian Nynorsk:",
    "Your task is to translate the upcoming English text into Norwegian Nynorsk.",
    "Norwegian Nynorsk rendition needed for this English text:",
    "Process the following English text and output its Norwegian Nynorsk equivalent.",
    "English -> Norwegian Nynorsk: Translate.",
    "Assist me by translating this from English to Norwegian Nynorsk.",
    "Kindly transform the English input into Norwegian Nynorsk.",
    "Provide the Norwegian Nynorsk translation for the English content below:",
    "Transcribe this passage from English into Norwegian Nynorsk."
]

DEFAULT_BATCH_SIZE = 5000

def count_lines(filename: str) -> int:
    try:
        with open(filename, "r", encoding="utf-8") as f: return sum(1 for _ in f)
    except FileNotFoundError: return 0

def create_chat_messages(record: Dict[str, Any], swap: bool, system_prompt_str: str) -> (List[Dict[str, str]], str):
    source, target = record.get("source", ""), record.get("target", "")
    if not source or not target: return None, None
    augmentation_choice = None
    if not swap:
        augmentation_choice = random.choice(EN_PROMPTS_TO_NN)
        user_content = f"{augmentation_choice}\n{source}"
        assistant_content = target
    else:
        augmentation_choice = random.choice(NN_PROMPTS_TO_EN)
        user_content = f"{augmentation_choice}\n{target}"
        assistant_content = source
    return [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ], augmentation_choice

def process_record_standard(record: Dict[str, Any], swap: bool) -> Dict[str, Any]:
    source, target = record.get("source", ""), record.get("target", "")
    if not source or not target: return None
    out = dict(record)
    en_label, nn_label = random.choice(EN_LABELS), random.choice(NN_LABELS)
    out["text"] = f"{en_label} {source}\n{nn_label} {target}" if not swap else f"{nn_label} {target}\n{en_label} {source}"
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Format and augment a parallel English–Nynorsk translation dataset."
    )
    parser.add_argument("--input_file", "-i", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", "-o", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--chat_template", help="HF model identifier with a built-in chat template.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for chat template processing (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--no_count_lines", action="store_true", help="Skip initial line count for faster startup.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Using {JSON_LIB_USED} for JSON operations.")

    tokenizer = None 
    system_prompt_for_chat = "You are a helpful assistant." 
    process_as_chat = False

    if args.chat_template:
        if AutoTokenizer is None:
            logging.error("The 'transformers' library must be installed for --chat_template. Run: pip install transformers")
            sys.exit(1)
        try:
            logging.info(f"Loading tokenizer for: {args.chat_template}")
            tokenizer = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
            logging.info(f"Tokenizer {args.chat_template} loaded successfully.")
            if not hasattr(tokenizer, "apply_chat_template") or not callable(tokenizer.apply_chat_template):
                logging.error(f"Tokenizer for '{args.chat_template}' lacks a callable 'apply_chat_template' method. Cannot proceed in chat mode.")
            else:
                process_as_chat = True
                if "Llama-3" in args.chat_template:
                    system_prompt_for_chat = "You are a helpful AI assistant."
                logging.info(f"CHAT MODE ACTIVE. Using chat template from '{args.chat_template}' with system prompt: \"{system_prompt_for_chat}\", Batch size: {args.batch_size}")
        except Exception as e:
            logging.error(f"Error loading tokenizer '{args.chat_template}': {e}. Cannot proceed in chat mode.", exc_info=args.debug)

    if not process_as_chat:
        logging.info("STANDARD MODE ACTIVE (either --chat_template not specified or tokenizer/method validation failed).")

    total_input = 0
    total_output = 0
    error_list = []
    pbar = None 

    try:
        num_lines = 0
        if not args.no_count_lines:
            logging.info(f"Counting lines in '{args.input_file}'...")
            num_lines = count_lines(args.input_file)
            if num_lines == 0 and args.input_file:
                 try:
                    with open(args.input_file, "r", encoding="utf-8") as f_check:
                        if not f_check.read(1): logging.warning(f"Input file '{args.input_file}' is empty.")
                 except FileNotFoundError: logging.error(f"Input file '{args.input_file}' not found."); sys.exit(1)
            logging.info(f"Found {num_lines} lines in '{args.input_file}'.")
        else: logging.info("Skipping initial line count.")

        with open(args.input_file, "r", encoding="utf-8") as infile, \
             open(args.output_file, "w", encoding="utf-8") as outfile:
            pbar_params = {"desc": "Processing lines", "unit": "lines"}
            if num_lines > 0: pbar_params["total"] = num_lines
            pbar = tqdm(**pbar_params)
            
            record_batch_originals = [] 
            messages_batch_for_template = [] 
            augmentation_batch_info = [] 
            output_lines_buffer = []

            for line_num, line in enumerate(infile, 1):
                total_input += 1
                line = line.strip()
                if not line: pbar.update(1); continue
                try: record = json_loads(line)
                except Exception as e: error_list.append((line_num, f"JSON load: {e}. L: '{line[:100]}...'")); pbar.update(1); continue
                
                swap = bool(random.getrandbits(1))

                if process_as_chat:
                    messages, augmentation = create_chat_messages(record, swap, system_prompt_for_chat) 
                    if messages:
                        record_batch_originals.append(dict(record)) 
                        messages_batch_for_template.append(messages)
                        augmentation_batch_info.append(augmentation)
                    if len(messages_batch_for_template) >= args.batch_size:
                        try:
                            formatted_texts = tokenizer.apply_chat_template(messages_batch_for_template, tokenize=False, add_generation_prompt=False)
                            for i, original_rec in enumerate(record_batch_originals):
                                out_record = original_rec
                                out_record["augmentation"] = augmentation_batch_info[i]
                                out_record["text"] = formatted_texts[i]
                                output_lines_buffer.append(json_dumps(out_record)) 
                                total_output += 1
                            outfile.write("\n".join(output_lines_buffer) + "\n")
                        except Exception as e:
                            logging.error(f"Err applying chat template batch (lines ~{line_num-args.batch_size+1}-{line_num}): {e}", exc_info=args.debug)
                            for i_err, _ in enumerate(record_batch_originals): error_list.append((f"Batch item {i_err} in failing batch", f"Failed batch: {e}"))
                        finally: 
                            record_batch_originals.clear()
                            messages_batch_for_template.clear()
                            augmentation_batch_info.clear()
                            output_lines_buffer.clear()
                else:
                    out_record = process_record_standard(record, swap)
                    if out_record: 
                        outfile.write(json_dumps(out_record) + "\n")
                        total_output += 1
                pbar.update(1)
            
            # Final batch processing for chat mode
            if process_as_chat and messages_batch_for_template:
                logging.info(f"Processing final batch of {len(messages_batch_for_template)} items...")
                try:
                    formatted_texts = tokenizer.apply_chat_template(messages_batch_for_template, tokenize=False, add_generation_prompt=False)
                    for i, original_rec in enumerate(record_batch_originals):
                        out_record = original_rec
                        out_record["augmentation"] = augmentation_batch_info[i]
                        out_record["text"] = formatted_texts[i]
                        output_lines_buffer.append(json_dumps(out_record)); total_output += 1
                    if output_lines_buffer: outfile.write("\n".join(output_lines_buffer) + "\n")
                except Exception as e:
                    logging.error(f"Err applying chat template final batch: {e}", exc_info=args.debug)
                    for i_err, _ in enumerate(record_batch_originals): error_list.append((f"Final batch item {i_err}", f"Failed final batch: {e}"))
                finally: 
                    record_batch_originals.clear()
                    messages_batch_for_template.clear()
                    augmentation_batch_info.clear()
                    output_lines_buffer.clear()
    except FileNotFoundError: logging.error(f"Input file '{args.input_file}' not found."); sys.exit(1)
    except IOError as e: logging.error(f"File I/O error: {e}"); sys.exit(1)
    except Exception as e: logging.error(f"Unexpected error: {e}", exc_info=args.debug); sys.exit(1)
    finally:
        if pbar: pbar.close()

    print(f"\n--- Processing Summary ---")
    print(f"Total input lines read: {total_input}")
    print(f"Total output lines generated: {total_output}")
    if error_list:
        print(f"\nEncountered {len(error_list)} errors.")
        max_errors_to_show = 10
        for i, (line_info, err) in enumerate(error_list[:max_errors_to_show]): print(f"  Err {i+1} (Line/Info: {line_info}): {err}")
        if len(error_list) > max_errors_to_show: print(f"  ... and {len(error_list) - max_errors_to_show} more errors.")
    else: 
        print("No errors encountered.")

if __name__ == "__main__":
    main()
