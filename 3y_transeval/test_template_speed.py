#!/usr/bin/env python3

import argparse
import json # Standard JSON library
import logging
import sys
import random
from typing import Dict, Any, List

# Attempt to use orjson for potentially faster JSON processing
try:
    import orjson
    JSON_LIB_USED = "orjson"
    def json_loads(s: str) -> Any:
        return orjson.loads(s)
    def json_dumps(obj: Any) -> str:
        # orjson.dumps returns bytes, so we need to decode
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json as std_json # Fallback to standard json
    JSON_LIB_USED = "standard json"
    def json_loads(s: str) -> Any:
        return std_json.loads(s)
    def json_dumps(obj: Any) -> str:
        return std_json.dumps(obj, ensure_ascii=False)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

NB_LABELS = [
    "Norwegian:", "Norwegian Bokmål:", "Norsk:", "Norsk bokmål:"
]
EN_LABELS = [
    "English:", "Engelsk:"
]

# Prompts for Norwegian Bokmål -> English and English -> Norwegian Bokmål
NB_PROMPTS_TO_EN = [
    "Oversett denne teksten fra norsk bokmål til engelsk:", "Vennligst oversett følgende tekst fra norsk bokmål til engelsk:", "Kan du oversette teksten nedenfor (som er på norsk bokmål) til engelsk?", "Oversett fra norsk bokmål til engelsk:", "Gjengi denne norsk bokmål-teksten på engelsk.", "Konverter dette tekstutdraget fra norsk bokmål til engelsk.", "Oversettelse fra norsk bokmål til engelsk ønskes for:", "Jeg trenger denne teksten oversatt fra norsk bokmål til engelsk:", "Oversett følgende til engelsk (teksten er på norsk bokmål):", "Norsk bokmål-tekst for oversettelse til engelsk:", "Oversett denne teksten (skrevet på norsk bokmål) til engelsk:", "Vennligst gjør om følgende tekst, som er på norsk bokmål, til engelsk:", "Kan du gjengi teksten under, som er på norsk bokmål, på engelsk?", "Til engelsk, oversett denne norsk bokmål-teksten:", "Kildespråk: Norsk bokmål. Målspråk: Engelsk. Oversett teksten:", "Formuler denne norsk bokmål-teksten på engelsk.", "Jeg ønsker en engelsk versjon av denne teksten på norsk bokmål:", "Overfør meningen i denne teksten fra norsk bokmål til engelsk.", "Gi meg en engelsk oversettelse av det følgende (på norsk bokmål):", "Lag en engelsk gjengivelse av teksten nedenfor (som er norsk bokmål).", "Norsk bokmål -> Engelsk: Oversett.", "Hjelp meg å oversette dette fra norsk bokmål til engelsk.", "Vær snill og konverter denne norsk bokmål-teksten til engelsk:", "Tekst på norsk bokmål for engelsk oversettelse:", "Sett om denne passasjen fra norsk bokmål til engelsk.",
    "Oversett denne teksten fra norsk til engelsk:", "Vennligst oversett følgende tekst fra norsk til engelsk:", "Kan du oversette teksten nedenfor fra norsk til engelsk?", "Oversett fra norsk til engelsk:", "Gjengi denne norske teksten på engelsk.", "Konverter dette tekstutdraget fra norsk til engelsk.", "Oversettelse fra norsk til engelsk ønskes for:", "Jeg trenger denne teksten oversatt fra norsk til engelsk:", "Oversett følgende til engelsk (teksten er på norsk):", "Norsk tekst for oversettelse til engelsk:", "Oversett denne teksten til engelsk:", "Vennligst gjør om følgende tekst til engelsk:", "Kan du gjengi teksten under på engelsk?", "Til engelsk, oversett:", "Kildespråk: Norsk. Målspråk: Engelsk. Oversett teksten:", "Formuler denne norske teksten på engelsk.", "Jeg ønsker en engelsk versjon av denne norske teksten:", "Overfør meningen i denne teksten fra norsk til engelsk.", "Gi meg en engelsk oversettelse av det følgende (som er norsk):", "Lag en engelsk gjengivelse av teksten nedenfor.", "Norsk -> Engelsk: Oversett.", "Hjelp meg å oversette dette fra norsk til engelsk.", "Vær snill og konverter norsk tekst til engelsk:", "Tekst for engelsk oversettelse (fra norsk):", "Sett om denne passasjen fra norsk til engelsk.",
    "Translate this text from Norwegian Bokmål to English:", "Please translate the following passage from Norwegian Bokmål into English.", "Could you convert this Norwegian Bokmål text to English?", "Render the text below from Norwegian Bokmål to English.", "Norwegian Bokmål to English translation task:", "Translate from Norwegian Bokmål to English:", "I need an English translation of the following Norwegian Bokmål text:", "Provide an English version of the text (originally in Norwegian Bokmål):", "Source: Norwegian Bokmål. Target: English. Translate the provided text:", "Translate this Norwegian Bokmål text to English:", "Please convert the following text, which is in Norwegian Bokmål, into English.", "Could you render the passage below (in Norwegian Bokmål) in English?", "To English, please translate this Norwegian Bokmål text:", "Here is a text in Norwegian Bokmål. Please provide the English translation:", "Task: Translate the following from its original language (Norwegian Bokmål) to English.", "Rephrase this Norwegian Bokmål text in English.", "Convert the following from Norwegian Bokmål into English:", "Your task is to translate the upcoming Norwegian Bokmål text into English.", "English rendition needed for this Norwegian Bokmål text:", "Process the following Norwegian Bokmål text and output its English equivalent.", "Norwegian Bokmål -> English: Translate.", "Assist me by translating this from Norwegian Bokmål to English.", "Kindly transform the Norwegian Bokmål input into English.", "Provide the English translation for the Norwegian Bokmål content below:", "Transcribe this passage from Norwegian Bokmål into English.",
    "Translate this text from Norwegian to English:", "Please translate the following passage from Norwegian into English.", "Could you convert this Norwegian text to English?", "Render the text below from Norwegian to English.", "Norwegian to English translation task:", "Translate from Norwegian to English:", "I need an English translation of the following Norwegian text:", "Provide an English version of the text (originally in Norwegian):", "Source: Norwegian. Target: English. Translate the provided text:", "Translate this text to English:", "Please convert the following text into English.", "Could you render the passage below in English?", "To English, please translate the text:", "Here is a text in Norwegian. Please provide the English translation:", "Task: Translate the following from its original language (Norwegian) to English.", "Rephrase this Norwegian text in English.", "Convert the following from Norwegian into English:", "Your task is to translate the upcoming Norwegian text into English.", "English rendition needed for this Norwegian text:", "Process the following Norwegian text and output its English equivalent.", "Norwegian -> English: Translate.", "Assist me by translating this from Norwegian to English.", "Kindly transform the Norwegian input into English.", "Provide the English translation for the Norwegian content below:", "Transcribe this passage from Norwegian into English."
]

EN_PROMPTS_TO_NB = [
    "Translate this text from English to Norwegian Bokmål:", "Please translate the following passage from English into Norwegian Bokmål.", "Could you convert this English text to Norwegian Bokmål?", "Render the text below from English to Norwegian Bokmål.", "English to Norwegian Bokmål translation task:", "Translate from English to Norwegian Bokmål:", "I need a Norwegian Bokmål translation of the following English text:", "Provide a Norwegian Bokmål version of the text (originally in English):", "Source: English. Target: Norwegian Bokmål. Translate the provided text:", "Translate this English text to Norwegian Bokmål:", "Please convert the following text into Norwegian Bokmål.", "Could you render the passage below in Norwegian Bokmål?", "To Norwegian Bokmål, please translate the text:", "Here is a text in English. Please provide the Norwegian Bokmål translation:", "Task: Translate the following from its original language (English) to Norwegian Bokmål.", "Rephrase this English text in Norwegian Bokmål.", "Convert the following from English into Norwegian Bokmål:", "Your task is to translate the upcoming English text into Norwegian Bokmål.", "Norwegian Bokmål rendition needed for this English text:", "Process the following English text and output its Norwegian Bokmål equivalent.", "English -> Norwegian Bokmål: Translate.", "Assist me by translating this from English to Norwegian Bokmål.", "Kindly transform the English input into Norwegian Bokmål.", "Provide the Norwegian Bokmål translation for the English content below:", "Transcribe this passage from English into Norwegian Bokmål.",
    "Oversett denne teksten fra engelsk til norsk bokmål:", "Vennligst oversett følgende tekst fra engelsk til norsk bokmål:", "Kan du oversette teksten nedenfor fra engelsk til norsk bokmål?", "Oversett fra engelsk til norsk bokmål:", "Gjengi denne engelske teksten på norsk bokmål.", "Konverter dette tekstutdraget fra engelsk til norsk bokmål.", "Oversettelse fra engelsk til norsk bokmål ønskes for:", "Jeg trenger denne teksten oversatt fra engelsk til norsk bokmål:", "Oversett følgende til norsk bokmål (teksten er på engelsk):", "Engelsk tekst for oversettelse til norsk bokmål:", "Oversett denne teksten (fra engelsk) til norsk bokmål:", "Vennligst gjør om følgende engelske tekst til norsk bokmål:", "Kan du gjengi teksten under (som er på engelsk) på norsk bokmål?", "Til norsk bokmål, oversett denne engelske teksten:", "Kildespråk: Engelsk. Målspråk: Norsk bokmål. Oversett teksten:", "Formuler denne engelske teksten på norsk bokmål.", "Jeg ønsker en norsk bokmål-versjon av denne engelske teksten:", "Overfør meningen i denne teksten fra engelsk til norsk bokmål.", "Gi meg en norsk bokmål-oversettelse av det følgende (som er engelsk):", "Lag en norsk bokmål-gjengivelse av den engelske teksten nedenfor.", "Engelsk -> Norsk bokmål: Oversett.", "Hjelp meg å oversette dette fra engelsk til norsk bokmål.", "Vær snill og konverter engelsk tekst til norsk bokmål:", "Tekst for norsk bokmål-oversettelse (fra engelsk):", "Sett om denne passasjen fra engelsk til norsk bokmål.",
    "Oversett denne teksten fra engelsk til norsk:", "Vennligst oversett følgende tekst fra engelsk til norsk:", "Kan du oversette teksten nedenfor fra engelsk til norsk?", "Oversett fra engelsk til norsk:", "Gjengi denne engelske teksten på norsk.", "Konverter dette tekstutdraget fra engelsk til norsk.", "Oversettelse fra engelsk til norsk ønskes for:", "Jeg trenger denne teksten oversatt fra engelsk til norsk:", "Oversett følgende til norsk (teksten er på engelsk):", "Engelsk tekst for oversettelse til norsk:", "Oversett denne teksten til norsk:", "Vennligst gjør om følgende tekst til norsk:", "Kan du gjengi teksten under på norsk?", "Til norsk, oversett:", "Kildespråk: Engelsk. Målspråk: Norsk. Oversett teksten:", "Formuler denne engelske teksten på norsk.", "Jeg ønsker en norsk versjon av denne engelske teksten:", "Overfør meningen i denne teksten fra engelsk til norsk.", "Gi meg en norsk oversettelse av det følgende (som er engelsk):", "Lag en norsk gjengivelse av teksten nedenfor.", "Engelsk -> Norsk: Oversett.", "Hjelp meg å oversette dette fra engelsk til norsk.", "Vær snill og konverter engelsk tekst til norsk:", "Tekst for norsk oversettelse (fra engelsk):", "Sett om denne passasjen fra engelsk til norsk."
]

DEFAULT_BATCH_SIZE = 5000

def count_lines(filename: str) -> int:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def create_chat_messages(
    record: Dict[str, Any],
    swap: bool,
    system_prompt_str: str # Explicitly pass the string
) -> (List[Dict[str, str]], str):
    source = record.get("source", "")
    target = record.get("target", "")
    if not source or not target:
        return None, None

    augmentation = None
    if not swap:
        augmentation = random.choice(EN_PROMPTS_TO_NB)
        user_content = f"{augmentation}\n{source}"
        assistant_content = target
    else:
        augmentation = random.choice(NB_PROMPTS_TO_EN)
        user_content = f"{augmentation}\n{target}"
        assistant_content = source

    # Use the passed system_prompt_str directly
    messages = [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return messages, augmentation

def process_record_standard(
    record: Dict[str, Any],
    swap: bool,
) -> Dict[str, Any]:
    source = record.get("source", "")
    target = record.get("target", "")
    if not source or not target:
        return None

    out = dict(record)
    en_label = random.choice(EN_LABELS)
    nb_label = random.choice(NB_LABELS)
    if not swap:
        text_out = f"{en_label} {source}\n{nb_label} {target}"
    else:
        text_out = f"{nb_label} {target}\n{en_label} {source}"
    out["text"] = text_out
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Format and augment a parallel English-Norwegian(Bokmål) translation dataset."
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
    system_prompt_for_chat = "You are a helpful assistant." # Default system prompt string

    if args.chat_template:
        if AutoTokenizer is None:
            logging.error("The 'transformers' library must be installed for --chat_template. Run: pip install transformers")
            sys.exit(1)
        try:
            logging.info(f"Loading tokenizer for: {args.chat_template}")
            tokenizer = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
            logging.info(f"Tokenizer {args.chat_template} loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading tokenizer '{args.chat_template}': {e}")
            sys.exit(1)

        if not hasattr(tokenizer, "apply_chat_template") or not callable(tokenizer.apply_chat_template):
            logging.error(f"Tokenizer for '{args.chat_template}' lacks a callable 'apply_chat_template' method.")
            sys.exit(1)
        
        # *** MODIFICATION HERE ***
        # Determine system prompt string without getattr on tokenizer for this specific test
        if "Llama-3" in args.chat_template: # Simple check for Llama 3 models
            system_prompt_for_chat = "You are a helpful AI assistant." # Standard for Llama3 instruct
            # Or, if you know the specific Llama 3.1 system prompt:
            # system_prompt_for_chat = "This is a system prompt for Llama 3.1..." 
        # else: system_prompt_for_chat remains "You are a helpful assistant."
        
        logging.info(f"Using chat template from '{args.chat_template}' with system prompt: \"{system_prompt_for_chat}\" and batch size: {args.batch_size}")
    else:
        logging.info("Running in standard (non-chat) mode.")

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
                        if not f_check.read(1):
                             logging.warning(f"Input file '{args.input_file}' is empty.")
                 except FileNotFoundError:
                    logging.error(f"Input file '{args.input_file}' not found.")
                    sys.exit(1)
            logging.info(f"Found {num_lines} lines in '{args.input_file}'.")
        else:
            logging.info("Skipping initial line count.")

        with open(args.input_file, "r", encoding="utf-8") as infile, \
             open(args.output_file, "w", encoding="utf-8") as outfile:

            pbar_params = {"desc": "Processing lines", "unit": "lines"}
            if num_lines > 0:
                pbar_params["total"] = num_lines
            
            pbar = tqdm(**pbar_params)

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
                    record = json_loads(line)
                except Exception as e: 
                    error_list.append((line_num, f"JSON load error: {e}. Line: '{line[:100]}...'"))
                    pbar.update(1)
                    continue
                
                swap = bool(random.getrandbits(1))

                if args.chat_template and tokenizer:
                    # Pass the determined system_prompt_for_chat string
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
                                output_lines_buffer.append(json_dumps(out_record)) 
                                total_output += 1
                            outfile.write("\n".join(output_lines_buffer) + "\n")
                        except Exception as e:
                            logging.error(f"Error applying chat template to batch (lines approx {line_num-args.batch_size+1}-{line_num}): {e}")
                            for i, _ in enumerate(record_batch_originals):
                                error_list.append((f"Batch item {i}", f"Failed in batch processing: {e}"))
                        finally:
                            record_batch_originals.clear()
                            messages_batch_for_template.clear()
                            augmentation_batch_info.clear()
                            output_lines_buffer.clear()
                else: 
                    out_record = process_record_standard(record, swap)
                    if out_record is not None:
                        outfile.write(json_dumps(out_record) + "\n")
                        total_output += 1
                pbar.update(1)

            if args.chat_template and tokenizer and messages_batch_for_template:
                logging.info(f"Processing final batch of {len(messages_batch_for_template)} items...")
                try:
                    formatted_texts = tokenizer.apply_chat_template(
                        messages_batch_for_template, tokenize=False, add_generation_prompt=False
                    )
                    for i, original_rec in enumerate(record_batch_originals):
                        out_record = original_rec
                        out_record["augmentation"] = augmentation_batch_info[i]
                        out_record["text"] = formatted_texts[i]
                        output_lines_buffer.append(json_dumps(out_record))
                        total_output += 1
                    if output_lines_buffer: 
                         outfile.write("\n".join(output_lines_buffer) + "\n")
                except Exception as e:
                    logging.error(f"Error applying chat template to the final batch: {e}")
                    for i, _ in enumerate(record_batch_originals):
                        error_list.append((f"Final batch item {i}", f"Failed in final batch processing: {e}"))
                finally:
                    record_batch_originals.clear()
                    messages_batch_for_template.clear()
                    augmentation_batch_info.clear()
                    output_lines_buffer.clear()

    except FileNotFoundError:
        logging.error(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    except IOError as e:
        logging.error(f"File I/O error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=args.debug)
        sys.exit(1)
    finally:
        if pbar is not None:
             pbar.close()

    print(f"\n--- Processing Summary ---")
    print(f"Total input lines read: {total_input}")
    print(f"Total output lines generated: {total_output}")
    if error_list:
        print(f"\nEncountered {len(error_list)} errors during processing.")
        max_errors_to_show = 10
        for i, (line_info, err) in enumerate(error_list[:max_errors_to_show]):
            print(f"  Error {i+1} (Line/Info: {line_info}): {err}")
        if len(error_list) > max_errors_to_show:
            print(f"  ... and {len(error_list) - max_errors_to_show} more errors.")
    else:
        print("No errors encountered.")

if __name__ == "__main__":
    main()
