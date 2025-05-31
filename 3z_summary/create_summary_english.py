#!/usr/bin/env python3

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

# EuroEval-style base prompt templates (including minimal version)
PROMPT_BASES = [
    # Minimal
    "Document: {article}\nSummary: {highlights}",
    # EuroEval base
    "The following are documents with accompanying summaries.\nDocument: {article}\nSummary: {highlights}",
    # Instruct-style: request summary
    "Document: {article}\n\nWrite a summary of the above document.\nSummary: {highlights}",
    # Placeholder augmentations (expand later)
    "Article: {article}\nSummarize this document.\nSummary: {highlights}",
]

INSTRUCTION_BASES = [
    # The default instruction prompt
    "Document: {article}\n\nWrite a summary of the above document.",
    # Placeholder (add more instructive variations here)
    "Article: {article}\n\nPlease summarize the text.",
    "Input: {article}\n\nProduce a concise summary.",
]

DEFAULT_BATCH_SIZE = 5000

def count_lines(filename: str) -> int:
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def create_chat_messages(record: Dict[str, Any], system_prompt: str) -> (List[Dict[str, str]], str):
    article, highlights = record.get("article", ""), record.get("highlights", "")
    if not article or not highlights:
        return None, None
    # For this example, randomly choose an instruction
    instruction = random.choice(INSTRUCTION_BASES).format(article=article)
    # Assistant always outputs the highlights
    return [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": highlights}
    ], instruction

def process_record_standard(record: Dict[str, Any]) -> Dict[str, Any]:
    article, highlights = record.get("article", ""), record.get("highlights", "")
    if not article or not highlights:
        return None
    out = dict(record)
    template = random.choice(PROMPT_BASES)
    out["text"] = template.format(article=article, highlights=highlights)
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Format and augment summarization dataset for EuroEval-style evaluation."
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
    system_prompt_for_chat = "You are a helpful assistant."
    process_as_chat = False

    if args.chat_template:
        if AutoTokenizer is None:
            logging.error("transformers not installed, cannot use --chat_template")
            sys.exit(1)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
            if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
                process_as_chat = True
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
        instruction_batch_info = []
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

            if process_as_chat:
                messages, instruction = create_chat_messages(record, system_prompt_for_chat)
                if messages:
                    record_batch_originals.append(dict(record))
                    messages_batch_for_template.append(messages)
                    instruction_batch_info.append(instruction)
                if len(messages_batch_for_template) >= args.batch_size:
                    try:
                        formatted_texts = tokenizer.apply_chat_template(
                            messages_batch_for_template, tokenize=False, add_generation_prompt=False
                        )
                        for i, original_rec in enumerate(record_batch_originals):
                            out_record = original_rec
                            out_record["augmentation"] = instruction_batch_info[i]
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
                        instruction_batch_info.clear()
                        output_lines_buffer.clear()
            else:
                out_record = process_record_standard(record)
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
                    out_record["augmentation"] = instruction_batch_info[i]
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
                instruction_batch_info.clear()
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
