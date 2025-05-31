import argparse
import json
import random
from tqdm import tqdm
import sys
import os # For input_file basename

from transformers import AutoTokenizer
from huggingface_hub.utils import HfHubHTTPError

try:
    from transformers.utils.hub import OfflineModeIsEnabled
except ImportError:
    try:
        from transformers import OfflineModeIsEnabled
    except ImportError:
        try:
            from transformers.file_utils import OfflineModeIsEnabled
        except ImportError:
            print("Warning: Could not import 'OfflineModeIsEnabled'. Offline functionality checks may be affected.")
            class OfflineModeIsEnabled(Exception): pass # type: ignore

def human_format(num: float) -> str:
    """Formats a number into a human-readable string with k, M, B, T suffixes."""
    num = float(num)
    if abs(num) < 1000:
        # For numbers less than 1000, show as integer without decimal.
        return f"{num:.0f}"
    
    for unit in ['', 'k', 'M', 'B', 'T']: # Suffixes for Kilo, Mega, Giga, Tera
        if abs(num) < 1000.0:
            # Format with one decimal place and the unit
            return f"{num:.1f}{unit}"
        num /= 1000.0
    # If it's larger than Tera, show with 'P' (Peta)
    return f"{num:.1f}P"

def count_tokens_in_text(text_content: str, tokenizer) -> int:
    """Counts tokens in a single text string, excluding special tokens."""
    if not isinstance(text_content, str):
        return 0
    return len(tokenizer.encode(text_content, add_special_tokens=False))

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the number of real tokens in the 'text' field of a JSONL file."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Name of the Hugging Face tokenizer to use (e.g., 'gpt2', 'meta-llama/Llama-3.1-8B')."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N lines to estimate total tokens. If not provided, processes the entire file."
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help="Hugging Face API token, if required for private/gated models."
    )

    args = parser.parse_args()
    input_file_basename = os.path.basename(args.input_file)

    print(f"Loading tokenizer: {args.tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=args.hf_token)
    except (HfHubHTTPError, OfflineModeIsEnabled, ValueError, OSError) as e:
        print(f"Error loading tokenizer '{args.tokenizer_name}': {e.__class__.__name__}: {e}")
        print("\nPlease ensure:")
        print("1. The tokenizer name is correct.")
        print("2. You have an active internet connection (or the model is cached).")
        print("3. For gated models (e.g., Llama): you are logged in (`huggingface-cli login`) "
              "or provided a valid --hf_token with permissions.")
        print("4. The model/tokenizer files exist and are accessible.")
        sys.exit(1)
    print("Tokenizer loaded.")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f_check:
            f_check.readline() 
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening or reading input file {args.input_file}: {e}")
        sys.exit(1)

    if args.sample is not None and args.sample > 0:
        # --- Sampling Mode ---
        print(f"\nSampling mode: estimating tokens for '{input_file_basename}' based on {args.sample} lines.")

        total_lines_in_file = 0
        print("Counting total lines in file (this may take a while for large files)...")
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for _ in tqdm(f, desc="Counting lines", unit=" lines"):
                    total_lines_in_file += 1
        except Exception as e:
            print(f"Error reading file to count lines: {e}")
            sys.exit(1)

        if total_lines_in_file == 0:
            print("Input file is empty. Estimated tokens: 0")
            return

        actual_sample_size = args.sample
        if args.sample >= total_lines_in_file:
            print(f"Sample size ({args.sample}) is >= total lines ({total_lines_in_file:,}). Switching to full file processing.")
            args.sample = None 
        else:
            line_indices_to_sample = sorted(random.sample(range(total_lines_in_file), actual_sample_size))
            
            sampled_texts = []
            malformed_sampled_lines = 0
            empty_text_fields_in_sample = 0
            
            print(f"Reading {actual_sample_size} random lines for sampling...")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                current_line_idx = 0
                sample_ptr = 0
                for line in tqdm(f, total=total_lines_in_file, desc="Scanning for samples", unit=" lines"):
                    if sample_ptr < len(line_indices_to_sample) and current_line_idx == line_indices_to_sample[sample_ptr]:
                        try:
                            data = json.loads(line)
                            text_content = data.get("text")
                            if text_content is None:
                                pass
                            elif isinstance(text_content, str):
                                if text_content:
                                    sampled_texts.append(text_content)
                                else:
                                    empty_text_fields_in_sample +=1
                            else:
                                print(f"Warning: 'text' field in sampled line {current_line_idx + 1} is not a string (type: {type(text_content)}), skipping.")
                        except json.JSONDecodeError:
                            malformed_sampled_lines += 1
                        sample_ptr += 1
                    current_line_idx += 1
                    if sample_ptr >= len(line_indices_to_sample) and len(line_indices_to_sample) > 0:
                        break 
            
            estimated_total_tokens = 0
            avg_tokens_per_valid_sampled_line = 0

            if sampled_texts:
                tokens_in_sample = 0
                print(f"Tokenizing {len(sampled_texts):,} sampled texts...")
                for text in tqdm(sampled_texts, desc="Tokenizing samples", unit=" texts"):
                    tokens_in_sample += count_tokens_in_text(text, tokenizer)
                
                avg_tokens_per_valid_sampled_line = tokens_in_sample / len(sampled_texts)
                
                proportion_of_valid_lines_in_sample = len(sampled_texts) / actual_sample_size
                estimated_total_valid_lines_in_file = total_lines_in_file * proportion_of_valid_lines_in_sample
                estimated_total_tokens = avg_tokens_per_valid_sampled_line * estimated_total_valid_lines_in_file
            
            print("\n--- Sampling Estimate ---")
            print(f"File: {input_file_basename} ({total_lines_in_file:,} lines total)")
            print(f"Sampled: {actual_sample_size:,} lines")
            if malformed_sampled_lines > 0:
                 print(f" - Malformed JSON lines skipped in sample: {malformed_sampled_lines:,}")
            if empty_text_fields_in_sample > 0:
                 print(f" - Empty 'text' fields in sample: {empty_text_fields_in_sample:,}")
            
            if not sampled_texts:
                print(" - No valid, non-empty 'text' fields found in the sampled lines to estimate tokens.")
            else:
                print(f" - Valid 'text' fields processed from sample: {len(sampled_texts):,}")
                print(f" - Avg. tokens/valid line in sample: {avg_tokens_per_valid_sampled_line:.2f}")
            
            print(f"==> Estimated Total Tokens: {estimated_total_tokens:,.0f} (~{human_format(estimated_total_tokens)} tokens) <==")
            return

    # --- Full File Processing Mode ---
    if args.sample is None: # This condition ensures this block runs if --sample was not used or if it was reset
        print(f"\nProcessing entire file '{input_file_basename}' (this may take a while)...")
        total_tokens = 0
        lines_with_text_field = 0
        lines_with_empty_text_field = 0
        malformed_lines = 0
        total_lines_processed = 0

        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="Processing file", unit=" lines")):
                    total_lines_processed += 1
                    try:
                        data = json.loads(line)
                        text_content = data.get("text")
                        if text_content is not None:
                            if isinstance(text_content, str):
                                if text_content:
                                    total_tokens += count_tokens_in_text(text_content, tokenizer)
                                    lines_with_text_field += 1
                                else:
                                    lines_with_empty_text_field +=1
                            else:
                                print(f"Warning: 'text' field in line {i+1} is not a string (type: {type(text_content)}), skipping.")
                    except json.JSONDecodeError:
                        if malformed_lines < 5: # Show first few errors
                            print(f"Warning: Skipping malformed JSON line {i+1}: {line.strip()[:80]}...")
                        elif malformed_lines == 5:
                            print("Warning: Further malformed line warnings will be suppressed.")
                        malformed_lines += 1
        except Exception as e:
            print(f"Error during full file processing: {e}")
            sys.exit(1)
        
        print("\n--- Full File Scan Results ---")
        print(f"File: {input_file_basename} ({total_lines_processed:,} lines processed)")
        if malformed_lines > 0:
            print(f" - Skipped {malformed_lines:,} malformed JSON lines.")
        print(f"Lines with non-empty 'text' field: {lines_with_text_field:,}")
        if lines_with_empty_text_field > 0:
            print(f"Lines with an empty string in 'text' field: {lines_with_empty_text_field:,}")
        
        avg_tokens = 0
        if lines_with_text_field > 0:
            avg_tokens = total_tokens / lines_with_text_field
            print(f"Avg. tokens/valid line: {avg_tokens:.2f}")
        elif total_lines_processed > 0 and malformed_lines < total_lines_processed and lines_with_text_field == 0:
            print(" - No lines with non-empty 'text' field found.")
        
        print(f"==> Total 'Real' Tokens: {total_tokens:,.0f} (~{human_format(total_tokens)} tokens) <==")

if __name__ == "__main__":
    main()
