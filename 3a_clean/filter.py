#!/usr/bin/env python3

import argparse
import json
from tqdm import tqdm

def process_line(line):
    try:
        record = json.loads(line)
        # Check for 'nob_Latn' and confidence > 0.99
        if record['language'] != 'nob_Latn':
            return None, f"Language is {record['language']}"
        if record.get('language_confidence', 0) <= 0.99:
            return None, f"Language confidence is {record.get('language_confidence', 0)}"
        # Check if the result text is at least 50 characters long
        result_text = record.get('result', '')
        if len(result_text) < 50:
            return None, "Result text is less than 50 characters long"
        # If the record passes all checks, keep the selected fields
        filtered_record = {
            'id': record['id'],
            'text': result_text
        }
        return filtered_record, None
    except (ValueError, KeyError) as e:
        return None, f"JSON error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Filter and process JSONL data.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_file", required=True, help="Path to output JSONL file.")
    args = parser.parse_args()

    error_count = 0
    error_file = "error_report.txt"
    progress_bar = None

    try:
        # Count total lines in the input file for progress bar
        total_lines = sum(1 for _ in open(args.input_file, "r", encoding="utf-8"))
        progress_bar = tqdm(total=total_lines, desc="Processing lines", unit="line")
        
        with open(args.input_file, "r", encoding="utf-8") as infile, \
             open(args.output_file, "w", encoding="utf-8") as outfile, \
             open(error_file, "w", encoding="utf-8") as errfile:
            
            for line in infile:
                filtered_record, error_message = process_line(line)
                if filtered_record:
                    outfile.write(json.dumps(filtered_record) + '\n')
                else:
                    errfile.write(error_message + '\n')
                    error_count += 1
                progress_bar.update(1)
    
    finally:
        if progress_bar:
            progress_bar.close()
    
    print(f"\nProcessing complete. {error_count} errors recorded in {error_file}.")

if __name__ == "__main__":
    main()
