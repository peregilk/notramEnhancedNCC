import argparse
import json
from tqdm import tqdm

def sanitize_json_string(json_string):
    # Remove leading/trailing newlines or spaces
    json_string = json_string.strip()
    return json_string

def parse_askLLMresult(askLLMresult):
    askLLMresult = askLLMresult.strip()
    if not askLLMresult.startswith("[") or not askLLMresult.endswith("]"):
        raise ValueError("askLLMresult does not have the expected brackets format")
    # Remove leading '[' and trailing ']\n'
    askLLMresult = askLLMresult[1:-1].strip()
    try:
        return json.loads(f"[{askLLMresult}]")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e} for data: {askLLMresult}")

def recursive_parse_json(json_data):
    if isinstance(json_data, str):
        if json_data.strip() == "":
            return json_data  # Return empty strings as is
        json_data = sanitize_json_string(json_data)
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decode error: {e} for data: {json_data}")
    if isinstance(json_data, list):
        return [recursive_parse_json(item) for item in json_data]
    if isinstance(json_data, dict):
        return {key: recursive_parse_json(value) for key, value in json_data.items()}
    return json_data

def process_dataset(input_file, output_file, verbose, minimum_translation_score):
    print("Loading dataset...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Skipping line due to JSON decode error: {e}")
                    print(f"Offending line: {line}")
                continue

    print("Loaded dataset. Total records:", len(data))
    
    skipped_records = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in tqdm(data, desc="Processing dataset"):
            try:
                askLLMresult = record.get('askLLMresult', '')

                if not askLLMresult:
                    if verbose:
                        print(f"Empty askLLMresult in record: {record['uuid']}")
                    skipped_records += 1
                    continue

                try:
                    askLLMresult = parse_askLLMresult(askLLMresult)
                except ValueError as e:
                    if verbose:
                        print(f"First JSON decode error in askLLMresult: {e} in record: {record['uuid']}")
                    skipped_records += 1
                    continue

                combined_text_parts = []
                translation_score = None

                for item in askLLMresult:
                    if isinstance(item, dict):
                        value = item.get('value')
                        if value is not None:
                            combined_text_parts.append(value)
                        if 'translation_score' in item:
                            try:
                                score = float(item['translation_score'])
                                if translation_score is None or score < translation_score:
                                    translation_score = score
                            except ValueError:
                                continue
                    else:
                        if verbose:
                            print(f"Skipping non-dict item in askLLMresult: {item} in record: {record['uuid']}")
                        skipped_records += 1
                        continue

                if translation_score is not None and translation_score < minimum_translation_score:
                    if verbose:
                        print(f"Skipping record due to low translation score: {translation_score} in record: {record['uuid']}")
                    skipped_records += 1
                    continue

                if combined_text_parts:
                    combined_text = "\n\n".join(combined_text_parts).strip()

                    output_record = {
                        "source": input_file,
                        "uuid": record['uuid'],
                        "text": combined_text
                    }
                    f.write(json.dumps(output_record) + '\n')
                else:
                    if verbose:
                        print(f"Skipping record because no valid items were found in askLLMresult: {record['uuid']}")
                    skipped_records += 1
            except KeyError as e:
                if verbose:
                    print(f"Skipping record due to missing column: {e} in record: {record['uuid']}")
                skipped_records += 1
                continue
            except ValueError as e:
                if verbose:
                    print(f"Skipping record due to value error: {e} in record: {record['uuid']}")
                skipped_records += 1
                continue
            except Exception as e:
                if verbose:
                    print(f"Skipping record due to unexpected error: {e} in record: {record['uuid']}")
                skipped_records += 1
                continue

    print(f"Processing completed. Total skipped records: {skipped_records}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset and convert to JSONLines format.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONLines file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONLines file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--minimum_translation_score', type=float, default=3, help='Minimum translation score to include a record')

    args = parser.parse_args()
    process_dataset(args.input_file, args.output_file, args.verbose, args.minimum_translation_score)
