#!/usr/bin/env python3
import argparse
import json
import re
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate a JSONLines corpus using MinHash and datasketch.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONLines file")
    parser.add_argument("--output_file", required=True, help="Path to output deduplicated JSONLines file")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard similarity threshold (default 0.85)")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of MinHash permutations (default 256)")
    parser.add_argument("--show_examples", type=int, default=0, help="Show up to N duplicate examples (kept vs. removed)")
    return parser.parse_args()


def extract_user_assistant(text):
    pattern = re.compile(r"<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>\s*<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", re.DOTALL)
    matches = pattern.findall(text)
    return "\n\n".join([q.strip() + "\n" + a.strip() for q, a in matches]) if matches else text


def get_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf-8'))
    return m


def main():
    args = parse_args()

    print(f"Loading input file: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f]

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    keep_flags = [True] * len(documents)
    duplicate_examples = []

    print("Indexing and deduplicating...")
    for i, doc in tqdm(enumerate(documents), total=len(documents)):
        content = extract_user_assistant(doc['text'])
        m = get_minhash(content, args.num_perm)
        duplicates = lsh.query(m)
        if duplicates:
            keep_flags[i] = False
            if len(duplicate_examples) < args.show_examples:
                ref_idx = int(duplicates[0].split('_')[1])
                ref_content = extract_user_assistant(documents[ref_idx]['text'])
                duplicate_examples.append((ref_content, content))
        else:
            lsh.insert(f"doc_{i}", m)

    print("Writing deduplicated output...")
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        kept = 0
        for i, doc in enumerate(documents):
            if keep_flags[i]:
                out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                kept += 1

    print(f"Done. Kept {kept} out of {len(documents)} documents.")

    if args.show_examples > 0:
        print(f"\nShowing up to {args.show_examples} duplicate examples (kept vs. removed):\n")
        for kept_text, removed_text in duplicate_examples:
            print("\n" + "-" * 80)
            print("[KEPT]:")
            print(kept_text)
            print("\n[REMOVED]:")
            print(removed_text)


if __name__ == "__main__":
    main()

