#!/usr/bin/env python3
"""
clean.py
========
Fix and filter JSON-Lines datasets (fields: id, text).

• ftfy.fix_text on every text field
• drop records whose fixed text is < 50 characters
• guarantee unique ids by appending _N where N is the 1-based line index
• reads **all** .jsonl files from --input_folder and writes cleaned
  files with the same names to --output_folder
• uses a process pool: each file handled by a separate CPU core

Progress
--------
* an outer tqdm bar shows total files processed
* per-file stats (kept / fixed / dropped) printed after each file

Usage
-----
python clean.py --input_folder raw_dir --output_folder clean_dir \
                --workers 32
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from ftfy import fix_text
from tqdm import tqdm


def clean_file(src_path: str, dst_path: str) -> tuple[str, int, int, int]:
    """
    Returns tuple (file_name, kept, fixed, dropped)
    """
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    fixed = dropped = kept = 0

    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for n, line in enumerate(fin, 1):
            try:
                rec = json.loads(line)
                original = rec["text"]
            except Exception:
                dropped += 1
                continue

            new = fix_text(original)
            if len(new) < 50:
                dropped += 1
                continue

            if new != original:
                fixed += 1
            kept += 1

            rec["text"] = new
            rec["id"] = f"{rec['id']}_{n}"
            json.dump(rec, fout, ensure_ascii=False)
            fout.write("\n")

    return src.name, kept, fixed, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="number of parallel processes (default: all CPUs)")
    args = ap.parse_args()

    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    files = sorted(in_dir.glob("*.jsonl"))

    if not files:
        print("No .jsonl files found.")
        return

    print(f"Processing {len(files)} files with {args.workers} workers …")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(clean_file, str(src), str(out_dir / src.name)): src
                   for src in files}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Files", unit="file"):
            name, kept, fixed, dropped = fut.result()
            print(f"{name}: kept {kept}, fixed {fixed}, dropped {dropped}")

    print("Done.")


if __name__ == "__main__":
    main()
