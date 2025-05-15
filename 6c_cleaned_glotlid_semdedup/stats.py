#!/usr/bin/env python3
"""
Efficient, parallelized .jsonl stats script.
- Scans for *.jsonl files in a folder (default: CWD).
- Counts lines and tokens (from 'text' field) per file using HF tokenizer.
- Runs file processing in parallel using multiprocessing.
- Outputs Markdown table (default: stats.md, override with --output).
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import concurrent.futures
import os

from tqdm import tqdm

# Optional: use orjson if available
try:
    import orjson as json
except ImportError:
    import json

_TOKENIZER = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute line/token stats for *.jsonl files (from 'text' field, parallelized).")
    parser.add_argument("--folder", type=Path, default=Path("."), help="Folder to scan (default: current directory).")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF tokenizer name or path.")
    parser.add_argument("--output", type=Path, default=Path("stats.md"), help="Markdown file to write results (default: stats.md).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes (default: all cores).")
    return parser.parse_args()

def init_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

def load_tokenizer_once(tokenizer_name: str):
    """Load tokenizer only once per process (multiprocessing safe)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return _TOKENIZER

def safe_json_parse(line: str | bytes, file_name: str, line_no: int) -> dict | None:
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        return json.loads(line)
    except Exception as err:
        logging.debug("JSON error in %s line %d: %s", file_name, line_no, err)
        return None

def count_file_stats_worker(args: Tuple[str, str]) -> Tuple[str, int, int]:
    """Worker: counts lines and tokens in a file."""
    file_path, tokenizer_name = args
    tokenizer = load_tokenizer_once(tokenizer_name)
    total_lines = 0
    total_tokens = 0
    with open(file_path, "r", encoding="utf-8") as fp:
        for i, raw in enumerate(fp, start=1):
            obj = safe_json_parse(raw, file_path, i)
            if obj and isinstance(obj.get("text"), str):
                total_lines += 1
                total_tokens += len(tokenizer.encode(obj["text"], add_special_tokens=False))
    return (Path(file_path).name, total_lines, total_tokens)

def fmt(n: int) -> str:
    return f"{n:,}"

def make_markdown_table(stats: List[Tuple[str, int, int]]) -> str:
    rows = [
        "| File | Lines | Tokens |",
        "| --- | ---: | ---: |",
    ]
    tot_lines = tot_tokens = 0
    for name, lines, tokens in stats:
        rows.append(f"| {name} | {fmt(lines)} | {fmt(tokens)} |")
        tot_lines += lines
        tot_tokens += tokens
    rows.append(f"| **Total** | **{fmt(tot_lines)}** | **{fmt(tot_tokens)}** |")
    return "\n".join(rows)

def main() -> None:
    args = parse_args()
    init_logging(args.debug)
    folder = args.folder.resolve()
    if not folder.is_dir():
        logging.error("Folder does not exist or is not a directory: %s", folder)
        raise SystemExit(1)
    logging.info("Scanning folder: %s", folder)
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        logging.warning("No *.jsonl files found in %s", folder)
        return
    # Pre-load tokenizer in main process (for HF cache warmup)
    load_tokenizer_once(args.tokenizer)
    jobs = [(str(fp), args.tokenizer) for fp in files]
    stats: List[Tuple[str, int, int]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for result in tqdm(executor.map(count_file_stats_worker, jobs), total=len(jobs), desc="Files", unit="file"):
            stats.append(result)
    md_table = make_markdown_table(stats)
    # Write output
    args.output.write_text(md_table + "\n", encoding="utf-8")
    print(f"Stats written to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
