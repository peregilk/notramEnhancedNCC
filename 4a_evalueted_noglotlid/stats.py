#!/usr/bin/env python3
"""
stats.py

Scan a directory (current working directory by default) for all *.jsonl files,
count lines and tokens per file (tokens counted **only** in the `"text"` field
of each JSON line) using a specified tokenizer, and output a neatly-formatted
Markdown table.

✓ Uses nested tqdm progress bars: one for files, one for lines inside each file.  
✓ Defaults to the meta-llama/Llama-3.1-8B-Instruct tokenizer.  
✓ Debug logging switch (--debug).

Usage
-----
    python stats.py [--folder <path>] [--tokenizer <model-name>] [--debug]

Example
-------
    python stats.py --folder ./data --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm
from transformers import AutoTokenizer


class FileStats:
    """Data holder for per-file statistics."""

    __slots__ = ("name", "lines", "tokens")

    def __init__(self, name: str, lines: int, tokens: int) -> None:
        self.name = name
        self.lines = lines
        self.tokens = tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute line/token stats for *.jsonl files (tokens taken from the 'text' field)."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path.cwd(),
        help="Folder to scan (default: current directory).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face tokenizer to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def init_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def safe_json_parse(line: str, file_name: str, line_no: int) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as err:
        logging.warning("JSON error in %s line %d: %s", file_name, line_no, err)
        return None


def iter_jsonl(path: Path) -> Iterable[tuple[str, int]]:
    """Yield (text, line_number) tuples from JSON-Lines file; skip malformed rows."""
    with path.open("r", encoding="utf-8") as fp:
        for i, raw in enumerate(fp, start=1):
            obj = safe_json_parse(raw, path.name, i)
            if obj and isinstance(obj.get("text"), str):
                yield obj["text"], i
            else:
                logging.debug("Missing 'text' in %s line %d – skipped", path.name, i)


def count_file_stats(path: Path, tokenizer) -> FileStats:
    """Count lines and tokens (in the 'text' field) for one file."""
    total_lines = 0
    total_tokens = 0

    for text, _ in tqdm(iter_jsonl(path), desc=path.name, leave=False):
        total_lines += 1
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))

    logging.debug("Processed %s: %s lines, %s tokens", path.name, total_lines, total_tokens)
    return FileStats(path.name, total_lines, total_tokens)


def gather_stats(folder: Path, tokenizer) -> List[FileStats]:
    files = sorted(folder.glob("*.jsonl"))
    stats: List[FileStats] = []

    for file_path in tqdm(files, desc="Files", unit="file"):
        stats.append(count_file_stats(file_path, tokenizer))

    return stats


def fmt(n: int) -> str:
    """Thousands-separated integer as string."""
    return f"{n:,}"


def print_markdown_table(stats: List[FileStats]) -> None:
    rows = [
        "| File | Lines | Tokens |",
        "| --- | ---: | ---: |",
    ]

    tot_lines = tot_tokens = 0
    for fs in stats:
        rows.append(f"| {fs.name} | {fmt(fs.lines)} | {fmt(fs.tokens)} |")
        tot_lines += fs.lines
        tot_tokens += fs.tokens

    rows.append(f"| **Total** | **{fmt(tot_lines)}** | **{fmt(tot_tokens)}** |")

    print("\n".join(rows))


def main() -> None:
    args = parse_args()
    init_logging(args.debug)

    folder = args.folder.resolve()
    if not folder.is_dir():
        logging.error("Folder does not exist or is not a directory: %s", folder)
        raise SystemExit(1)

    logging.info("Scanning folder: %s", folder)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    statistics = gather_stats(folder, tokenizer)

    if not statistics:
        logging.warning("No *.jsonl files found in %s", folder)

    print_markdown_table(statistics)


if __name__ == "__main__":
    main()
