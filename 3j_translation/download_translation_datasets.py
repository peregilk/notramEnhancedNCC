#!/usr/bin/env python3
"""
download_and_filter_parallel.py

Download and filter parallel corpora for specified language pairs from Hugging Face datasets.
Applies semantic similarity filtering and saves the results in JSONL format.

Each entry in the output file has the following structure:
{
  "id": "source_0",
  "source": "Source language sentence.",
  "target": "Target language sentence.",
  "distance": 0.1234
}

Usage:
    python download_and_filter_parallel.py \
        --mode enno \
        --output_file train_en_nb.jsonl \
        --sim_threshold 0.9 \
        --batch_size 64 \
        --device cuda \
        --max_examples 10000 \
        --log_level INFO
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, util

# Language code mappings
LANGUAGE_CODES = {
    'en': {'en', 'eng', 'english'},
    'nb': {'nb', 'nob', 'no', 'norwegian'},
    'nn': {'nn', 'nno'}
}

# Dataset configurations for different modes
DATASET_CONFIGS = {
    'enno': [
        ("Helsinki-NLP/opus-100", "en-nb"),
        ("SEACrowd/kde4", "en-nb"),
        ("yhavinga/ccmatrix", "en-no"),
        ("Helsinki-NLP/opus_paracrawl", None),
        ("sentence-transformers/parallel-sentences-jw300", "en-nb"),
    ],
    'ennn': [
        ("Helsinki-NLP/opus-100", "en-nn"),
        # Add more datasets supporting en-nn
    ],
    'nonn': [
        ("Helsinki-NLP/opus-100", "nb-nn"),
        # Add more datasets supporting nb-nn
    ]
}

def extract_pair(example: Dict, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    """
    Extract source and target language sentence pair from the example.
    Returns (source, target) or (None, None) if not valid.
    """
    translation = example.get("translation")
    if not translation:
        return None, None

    src = next((translation[k] for k in LANGUAGE_CODES[src_lang] if k in translation and translation[k]), None)
    tgt = next((translation[k] for k in LANGUAGE_CODES[tgt_lang] if k in translation and translation[k]), None)

    if src and tgt:
        return src.strip(), tgt.strip()
    return None, None

def batched(iterable: Iterable, size: int) -> Iterable[List]:
    """
    Yield successive batches of specified size from the iterable.
    """
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            break
        yield batch

def process_batch(
    src_texts: List[str],
    tgt_texts: List[str],
    model: SentenceTransformer,
    threshold: float,
    out_f,
    kept: int,
    next_id: int,
    max_examples: int
) -> Tuple[int, int]:
    """
    Process a batch of sentence pairs: compute embeddings, filter by similarity,
    and write to output file.
    Returns updated counts of kept examples and next ID.
    """
    embeds_src = model.encode(src_texts, convert_to_tensor=True, show_progress_bar=False)
    embeds_tgt = model.encode(tgt_texts, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(embeds_src, embeds_tgt).diagonal().cpu().numpy()

    for src, tgt, sim in zip(src_texts, tgt_texts, sims):
        if sim < threshold:
            continue
        out_f.write(json.dumps({
            "id": f"source_{next_id}",
            "source": src,
            "target": tgt,
            "distance": round(float(1.0 - sim), 4)
        }, ensure_ascii=False) + "\n")
        next_id += 1
        kept += 1
        if max_examples and kept >= max_examples:
            break
    return kept, next_id

def main():
    parser = argparse.ArgumentParser(
        description="Download parallel corpora, filter by semantic similarity, and save as JSONL."
    )
    parser.add_argument("--mode", required=True, choices=["enno", "ennn", "nonn"],
                        help="Language pair mode: enno, ennn, or nonn.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--sim_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold (default: 0.85).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding.")
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda" (default: cpu).')
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Optional cap on total kept examples.")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine source and target languages based on mode
    if args.mode == 'enno':
        src_lang, tgt_lang = 'en', 'nb'
    elif args.mode == 'ennn':
        src_lang, tgt_lang = 'en', 'nn'
    elif args.mode == 'nonn':
        src_lang, tgt_lang = 'nb', 'nn'
    else:
        logging.error(f"Unsupported mode: {args.mode}")
        sys.exit(1)

    datasets_to_process = DATASET_CONFIGS.get(args.mode, [])
    if not datasets_to_process:
        logging.error(f"No datasets configured for mode: {args.mode}")
        sys.exit(1)

    logging.info("Loading sentence transformer model...")
    model = SentenceTransformer(
        "BAAI/bge-m3",
        device=args.device
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_file).open("w", encoding="utf-8") as out_f:
        next_id = 0
        kept = 0

        for hf_name, config in datasets_to_process:
            logging.info(f"Processing dataset: {hf_name} [{config}]")
            try:
                if config:
                    ds: Dataset = load_dataset(hf_name, config, split="train")
                else:
                    if hf_name == "Helsinki-NLP/opus_paracrawl":
                        ds: Dataset = load_dataset(
                            hf_name,
                            lang1=src_lang,
                            lang2=tgt_lang,
                            split="train"
                        )
                    else:
                        raise ValueError(f"Unsupported dataset configuration for {hf_name}")
            except Exception as e:
                logging.error(f"Failed to load {hf_name}: {e}")
                continue

            buffer_src, buffer_tgt = [], []

            for ex in ds:
                src, tgt = extract_pair(ex, src_lang, tgt_lang)
                if not src:
                    continue

                buffer_src.append(src)
                buffer_tgt.append(tgt)

                if len(buffer_src) < args.batch_size:
                    continue

                kept, next_id = process_batch(
                    buffer_src, buffer_tgt,
                    model, args.sim_threshold, out_f,
                    kept, next_id, args.max_examples
                )
                buffer_src.clear()
                buffer_tgt.clear()

                if args.max_examples and kept >= args.max_examples:
                    break

            # Process any remaining examples in the buffer
            if buffer_src and (not args.max_examples or kept < args.max_examples):
                kept, next_id = process_batch(
                    buffer_src, buffer_tgt,
                    model, args.sim_threshold, out_f,
                    kept, next_id, args.max_examples
                )

            if args.max_examples and kept >= args.max_examples:
                break

    logging.info(f"Finished processing. Kept {kept:,} sentence pairs. Output saved to {args.output_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Partial output saved.")
        sys.exit(130)
