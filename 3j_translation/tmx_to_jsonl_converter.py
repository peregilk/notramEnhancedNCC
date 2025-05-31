#!/usr/bin/env python3

import argparse
import os
import tarfile
import tempfile
import requests
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

NB_LICENSE = "CC0-1.0"
NN_LICENSE = "CC0-1.0"
CREATORS = [{"type": "publisher", "name": "European Language Grid - Nynorsk News Press Agency"}]
SEM_MODEL_NAME = "BAAI/bge-m3"

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total, unit='B', unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def extract_tgz(tgz_path, extract_dir):
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

def tmx_to_jsonl_with_sem(
    tmx_path, output_file, model_name=SEM_MODEL_NAME, device="cpu"
):
    model = SentenceTransformer(model_name, device=device)
    tree = ET.parse(tmx_path)
    root = tree.getroot()
    tus = list(root.iter('tu'))
    i = 0
    with open(output_file, "w", encoding="utf-8") as out_f, tqdm(
        total=len(tus), desc="Converting TMX to JSONL", unit="tu"
    ) as pbar:
        for tu in tus:
            entry = {}
            for tuv in tu.iter('tuv'):
                lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
                if lang:
                    lang = lang.lower()
                seg = tuv.find('seg')
                if lang in ('nb', 'nn') and seg is not None:
                    entry[lang] = seg.text
            if 'nb' in entry and 'nn' in entry:
                nb_text = entry['nb']
                nn_text = entry['nn']
                emb_nb = model.encode(nb_text, convert_to_tensor=True)
                emb_nn = model.encode(nn_text, convert_to_tensor=True)
                sim = float(util.cos_sim(emb_nb, emb_nn).item())
                distance = round(1.0 - sim, 4)
                i += 1
                json_obj = {
                    "id": f"nynorsk_pressekontor_{i}",
                    "nb": nb_text,
                    "nn": nn_text,
                    "semantic_distance": distance,
                    "nb_license": NB_LICENSE,
                    "nn_license": NN_LICENSE,
                    "nb_creators": CREATORS,
                    "nn_creators": CREATORS
                }
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Download, extract, and convert Norwegian TMX corpus to JSONL with semantic distance.")
    parser.add_argument("--url", type=str,
                        default="https://www.nb.no/sbfil/tekst/2011_2019_tm_npk_ntb.tar.gz",
                        help="URL to the .tar.gz file containing the TMX corpus.")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl file")
    parser.add_argument("--device", type=str, default="cpu", help="Device for embedding model, e.g., 'cuda' or 'cpu'.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tgz_path = os.path.join(tmpdir, "corpus.tar.gz")
        download_file(args.url, tgz_path)
        extract_tgz(tgz_path, tmpdir)

        # Find TMX file
        tmx_path = None
        for rootdir, dirs, files in os.walk(tmpdir):
            for fname in files:
                if fname.endswith('.tmx'):
                    tmx_path = os.path.join(rootdir, fname)
                    break
            if tmx_path:
                break
        if tmx_path is None:
            raise FileNotFoundError("No .tmx file found in the extracted archive.")

        tmx_to_jsonl_with_sem(tmx_path, args.output, device=args.device)
        print(f"Conversion complete. Output written to {args.output}")

if __name__ == "__main__":
    main()
