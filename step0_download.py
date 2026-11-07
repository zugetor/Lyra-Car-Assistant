#!/usr/bin/env python3
"""
step0_download.py  --  fetch everything the pipeline needs into ./cache/

  * Laya model   convaiinnovations/laya-multilingual @ SOURCE_REVISION   (~650 MB)
  * English text wikitext-103-raw-v1 train (counting) + test (held-out check)  (~315 MB)
  * Thai test text: 500 Wikipedia articles via the HF datasets-server rows API  (~5 MB)
  * with --langs en,th,zh also 500 Chinese articles, for step 1 --langs en,th,zh

Writes cache/manifest.json with the local paths; the later steps read it, so they run offline.
Already-downloaded files are reused. Set HF_TOKEN for faster Hub downloads.

Usage:
  python step0_download.py
  python step0_download.py --revision main      # newest model instead of the tested snapshot
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

from huggingface_hub import hf_hub_download, snapshot_download

from common import CACHE_DIR, HF_CACHE, MANIFEST, SOURCE_REPO, SOURCE_REVISION

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_TRAIN = ["wikitext-103-raw-v1/train-00000-of-00002.parquet",
                  "wikitext-103-raw-v1/train-00001-of-00002.parquet"]
WIKITEXT_TEST = "wikitext-103-raw-v1/test-00000-of-00001.parquet"

ROWS_API = "https://datasets-server.huggingface.co/rows"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIGS = {"th": "20231101.th", "zh": "20231101.zh"}
WIKI_OFFSETS = [5000, 10000, 15000, 20000, 25000]   # 100 articles from each offset
TEXT_DIR = os.path.join(CACHE_DIR, "test_text")


def fetch_rows(lang, offset):
    path = os.path.join(TEXT_DIR, f"{lang}_rows_{offset}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    q = urllib.parse.urlencode({"dataset": WIKI_DATASET, "config": WIKI_CONFIGS[lang],
                                "split": "train", "offset": offset, "length": 100})
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{ROWS_API}?{q}", headers={"User-Agent": "laya-pipeline"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not json.loads(data).get("rows"):
                raise ValueError("no rows in response")
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:  # the rows API rate-limits now and then
            wait = 5 * (attempt + 1)
            print(f"    {lang}@{offset}: {e} -- retry in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"[!] could not fetch {lang} rows at offset {offset}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=SOURCE_REVISION, help="model revision (default: tested snapshot)")
    ap.add_argument("--langs", default="en,th", help="en,th (default) or en,th,zh to also fetch Chinese test text")
    args = ap.parse_args()
    os.makedirs(TEXT_DIR, exist_ok=True)

    print(f"[*] Model {SOURCE_REPO}@{args.revision[:12]} ...")
    model_dir = snapshot_download(SOURCE_REPO, revision=args.revision, cache_dir=HF_CACHE)

    print("[*] English text (wikitext-103) ...")
    en_train = [hf_hub_download(WIKITEXT_REPO, f, repo_type="dataset", cache_dir=HF_CACHE) for f in WIKITEXT_TRAIN]
    en_test = [hf_hub_download(WIKITEXT_REPO, WIKITEXT_TEST, repo_type="dataset", cache_dir=HF_CACHE)]

    print("[*] Thai test text (Wikipedia) ...")
    th = [fetch_rows("th", o) for o in WIKI_OFFSETS]
    zh = [fetch_rows("zh", o) for o in WIKI_OFFSETS] if "zh" in args.langs.split(",") else []

    manifest = {"model_dir": model_dir, "model_repo": SOURCE_REPO, "model_revision": args.revision,
                "en_train": en_train, "en_test": en_test, "th_test": th, "zh_test": zh}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    for k, v in manifest.items():
        print(f"    {k:14} {f'{len(v)} file(s)' if isinstance(v, list) else v}")
    print(f"[+] Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
