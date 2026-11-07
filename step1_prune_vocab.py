#!/usr/bin/env python3
"""
step1_prune_vocab.py  --  cut the Laya vocab down to English + Thai (+ Chinese with --langs en,th,zh)

Source: convaiinnovations/laya-multilingual (mmBERT-base + Laya decision head,
256k Gemma 2 BPE vocab). Only one weight is vocab-sized:
encoder.embeddings.tok_embeddings.weight [256000, 768]. No MLM decoder is stored.

Keep:
  * added/special tokens, byte-fallback tokens <0x00>..<0xFF>, bare "▁" tokens
  * every token holding a Thai char (and, with --langs ...,zh, every CJK token incl. CJK punctuation)
  * single-char tokens: ASCII, Latin letters (é ü ñ ...), and SYMBOLS below
  * ASCII tokens seen >= --min-count times when tokenizing English text (wikitext-103)
  * every token used by keep_text.txt (Laya prompt words, control keys/payloads, car commands,
    common car words), test_commands.txt and car_questions.json + questions/*.json (the Laya questions used at
    runtime): as written, lower-cased, upper-cased, and each word glued to Thai text
    without a space
  * closure: every token a BPE merge needs to build a kept token
Drop the rest (Chinese / CJK unless asked for, Cyrillic, Arabic, kana, Hangul, accented-Latin words,
non-English ASCII words, code fragments, emoji, ...).

Writes a Laya checkpoint folder that `laya.load(OUT_DIR)` can open:
  OUT_DIR/config.json, rl_agent_config.json, README.md
  OUT_DIR/encoder/config.json          vocab_size updated
  OUT_DIR/tokenizer/tokenizer.json     pruned vocab + merges, ids remapped
  OUT_DIR/tokenizer/tokenizer_config.json
  OUT_DIR/model.safetensors            embedding rows sliced, rest copied as-is (FP16)
  OUT_DIR/kept_old_ids.json            new id i == old id kept_old_ids[i]
  OUT_DIR/prune_report.json

Inputs come from cache/manifest.json (run step0_download.py first). No torch needed.

Usage:
  python step1_prune_vocab.py                    # English + Thai, --min-count 20
  python step1_prune_vocab.py --langs en,th,zh   # also keep Chinese (+24k tokens, ~+18 MB INT8)
  python step1_prune_vocab.py --sweep            # print vocab / size per --min-count, write nothing
  python step1_prune_vocab.py --min-count 100    # smaller model, more English words split
"""
import argparse
import glob
import hashlib
import itertools
import json
import os
import shutil
import struct
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from tokenizers import Tokenizer

from common import (CACHE_DIR, CAR_QUESTIONS, EMBED_KEY, KEEP_TEXT, META, PRUNED_DIR, QUESTIONS_DIR,
                    SOURCE_REPO, TEST_COMMANDS, category, is_latin_letter, iter_texts, load_manifest,
                    question_strings, read_list)

# Non-ASCII symbols worth keeping (these sit outside the Thai/CJK ranges).
# U+200B zero-width space is common in Thai text as a word-break hint; NBSP / ZWJ / ZWNJ / BOM too.
SYMBOLS = set("°℃℉’‘“”…×÷–—•·€£¥№™®©±µ") | set("\u200b\u200c\u200d\u00a0\u2060\ufeff")

# Measured on wikitext-103 test (held out), share of English word occurrences that
# tokenize differently after pruning, and estimated INT8 size:
#   min 3 -> 147k vocab, 0.04%, ~227 MB | min 20 -> 129k, 0.22%, ~214 MB
#   min 100 -> 103k, 1.4%, ~195 MB      | min 300 -> 79k, 4.3%, ~178 MB
DEFAULT_MIN_COUNT = 20


DEFAULT_LANGS = "en,th"
KEEP_CATS = {"special", "byte", "space", "thai"}   # + "cjk" when --langs has zh (set in main)


def keep_variants(text):
    """The text as written, lower- and upper-cased, plus each word glued to a Thai char with no space
    (as in "ต่อWiFi"), because a glued word tokenizes without the leading "▁"."""
    for v in (text, text.lower(), text.upper()):
        yield v
        for w in v.split():
            yield "ก" + w


def always_keep_by_rule(tok, cat):
    if cat in KEEP_CATS:
        return True
    body = tok.replace(META, "")
    if len(body) == 1 and (ord(body) < 128 or is_latin_letter(body) or body in SYMBOLS):
        return True
    if body and all(c in SYMBOLS for c in body):
        return True
    return False


def count_tokens(tok, paths, vocab_size, cache_file):
    """Count how often each token id appears when tokenizing `paths`. Cached on disk."""
    key = hashlib.sha1(json.dumps([[os.path.basename(p), os.path.getsize(p)] for p in paths]).encode()).hexdigest()[:12]
    key_file = cache_file + ".key"
    if os.path.exists(cache_file) and os.path.exists(key_file) and open(key_file).read().strip() == key:
        print(f"[*] Using cached English token counts: {os.path.relpath(cache_file)}")
        return np.load(cache_file), None
    print("[*] Tokenizing English text to count token use (~1 min) ...")
    counts = np.zeros(vocab_size, dtype=np.int64)
    t0, n_lines, batch = time.time(), 0, []

    def flush():
        encs = tok.encode_batch(batch, add_special_tokens=False)
        ids = np.fromiter(itertools.chain.from_iterable(e.ids for e in encs), dtype=np.int64)
        counts[:] += np.bincount(ids, minlength=vocab_size)

    for line in iter_texts(paths):
        batch.append(line)
        if len(batch) == 20000:
            flush()
            n_lines += len(batch)
            batch = []
    if batch:
        flush()
        n_lines += len(batch)
    print(f"    {n_lines:,} lines, {counts.sum():,} tokens, {time.time() - t0:.0f}s")
    np.save(cache_file, counts)
    with open(key_file, "w") as f:
        f.write(key)
    return counts, n_lines


def split_merge(m):
    return tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))


def build_keep_set(vocab, cats, counts, min_count, forced_ids, producers):
    keep = set(forced_ids)
    for t, i in vocab.items():
        if always_keep_by_rule(t, cats[i]) or (cats[i] == "ascii" and counts[i] >= min_count):
            keep.add(i)
    # closure: a kept token must stay buildable through every merge that makes it
    id2tok = {i: t for t, i in vocab.items()}
    stack = list(keep)
    while stack:
        t = id2tok[stack.pop()]
        for a, b in producers.get(t, ()):
            for part in (a, b):
                pid = vocab.get(part)
                if pid is not None and pid not in keep:
                    keep.add(pid)
                    stack.append(pid)
    return keep


def prune_tokenizer_json(tj, kept_old):
    """Return a pruned copy-in-place of tokenizer.json dict `tj` keeping old ids `kept_old` (sorted)."""
    model = tj["model"]
    id2tok = {i: t for t, i in model["vocab"].items()}
    remap = {old: new for new, old in enumerate(kept_old)}
    new_vocab = {id2tok[old]: new for new, old in enumerate(kept_old)}
    model["merges"] = [m for m in model["merges"]
                       if (lambda a, b: a in new_vocab and b in new_vocab and (a + b) in new_vocab)(*split_merge(m))]
    model["vocab"] = new_vocab
    for t in tj.get("added_tokens", []):
        t["id"] = remap[t["id"]]
    _remap_post_processor(tj.get("post_processor"), remap)
    return tj, remap


def _remap_post_processor(obj, remap):
    if isinstance(obj, dict):
        st = obj.get("special_tokens")
        if isinstance(st, dict):
            for v in st.values():
                if isinstance(v, dict) and "ids" in v:
                    v["ids"] = [remap[i] for i in v["ids"]]
        for v in obj.values():
            _remap_post_processor(v, remap)
    elif isinstance(obj, list):
        for v in obj:
            _remap_post_processor(v, remap)


def slice_safetensors(src, dst, key, rows):
    """Copy a .safetensors file, keeping only `rows` of tensor `key`. Pure byte copy, no torch."""
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    meta = header.pop("__metadata__", None)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header, chunks, off = {}, [], 0
    for name, info in items:
        s, e = info["data_offsets"]
        buf = mm[data_start + s: data_start + e]
        shape = list(info["shape"])
        if name == key:
            row_bytes = (e - s) // shape[0]
            buf = np.ascontiguousarray(np.asarray(buf).reshape(shape[0], row_bytes)[rows])
            shape[0] = len(rows)
        new_header[name] = {"dtype": info["dtype"], "shape": shape, "data_offsets": [off, off + buf.nbytes]}
        chunks.append(buf)
        off += buf.nbytes
    if meta is not None:
        new_header["__metadata__"] = meta
    hb = json.dumps(new_header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(memoryview(np.ascontiguousarray(c)))
    return {k: v["shape"] for k, v in new_header.items() if k != "__metadata__"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                    help=f"keep ASCII tokens seen at least this often in English text (default {DEFAULT_MIN_COUNT})")
    ap.add_argument("--out", default=PRUNED_DIR)
    ap.add_argument("--sweep", action="store_true", help="print vocab size for several --min-count values and exit")
    ap.add_argument("--langs", default=DEFAULT_LANGS, help="en,th (default) or en,th,zh to also keep Chinese")
    args = ap.parse_args()
    langs = set(args.langs.split(","))
    if "zh" in langs:
        KEEP_CATS.add("cjk")

    man = load_manifest()
    src = man["model_dir"]
    tok_path = os.path.join(src, "tokenizer", "tokenizer.json")
    with open(tok_path, encoding="utf-8") as f:
        tj = json.load(f)
    tok = Tokenizer.from_file(tok_path)
    model = tj["model"]
    assert model["type"] == "BPE", model["type"]
    vocab = model["vocab"]
    V = len(vocab)
    added = {t["content"] for t in tj.get("added_tokens", [])}
    cats = {i: category(t, added) for t, i in vocab.items()}
    print(f"[*] Source vocab {V:,} | merges {len(model['merges']):,} | added tokens {len(added)}")

    producers = defaultdict(list)
    for m in model["merges"]:
        a, b = split_merge(m)
        producers[a + b].append((a, b))

    forced = {t["id"] for t in tj.get("added_tokens", [])}
    q_strings = []
    for qpath in [CAR_QUESTIONS] + sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.json"))):
        with open(qpath, encoding="utf-8") as f:
            q_strings += question_strings(json.load(f))
    keep_text = read_list(KEEP_TEXT) + read_list(TEST_COMMANDS) + q_strings
    for text in keep_text:
        for variant in keep_variants(text):
            forced.update(tok.encode(variant, add_special_tokens=False).ids)
    if "zh" not in langs:
        forced = {i for i in forced if cats[i] != "cjk"}
    print(f"[*] Languages: {','.join(sorted(langs))}")
    print(f"[*] keep_text.txt + test_commands.txt + car_questions.json: {len(keep_text)} strings "
          f"-> {len(forced):,} forced tokens")

    counts, n_lines = count_tokens(tok, man["en_train"], V, os.path.join(CACHE_DIR, "en_token_counts.npy"))

    if args.sweep:
        print("\n min-count | kept vocab | embedding params | est. INT8 model")
        non_embed = 322_000_000 - V * 768
        for mc in (1, 3, 5, 10, 20, 50, 100, 300, 1000):
            k = len(build_keep_set(vocab, cats, counts, mc, forced, producers))
            print(f" {mc:9} | {k:10,} | {k * 768 / 1e6:13.1f} M | ~{(non_embed + k * 768) / 2**20:.0f} MB")
        return 0

    kept_old = sorted(build_keep_set(vocab, cats, counts, args.min_count, forced, producers))
    n_merges_before = len(model["merges"])
    tj, remap = prune_tokenizer_json(tj, kept_old)
    NV = len(kept_old)

    kept_by = Counter(cats[i] for i in kept_old)
    all_by = Counter(cats.values())
    print(f"[*] Kept {NV:,} / {V:,} tokens ({NV / V * 100:.1f}%), merges {len(tj['model']['merges']):,} / {n_merges_before:,}")
    for c in ("special", "byte", "space", "thai", "cjk", "ascii", "other"):
        print(f"      {c:8} {kept_by.get(c, 0):7,} / {all_by.get(c, 0):7,}")

    # ---- tokenizer
    os.makedirs(os.path.join(args.out, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "encoder"), exist_ok=True)
    new_tok_path = os.path.join(args.out, "tokenizer", "tokenizer.json")
    with open(new_tok_path, "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)
    shutil.copy(os.path.join(src, "tokenizer", "tokenizer_config.json"), os.path.join(args.out, "tokenizer"))
    new_tok = Tokenizer.from_file(new_tok_path)
    for name in ("<pad>", "<eos>", "<bos>", "<unk>", "<mask>"):
        assert tok.token_to_id(name) == new_tok.token_to_id(name), f"special id moved: {name}"

    # ---- configs
    with open(os.path.join(src, "encoder", "config.json")) as f:
        ecfg = json.load(f)
    for k in ("pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "mask_token_id"):
        if k in ecfg:
            assert remap[ecfg[k]] == ecfg[k], f"{k} moved"
    ecfg["vocab_size"] = NV
    with open(os.path.join(args.out, "encoder", "config.json"), "w") as f:
        json.dump(ecfg, f, indent=2)
    for name in ("config.json", "rl_agent_config.json", "README.md"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), args.out)

    # ---- weights
    print("[*] Slicing embedding rows in model.safetensors ...")
    shapes = slice_safetensors(os.path.join(src, "model.safetensors"),
                               os.path.join(args.out, "model.safetensors"), EMBED_KEY, np.array(kept_old))
    n_params = int(sum(np.prod(s) for s in shapes.values()))

    with open(os.path.join(args.out, "kept_old_ids.json"), "w") as f:
        json.dump(kept_old, f)
    report = {
        "source": SOURCE_REPO, "source_revision": man.get("model_revision"), "min_count": args.min_count,
        "langs": sorted(langs),
        "corpus": [os.path.basename(p) for p in man["en_train"]], "corpus_lines": n_lines,
        "vocab_before": V, "vocab_after": NV,
        "merges_before": n_merges_before, "merges_after": len(tj["model"]["merges"]),
        "kept_by_category": dict(kept_by), "total_by_category": dict(all_by),
        "params_after": n_params, "embedding_shape": shapes[EMBED_KEY],
        "files_mb": {p: round(os.path.getsize(os.path.join(args.out, p)) / 2**20, 1)
                     for p in ("model.safetensors", "tokenizer/tokenizer.json")},
    }
    with open(os.path.join(args.out, "prune_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Params {n_params / 1e6:.1f} M | model.safetensors {report['files_mb']['model.safetensors']} MB (FP16) "
          f"| tokenizer.json {report['files_mb']['tokenizer/tokenizer.json']} MB")
    print(f"[+] Saved Laya checkpoint to {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
