#!/usr/bin/env python3
"""
check_step1.py  --  test the pruned Laya checkpoint from step1_prune_vocab.py

Hard checks (exit code 1 if any fails):
  1. special token ids unchanged (<pad> <eos> <bos> <unk> <mask>)
  2. car commands in test_commands.txt (TH / EN / CN) tokenize exactly as before
     (as written, lower-cased and upper-cased)
     + numbers: 0-10,000, Thai digits, numbers glued to Thai, decimals, %, °C, kW ...
  3. every pure-Thai run in held-out Wikipedia text tokenizes exactly as before
     (and every pure-Chinese run, when step 1 ran with --langs en,th,zh)
  4. English (wikitext-103 test, not used for counting): share of word occurrences that
     tokenize exactly as before >= --min-en-parity (default 98%)
  5. weights: every kept embedding row == its source row; every other tensor byte-identical

Info only: line-level parity on mixed text, token count change, tokenizer RAM (fresh process), sizes.

Usage:
  python check_step1.py
  python check_step1.py --pruned other_dir --min-en-parity 95
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from tokenizers import Tokenizer

from common import (EMBED_KEY, PRUNED_DIR, TEST_COMMANDS, is_cjk, is_cjk_punct, is_thai, iter_texts,
                    load_manifest, read_list)

RESULTS = []


def record(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"   {'PASS' if ok else 'FAIL'}  {name}: {detail}")


def word_script(w):
    if any(is_thai(c) for c in w):
        return "thai"
    if any(is_cjk(c) for c in w):
        return "cjk"
    if all(ord(c) < 128 for c in w):
        return "ascii"
    return "other"


def same(orig, pruned, texts):
    a = orig.encode_batch(texts, add_special_tokens=False)
    b = pruned.encode_batch(texts, add_special_tokens=False)
    return [x.tokens == y.tokens for x, y in zip(a, b)], a, b


def script_runs(lines, pred):
    runs, cur = set(), []
    for line in lines:
        for ch in line + "\n":
            if pred(ch):
                cur.append(ch)
            elif cur:
                runs.add("".join(cur))
                cur = []
    return sorted(runs)


def mixed_text_info(orig, pruned, lines, label):
    ok, a, b = same(orig, pruned, lines)
    n_o, n_p = sum(len(x.ids) for x in a), sum(len(y.ids) for y in b)
    words = Counter(w for line in lines for w in line.split())
    uniq = sorted(words)
    wok, wa, wb = same(orig, pruned, uniq)
    by = defaultdict(lambda: [0, 0])
    examples = defaultdict(list)
    for w, s, x, y in zip(uniq, wok, wa, wb):
        sc = word_script(w)
        by[sc][0] += words[w] * s
        by[sc][1] += words[w]
        if not s and len(examples[sc]) < 2:
            examples[sc].append(f"{w!r}: {x.tokens} -> {y.tokens}")
    print(f"   info  {label}: {len(lines):,} lines, {sum(ok) / len(ok) * 100:.2f}% identical, "
          f"tokens {n_o:,} -> {n_p:,} ({(n_p / n_o - 1) * 100:+.2f}%)")
    for sc in ("thai", "cjk", "ascii", "other"):
        if by[sc][1]:
            print(f"         {sc:5} word occurrences identical {by[sc][0] / by[sc][1] * 100:6.2f}%"
                  + (f"   e.g. {examples[sc][0]}" if examples[sc] else ""))
    return by


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n, np.memmap(path, dtype=np.uint8, mode="r")


def check_weights(orig_st, pruned_st, kept_old):
    ho, so, mo = read_st(orig_st)
    hp, sp, mp = read_st(pruned_st)
    if set(ho) != set(hp):
        return False, "tensor names differ"
    for k in ho:
        a = mo[so + ho[k]["data_offsets"][0]: so + ho[k]["data_offsets"][1]]
        b = mp[sp + hp[k]["data_offsets"][0]: sp + hp[k]["data_offsets"][1]]
        if k == EMBED_KEY:
            ra = np.asarray(a).reshape(ho[k]["shape"][0], -1)[np.array(kept_old)]
            rb = np.asarray(b).reshape(hp[k]["shape"][0], -1)
            if ra.shape != rb.shape or not np.array_equal(ra, rb):
                return False, "embedding rows differ"
        elif ho[k]["shape"] != hp[k]["shape"] or not np.array_equal(a, b):
            return False, f"{k} differs"
    return True, f"{len(ho)} tensors; embedding {ho[EMBED_KEY]['shape']} -> {hp[EMBED_KEY]['shape']}"


CHILD = r'''
import os, re, subprocess, sys, psutil
from tokenizers import Tokenizer
p = psutil.Process()
def heap():
    try:
        o = subprocess.run(["vmmap", "--summary", str(os.getpid())], capture_output=True, text=True).stdout
        return re.search(r"^TOTAL\s+.*$", o.split("MALLOC ZONE")[-1], re.M).group(0).split()[6]
    except Exception:
        return "n/a (macOS vmmap only)"
r0, h0 = p.memory_info().rss / 2**20, heap()
t = Tokenizer.from_file(sys.argv[1])
r1, h1 = p.memory_info().rss / 2**20, heap()
print(f"load RSS +{r1 - r0:.0f} MB, live heap {h1}")
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pruned", default=PRUNED_DIR)
    ap.add_argument("--min-en-parity", type=float, default=98.0,
                    help="min %% of English word occurrences that must tokenize as before (default 98)")
    args = ap.parse_args()

    man = load_manifest()
    orig_tok_path = os.path.join(man["model_dir"], "tokenizer", "tokenizer.json")
    pruned_tok_path = os.path.join(args.pruned, "tokenizer", "tokenizer.json")
    orig, pruned = Tokenizer.from_file(orig_tok_path), Tokenizer.from_file(pruned_tok_path)
    print(f"vocab {orig.get_vocab_size():,} -> {pruned.get_vocab_size():,}\n")

    print("== tokenizer")
    moved = [t for t in ("<pad>", "<eos>", "<bos>", "<unk>", "<mask>") if orig.token_to_id(t) != pruned.token_to_id(t)]
    record("special token ids", not moved, "unchanged" if not moved else f"moved: {moved}")

    base = read_list(TEST_COMMANDS)
    cmds = sorted({v for c in base for v in (c, c.lower(), c.upper())})
    ok, a, b = same(orig, pruned, cmds)
    bad = [(c, x.tokens, y.tokens) for c, s, x, y in zip(cmds, ok, a, b) if not s]
    record("car commands", not bad, f"{len(base)} commands x (as-is, lower, upper) = {len(cmds)} strings, "
           f"{sum(ok)} identical" + "".join(f"\n         DIFF {c!r}: {x} -> {y}" for c, x, y in bad[:10]))

    nums = [str(i) for i in range(10001)]
    thai_digits = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
    numbers = (nums + [n.translate(thai_digits) for n in nums[:1001]]
               + [f"ตั้ง {n} องศา" for n in nums[:101]] + [f"ตั้ง{n}องศา" for n in nums[:101]]
               + [f"set it to {n}%" for n in nums[:101]]
               + ["1,000", "10,000", "100,000", "1,234.56", "3.5", "22.5", "0.5", "-5", "+3", "25°C", "25 °C",
                  "80%", "150 kW", "11kW", "7.4 kWh", "220V", "32A", "1/0", "17–33", "50-100", "2026-11-07", "12:30"])
    ok, a, b = same(orig, pruned, numbers)
    bad = [(n, x.tokens, y.tokens) for n, s, x, y in zip(numbers, ok, a, b) if not s]
    record("numbers", not bad, f"0-10,000, Thai digits 0-1,000, glued to Thai, units/decimals: "
           f"{sum(ok):,}/{len(numbers):,} identical" + "".join(f"\n         DIFF {n!r}: {x} -> {y}" for n, x, y in bad[:10]))

    rep = json.load(open(os.path.join(args.pruned, "prune_report.json")))
    with_zh = "zh" in rep.get("langs", ["zh"]) and man.get("zh_test")
    th_lines = list(iter_texts(man["th_test"]))
    zh_lines = list(iter_texts(man["zh_test"])) if with_zh else []
    th_runs = script_runs(th_lines, is_thai)
    zh_runs = script_runs(zh_lines, lambda c: is_cjk(c) and not is_cjk_punct(c))
    for label, runs in (("pure Thai runs", th_runs), ("pure Chinese runs", zh_runs)):
        if not runs:
            continue
        ok, a, b = same(orig, pruned, runs)
        bad = [r for r, s in zip(runs, ok) if not s]
        record(label, not bad, f"{len(runs) - len(bad):,}/{len(runs):,} identical" + (f"; e.g. {bad[:3]}" if bad else ""))

    en_lines = list(iter_texts(man["en_test"]))
    by = mixed_text_info(orig, pruned, en_lines, "English held-out")
    en_par = by["ascii"][0] / by["ascii"][1] * 100
    record("English word parity", en_par >= args.min_en_parity, f"{en_par:.2f}% of ASCII word occurrences identical "
           f"(need >= {args.min_en_parity}%)")
    mixed_text_info(orig, pruned, th_lines, "Thai Wikipedia (mixed text)")
    if zh_lines:
        mixed_text_info(orig, pruned, zh_lines, "Chinese Wikipedia (mixed text)")

    print("\n== weights")
    with open(os.path.join(args.pruned, "kept_old_ids.json")) as f:
        kept_old = json.load(f)
    ok, detail = check_weights(os.path.join(man["model_dir"], "model.safetensors"),
                               os.path.join(args.pruned, "model.safetensors"), kept_old)
    record("weights", ok, detail)

    print("\n== tokenizer RAM (fresh process)")
    for label, path in (("original", orig_tok_path), ("pruned  ", pruned_tok_path)):
        out = subprocess.run([sys.executable, "-c", CHILD, path], capture_output=True, text=True)
        print(f"   info  {label}: {out.stdout.strip() or out.stderr.strip()[-300:]}")

    rep = json.load(open(os.path.join(args.pruned, "prune_report.json")))
    print(f"\n== size\n   info  params {rep['params_after'] / 1e6:.1f} M | FP16 safetensors "
          f"{rep['files_mb']['model.safetensors']} MB | est. INT8 ~{rep['params_after'] / 2**20:.0f} MB")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{'ALL CHECKS PASSED' if not failed else 'FAILED: ' + ', '.join(failed)} ({len(RESULTS) - len(failed)}/{len(RESULTS)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
