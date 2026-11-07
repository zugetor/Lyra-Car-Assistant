"""Shared paths, constants and helpers for the Laya EN/TH/CN pipeline."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")           # downloads + intermediates (git-ignored)
HF_CACHE = os.path.join(CACHE_DIR, "hf")
MANIFEST = os.path.join(CACHE_DIR, "manifest.json")
PRUNED_DIR = os.path.join(HERE, "laya_pruned_enth")
FINETUNED_DIR = os.path.join(HERE, "laya_finetuned_enth")       # step2_finetune.py output
ASK_AGAIN_MARGIN = 0.5   # runtime margin gate: ask again when p(best) - p(runner-up) is below this
KEEP_TEXT = os.path.join(HERE, "keep_text.txt")             # English text whose tokens are always kept
TEST_COMMANDS = os.path.join(HERE, "test_commands.txt")     # TH/EN/CN commands that must tokenize as before
CAR_QUESTIONS = os.path.join(HERE, "car_questions.json")    # Laya questions used at runtime (tokens kept)
QUESTIONS_DIR = os.path.join(HERE, "questions")             # question variants tried by eval_laya.py (tokens kept)
EVAL_COMMANDS = os.path.join(HERE, "eval_commands.tsv")     # labeled commands for eval_laya.py
EVAL_HELDOUT = os.path.join(HERE, "eval_heldout.tsv")       # labeled paraphrases, never used to tune questions


def question_strings(obj):
    """Every string in car_questions.json (keys and values, minus _comment), as Laya renders them."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            out.append(k)
            if isinstance(v, str):
                out.append(f"{k}: {v}")
            out.extend(question_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(question_strings(v))
    elif isinstance(obj, str):
        out.append(obj)
        out.append(f"choice question: {obj}")
    return out

SOURCE_REPO = "convaiinnovations/laya-multilingual"
SOURCE_REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"   # snapshot the pipeline was tested on
EMBED_KEY = "encoder.embeddings.tok_embeddings.weight"          # the only vocab-sized weight
META = "\u2581"  # SentencePiece word-start marker used by the Gemma tokenizer


# --------------------------------------------------------------------------- manifest
def load_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"[!] {MANIFEST} not found. Run step0_download.py first.")
    with open(MANIFEST) as f:
        return json.load(f)


def read_list(path):
    """Non-empty lines of a text file, skipping # comments."""
    with open(path, encoding="utf-8") as f:
        return [s.strip() for s in f if s.strip() and not s.lstrip().startswith("#")]


# --------------------------------------------------------------------------- scripts
def is_thai(ch):
    return 0x0E00 <= ord(ch) <= 0x0E7F


def is_cjk_punct(ch):
    o = ord(ch)
    return 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF


def is_cjk(ch):
    o = ord(ch)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or is_cjk_punct(ch)


def is_latin_letter(ch):
    return 0x00C0 <= ord(ch) <= 0x024F and ch not in "×÷"


def category(tok, added):
    """special | byte | space | thai | cjk | ascii | other"""
    if tok in added:
        return "special"
    if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
        return "byte"
    body = tok.replace(META, "")
    if body == "":
        return "space"
    if any(is_thai(c) for c in body):
        return "thai"
    if any(is_cjk(c) for c in body):
        return "cjk"
    if all(ord(c) < 128 for c in body):
        return "ascii"
    return "other"


# --------------------------------------------------------------------------- text
_WIKITEXT_FIXES = [
    (re.compile(r" @(.)@ "), r"\1"),             # "1 @,@ 000" -> "1,000", "state @-@ of" -> "state-of"
    (re.compile(r" ([.,;:!?%)\]}])"), r"\1"),     # "word ," -> "word,"
    (re.compile(r"([(\[{$]) "), r"\1"),           # "( word" -> "(word"
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r'" (.*?) "'), r'"\1"'),
]


def detok_wikitext(line):
    """wikitext-103 'raw' still has spaced-out punctuation; undo it so it looks like real text."""
    for pat, rep in _WIKITEXT_FIXES:
        line = pat.sub(rep, line)
    return line


def iter_texts(paths):
    """Yield non-empty text lines from .parquet (wikitext "text" column), .json (datasets-server rows) or .txt."""
    for p in paths:
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(p).iter_batches(columns=["text"], batch_size=20000):
                for t in batch.column(0).to_pylist():
                    if t and t.strip() and not t.lstrip().startswith("= "):  # skip "= Heading =" lines
                        yield detok_wikitext(t.strip())
        elif p.endswith(".json"):
            with open(p, encoding="utf-8") as f:
                for r in json.load(f).get("rows", []):
                    for x in r["row"]["text"].split("\n"):
                        if x.strip():
                            yield x.strip()
        else:
            with open(p, encoding="utf-8") as f:
                for x in f:
                    if x.strip():
                        yield x.strip()
