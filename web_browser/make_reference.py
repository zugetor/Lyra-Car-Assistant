#!/usr/bin/env python3
"""
web_browser/make_reference.py  --  Python answers for the browser page's self-test

Writes web_browser/reference.json with what the Python side produces for the same bundle:
  tokens     token ids from the `tokenizers` library for every eval / test command, every option
             text, and odd strings (double spaces, leading / trailing spaces, emoji, tabs, accents)
  sequences  the full laya input (ids + mask positions) for some (command, question) pairs
  decisions  runtime.CarAssistant (INT8 ONNX, margin gate) on every eval command

The page's "Self-test" button runs lyra.js on the same inputs and reports any difference.
Re-run this after changing the model, the tokenizer or car_questions.json.

Usage:
  python web_browser/make_reference.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("USE_TF", "0")

from tokenizers import Tokenizer  # noqa: E402

from common import EVAL_COMMANDS, EVAL_HELDOUT, TEST_COMMANDS, read_list  # noqa: E402
from eval_laya import load_rows  # noqa: E402
from runtime import EXPORT_DIR, CarAssistant  # noqa: E402

ODD = ["  double  space", " leading space", "trailing space ", "", " ", "เปิด  แอร์", "tab\tseparated",
       "two\nlines", "emoji 😀 test", "café naïve Ünïcödé", "ต่อWiFiหน่อย", "1,000.5 kW 25°C", "!!!???",
       "ALL CAPS COMMAND", "MiXeD CaSe", "๑๒๓ เลขไทย", "a", "▁", "<0x41>", "​zero width"]


def main():
    car = CarAssistant(os.path.join(EXPORT_DIR, "laya.onnx"))
    tok = Tokenizer.from_file(os.path.join(EXPORT_DIR, "tokenizer", "tokenizer.json"))
    rows = load_rows(EVAL_COMMANDS) + load_rows(EVAL_HELDOUT)
    texts = [r["text"] for r in rows] + read_list(TEST_COMMANDS) + ODD

    qs = {"control": car.qcfg["control"], **car.Q.payload_qs}
    for q in qs.values():
        texts.append(f"{q['type']} question: {q['instructions']}")
        texts += [" " + (k if not v else f"{k}: {v}") for k, v in q["criteria"].items()]
    tokens = {t: tok.encode(t, add_special_tokens=False).ids for t in dict.fromkeys(texts)}

    from laya.common import build_sequence
    from laya.train import to_internal
    max_len, head_max_len = car.agent.cfg.get("max_len", 1024), car.agent.cfg.get("head_max_len", 256)
    sequences = []
    for r in rows[:12] + rows[-12:]:
        for qid, q in qs.items():
            ids, markers = build_sequence(car.agent.tok, r["text"], to_internal(qid, q),
                                          max_len=max_len, head_max_len=head_max_len)
            sequences.append({"text": r["text"], "qid": qid, "ids": ids, "markers": markers})

    decisions = []
    for r in rows:
        d = car.decide(r["text"])
        decisions.append({k: d[k] for k in ("text", "action", "control", "payload", "control_confidence",
                                            "control_margin", "payload_confidence", "payload_margin", "ask_again")}
                         | {"control_top": d["control_top"], "payload_top": d["payload_top"],
                            "expected": {r["control"]: r["payload"]}})

    out = os.path.join(HERE, "reference.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens, "sequences": sequences, "decisions": decisions}, f, ensure_ascii=False)
    print(f"[+] {os.path.relpath(out, ROOT)}: {len(tokens)} token cases, {len(sequences)} sequences, "
          f"{len(decisions)} decisions ({os.path.getsize(out) / 2**10:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
