#!/usr/bin/env python3
"""
web_test/server.py  --  try the 8-bit ONNX model in a browser

Type (or say) a Thai / English command and see what the car would do: the action, both
confidences against the ask-again thresholds, latency per call, and the tokens. Labeled presets
from the eval sets show right/wrong at a glance, and one button runs a whole eval set.

Runs runtime.CarAssistant (export/laya.onnx by default). Python standard library only, no web
framework. Listens on 127.0.0.1 so it is not reachable from other machines.

Usage:
  python web_test/server.py                      # http://127.0.0.1:8080
  python web_test/server.py --port 8090 --model laya_finetuned_enth   # torch checkpoint instead
  python web_test/server.py --gate laya          # ask again by laya's fitted confidence thresholds
"""
import argparse
import json
import os
import statistics
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from common import EVAL_COMMANDS, EVAL_HELDOUT  # noqa: E402
from eval_laya import lang_of, load_rows  # noqa: E402
from runtime import EXPORT_DIR, CarAssistant  # noqa: E402

SETS = {"heldout": EVAL_HELDOUT, "tuning": EVAL_COMMANDS}
CAR = None
LOCK = threading.Lock()   # one model, one request at a time


def correct(row, r):
    ok_c = r["control"] == row["control"]
    gold = row["payload"]
    if gold == "-":
        ok_p = True
    elif gold.isdigit():
        ok_p = r["payload"] == int(gold)
    else:
        ok_p = r["payload"] == gold
    return ok_c, ok_c and ok_p


def run_eval(name):
    rows = load_rows(SETS[name])
    misses, ms, n_c, n_e, asked, asked_laya = [], [], 0, 0, 0, 0
    for row in rows:
        with LOCK:
            r = CAR.decide(row["text"])
        ok_c, ok_e = correct(row, r)
        n_c += ok_c
        n_e += ok_e
        asked += r["ask_again"]
        asked_laya += r["laya_gate_would_ask"]
        ms.append(r["ms"]["total"])
        if not ok_e:
            misses.append({"text": row["text"], "expected": {row["control"]: row["payload"]}, "got": r["action"],
                           "confidence": r["control_confidence"], "ask_again": r["ask_again"]})
    return {"set": name, "n": len(rows), "control": n_c, "end_to_end": n_e, "ask_again": asked,
            "laya_gate_would_ask": asked_laya, "gate": CAR.gate,
            "median_ms": round(statistics.median(ms), 1), "misses": misses}


def presets():
    out = []
    for name, path in SETS.items():
        for row in load_rows(path):
            out.append({**row, "set": name, "lang": lang_of(row["text"])})
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self):  # health checks
        self.send_response(200 if self.path in ("/", "/index.html") else 404)
        self.end_headers()

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            return self.end_headers()
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/info":
            return self._send(200, {**CAR.info(), "presets": presets()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "send JSON"})
        if self.path == "/api/decide":
            text = str(body.get("text", "")).strip()
            if not text:
                return self._send(400, {"error": "empty text"})
            if len(text) > 500:
                return self._send(400, {"error": "text longer than 500 characters"})
            with LOCK:
                return self._send(200, CAR.decide(text))
        if self.path == "/api/eval":
            name = body.get("set", "heldout")
            if name not in SETS:
                return self._send(400, {"error": f"set must be one of {list(SETS)}"})
            return self._send(200, run_eval(name))
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # keep the console quiet except for errors
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    global CAR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(EXPORT_DIR, "laya.onnx"),
                    help="an .onnx file in a bundle folder, or a laya checkpoint folder")
    ap.add_argument("--gate", default="margin", choices=["margin", "laya"],
                    help="ask-again rule: margin between the two best options (default) or laya's fitted thresholds")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    model = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    print(f"[*] Loading {os.path.relpath(model, ROOT)} ...", flush=True)
    CAR = CarAssistant(model, gate=args.gate)
    info = CAR.info()
    print(f"[+] {info['backend']}, {info['model_mb']} MB, vocab {info['vocab']:,}, loaded in {info['load_s']}s, "
          f"gate {info['gate']}", flush=True)
    print(f"[+] http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
