#!/usr/bin/env python3
"""
web_browser/serve.py  --  static file server for the in-browser page (no inference here)

Browsers won't fetch() files or WASM from file://, so the page needs some HTTP server. This one only
hands out files; the model runs in the browser. It:
  * serves just web_browser/, export/ and the two eval .tsv files from the repo (everything else 404s)
  * sends Cross-Origin-Opener/Embedder-Policy headers, which let ONNX Runtime Web use several threads
  * listens on 127.0.0.1 only

Any static server works too (single-threaded WASM without those headers), e.g.
  python3 -m http.server 8081 --bind 127.0.0.1      then open http://127.0.0.1:8081/web_browser/

Usage:
  python web_browser/serve.py                 # http://127.0.0.1:8081/
  python web_browser/serve.py --port 8090
"""
import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_DIRS = ("/web_browser/", "/export/")
ALLOWED_FILES = ("/eval_commands.tsv", "/eval_heldout.tsv")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".js": "text/javascript", ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".data": "application/octet-stream",
                      ".json": "application/json", ".tsv": "text/tab-separated-values; charset=utf-8"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web_browser/")
            return self.end_headers()
        if ".." in path or not (path.startswith(ALLOWED_DIRS) or path in ALLOWED_FILES):
            return self.send_error(404)
        return super().do_GET()

    def do_HEAD(self):
        return self.do_GET() if self.path in ("/", "/index.html") else super().do_HEAD()

    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    for need in ("export/laya.onnx", "web_browser/vendor/ort-wasm-simd-threaded.wasm"):
        if not os.path.exists(os.path.join(ROOT, need)):
            print(f"[!] missing {need}: run step3_export_onnx.py / web_browser/fetch_vendor.py first")
    print(f"[+] http://{args.host}:{args.port}/web_browser/  (static files only; the model runs in the browser)",
          flush=True)
    ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=ROOT)).serve_forever()


if __name__ == "__main__":
    main()
