#!/usr/bin/env python3
"""
web_browser/fetch_vendor.py  --  download ONNX Runtime Web once, so the page runs offline

Fetches the pinned onnxruntime-web WASM build from jsDelivr into web_browser/vendor/ (git-ignored)
and checks each file against the SHA-256 jsDelivr publishes for it. Run it once; after that the
browser page needs no network.

Usage:
  python web_browser/fetch_vendor.py
"""
import base64
import hashlib
import json
import os
import sys
import urllib.request

VERSION = "1.30.0"      # onnxruntime-web; plain WASM build (no WebGPU), works in any modern browser
FILES = ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
CDN = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/"
META = f"https://data.jsdelivr.com/v1/package/npm/onnxruntime-web@{VERSION}/flat"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lyra-car-assistant"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    hashes = {f["name"].rsplit("/", 1)[-1]: f["hash"] for f in json.loads(get(META))["files"]
              if f["name"].startswith("/dist/")}
    for name in FILES:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            data = open(path, "rb").read()
        else:
            print(f"[*] {name} ...", flush=True)
            data = get(CDN + name)
        digest = base64.b64encode(hashlib.sha256(data).digest()).decode()
        if digest != hashes.get(name):
            raise SystemExit(f"[!] {name}: SHA-256 does not match jsDelivr's ({digest} != {hashes.get(name)})")
        with open(path, "wb") as f:
            f.write(data)
        print(f"[+] {name}: {len(data) / 2**20:.1f} MB, sha256 ok")
    with open(os.path.join(OUT, "VERSION"), "w") as f:
        f.write(f"onnxruntime-web {VERSION}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
