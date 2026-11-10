#!/usr/bin/env python3
"""
web_browser/build_pages.py  --  build the static demo for GitHub Pages; the model comes from Hugging Face

Writes OUT/ (default _site/) with everything the page needs except the model:
  index.html, lyra.js, tokens.css, coi-sw.js   the page (coi-sw.js adds the isolation headers Pages can't send)
  speech.js, trim.js, thai_text.js      hold-to-talk: Thai speech-to-text in the browser
  vendor/                               ONNX Runtime Web, via fetch_vendor.py (SHA-256 checked)
  eval_heldout.tsv, eval_commands.tsv   the example commands
  config.js                             model = HF_REPO at HF_REVISION, speech = SPEECH_REPO at SPEECH_REVISION,
                                        both kept in the browser's cache

The visitor's browser downloads the model straight from huggingface.co (HF sends CORS headers) and keeps it
in Cache Storage, so the second visit loads it from disk. That cache is keyed by URL, which is why
HF_REVISION is a commit id and not "main". After a new `hf upload`, put its commit id here.

.github/workflows/pages.yml runs this on every push to master and deploys OUT/.

Usage:
  python web_browser/build_pages.py                     # -> _site/
  python web_browser/build_pages.py --revision <commit>
  python -m http.server 8082 --bind 127.0.0.1 --directory _site     # try it the way Pages serves it
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fetch_vendor  # noqa: E402

HF_REPO = "zugetor/Lyra-Car-Assistant-enth"
HF_REVISION = "531964c77c59ce742f32c26506b14e11a76ee0b3"
SPEECH_REPO = "zugetor/zipformer-CarCMD-TH"                 # the Thai speech-to-text model (sherpa-onnx bundle)
SPEECH_REVISION = "525ddc8fb0827399d06e1988db876b91ad231540"
PAGE = ["index.html", "lyra.js", "coi-sw.js", "tokens.css", "speech.js", "trim.js", "thai_text.js"]
EXAMPLES = ["eval_heldout.tsv", "eval_commands.tsv"]
MARKER = ".nojekyll"      # GitHub Pages: serve files as they are. Also marks OUT/ as ours to delete.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=os.path.join(ROOT, "_site"))
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--revision", default=HF_REVISION)
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        if not os.path.exists(os.path.join(args.out, MARKER)):
            raise SystemExit(f"[!] {args.out} isn't empty and wasn't built by this script; not touching it")
        shutil.rmtree(args.out)
    fetch_vendor.main()
    os.makedirs(os.path.join(args.out, "vendor"))
    for name in PAGE:
        shutil.copy(os.path.join(HERE, name), args.out)
    for name in fetch_vendor.FILES:
        shutil.copy(os.path.join(fetch_vendor.OUT, name), os.path.join(args.out, "vendor"))
    for name in EXAMPLES:
        shutil.copy(os.path.join(ROOT, name), args.out)
    cfg = {"model": f"https://huggingface.co/{args.repo}/resolve/{args.revision}/", "examples": "./",
           "cache": "lyra-model",
           "speech": f"https://huggingface.co/{SPEECH_REPO}/resolve/{SPEECH_REVISION}/", "speechCache": "zipformer-carcmd-th"}
    with open(os.path.join(args.out, "config.js"), "w", encoding="utf-8") as f:
        f.write("// written by web_browser/build_pages.py\nwindow.LYRA_CONFIG = " + json.dumps(cfg, indent=2) + ";\n")
    open(os.path.join(args.out, MARKER), "w").close()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(args.out) for f in fs)
    print(f"[+] {os.path.relpath(args.out)}/: {size / 2**20:.1f} MB, model from {cfg['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
