#!/usr/bin/env bash
# Full pipeline: .venv -> step 0 download -> step 1 prune -> check.
# Extra args go to step 1, e.g.  ./run.sh --min-count 100   or   ./run.sh --sweep
# Uses python3 (override with PYTHON=...). Needs Python 3.11+.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
ver() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true; }

"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
  || { echo "!! Need Python 3.11+, found: $PY $(ver "$PY")"; exit 1; }

# (re)build .venv when it is missing, broken, or made with another Python version
if [ "$(ver .venv/bin/python)" != "$(ver "$PY")" ]; then
  echo "==> Creating .venv with $PY $(ver "$PY")"
  rm -rf .venv
  "$PY" -m venv .venv
fi
echo "==> Installing requirements"
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt

echo "==> Step 0: download model + text"
.venv/bin/python step0_download.py

echo "==> Step 1: prune vocab"
.venv/bin/python step1_prune_vocab.py "$@"
for a in "$@"; do [ "$a" = "--sweep" ] && exit 0; done

echo "==> Check step 1"
.venv/bin/python check_step1.py

echo "==> Eval: Laya on car commands, original vs pruned"
.venv/bin/python eval_laya.py
