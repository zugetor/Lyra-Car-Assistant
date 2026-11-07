#!/usr/bin/env python3
"""
step2_finetune.py  --  fine-tune the pruned Laya checkpoint on car commands

Wraps laya.train.finetune (ships with laya >= 0.4): train, calibrate the confidence
(per question type and option count), and save a checkpoint that laya.load() opens.

  base   laya_pruned_enth/        (step 1)
  data   data/train.jsonl         (gen_train_data.py)
  eval   data/val.jsonl           (eval_commands.tsv; scored before and after training)
  out    laya_finetuned_enth/

Options are shuffled every epoch (shuffle_options=("choice",)) so the model can't learn
"the 4th option is usually right". finetune_summary.json in the output holds the fitted
abstention thresholds: below them, the car should ask the driver again instead of acting. The held-out set (eval_heldout.tsv) is not used here;
run eval_laya.py afterwards to score it.

Usage:
  python step2_finetune.py                    # 4 epochs on MPS / CUDA / CPU (auto)
  python step2_finetune.py --dry-run          # parse the data and print the update budget only
  python step2_finetune.py --epochs 6 --device mps
"""
import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("USE_TF", "0")

from common import FINETUNED_DIR, HERE, PRUNED_DIR  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "train.jsonl"))
    ap.add_argument("--eval", default=os.path.join(DATA_DIR, "val.jsonl"))
    ap.add_argument("--base", default=PRUNED_DIR)
    ap.add_argument("--out", default=FINETUNED_DIR)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = micro-batch x grad-accum")
    ap.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from laya.train import TrainConfig, dry_run, finetune

    cfg = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                      loss=args.loss, shuffle_options=("choice",), eval_data=args.eval)
    if args.dry_run:
        print(json.dumps(dry_run(args.data, args.base, cfg), indent=2, default=str))
        return 0

    t0 = time.time()
    summary = finetune(args.data, args.base, args.out, config=cfg, device=args.device)
    summary["wall_minutes"] = round((time.time() - t0) / 60, 1)
    # laya keeps a resume copy while training; after a finished run it only doubles the folder size
    shutil.rmtree(os.path.join(args.out, "checkpoint_latest"), ignore_errors=True)
    with open(os.path.join(args.out, "finetune_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str)[:4000])
    print(f"[+] Saved fine-tuned checkpoint to {os.path.relpath(args.out)} in {summary['wall_minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
