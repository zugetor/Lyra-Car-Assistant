#!/usr/bin/env python3
"""
step3_export_onnx.py  --  export the fine-tuned Laya model to ONNX and quantize it to INT8

Exports the whole DecisionModel (encoder + decision head), not just the encoder, with the
five inputs laya's ONNXAgent feeds:

  input_ids      int64 [rows, seq]      one row per question
  attention_mask int64 [rows, seq]
  marker_pos     int64 [rows, options]  where each option's <mask> sits
  marker_mask    bool  [rows, options]
  qtype          int64 [rows]           0 choice, 1 score, 2 noul
  -> logits [rows, options], act_logits [rows, 2]

Then 8-bit quantization without calibration data. Two kinds (--quant):

  weight8 (default)  weights stored as int8 (MatMulNBits, 8 bits, blocks of 128; embedding table
                     int8 per row), math done in float. Logits stay within ~0.5 of FP32 and come out
                     the same on native ONNX Runtime and ONNX Runtime Web.
  dynamic            onnxruntime's quantize_dynamic: activations are quantized to int8 at run time
                     too. Smaller kernels, but the results depend on the runtime: low-ranked logits
                     moved by up to ~9 between ORT optimization levels, and the browser build flipped
                     6 of 141 eval answers. Kept for comparison.

Both graphs are checked against torch on real questions of different lengths and option counts
before anything is written as final.

The weights are then moved out of the graph into laya.onnx.data (--single-file keeps one file).
With the weights inside the .onnx, ONNX Runtime reads the whole protobuf and copies every weight into
its own tensors: the session alone takes ~400 MB of private memory for a 212 MB file. With external
data it maps the weight file instead and the session takes ~50 MB (macOS), at the same speed.

Output bundle (loads with laya.load("export", backend="onnx", onnx_path="export/laya.onnx")):
  export/laya.onnx                   graph (~1 MB)
  export/laya.onnx.data              8-bit weights (~211 MB), memory-mapped by ONNX Runtime
  export/tokenizer/, rl_agent_config.json, encoder/config.json, config.json
  export/car_questions.json          the questions the runtime asks
  export/finetune_summary.json       fitted confidence thresholds (ask again below them)

Needs onnx, onnxruntime, onnxscript (requirements.txt). onnxruntime has no Python 3.15 wheel yet:
run with Python 3.11-3.14.

Usage:
  python step3_export_onnx.py
  python step3_export_onnx.py --src laya_pruned_enth --keep-fp32
  python step3_export_onnx.py --quant dynamic         # the older activation-quantized INT8
  python step3_export_onnx.py --single-file           # weights inside laya.onnx (more RAM at load)
"""
import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import CAR_QUESTIONS, EVAL_HELDOUT, FINETUNED_DIR, HERE  # noqa: E402

EXPORT_DIR = os.path.join(HERE, "export")
INPUTS = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUTS = ["logits", "act_logits"]
MB = 2 ** 20


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act_logits = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, act_logits


def build_batches(tok, cfg, questions, texts):
    """Real question rows (same encoding ONNXAgent uses), in a few batch shapes."""
    from laya.common import QTYPES, build_sequence
    from laya.onnx_agent import _collate_items_numpy
    from laya.train import to_internal

    max_len, head_max_len = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)
    rows = []
    for text in texts:
        for qid, q in questions.items():
            iq = to_internal(qid, q)
            ids, markers = build_sequence(tok, text, iq, max_len=max_len, head_max_len=head_max_len)
            rows.append({"ids": ids, "markers": markers, "qtype": QTYPES[iq["t"]]})
    # each question alone (lengths from ~30 to ~200 tokens, below and above the 128-token
    # sliding window), mixed batches, and everything at once
    singles = [[r] for r in rows[: len(questions)]]
    batches = singles + [rows[:2], rows[: len(questions)], rows[3: 3 + 2 * len(questions)], rows]
    return [_collate_items_numpy([b], tok.pad_token_id) for b in batches if b]


def torch_run(model, feed):
    with torch.no_grad():
        t = {k: torch.from_numpy(v) for k, v in feed.items()}
        lo, act = model(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
    return lo.numpy(), act.numpy()


def compare(name, sess, model, batches):
    worst, flips, total = 0.0, 0, 0
    for feed in batches:
        lo_t, act_t = torch_run(model, feed)
        lo_o, act_o = sess.run(OUTPUTS, feed)
        mask = feed["marker_mask"]
        diff = np.abs(np.where(mask, lo_t - lo_o, 0)).max()
        worst = max(worst, float(diff))
        pt, po = np.where(mask, lo_t, -1e9).argmax(-1), np.where(mask, lo_o, -1e9).argmax(-1)
        flips += int((pt != po).sum())
        total += len(pt)
    print(f"   {name}: max |logit diff| {worst:.4f}, answers that flip {flips}/{total} rows, "
          f"shapes {[tuple(b['input_ids'].shape) for b in batches]}")
    return worst, flips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=FINETUNED_DIR, help="laya checkpoint to export (default: fine-tuned)")
    ap.add_argument("--out", default=EXPORT_DIR)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--keep-fp32", action="store_true", help="keep export/laya_fp32.onnx (~800 MB)")
    ap.add_argument("--single-file", action="store_true",
                    help="keep the weights inside laya.onnx instead of laya.onnx.data (uses ~2x the RAM at load)")
    ap.add_argument("--quant", default="weight8", choices=["weight8", "dynamic"],
                    help="weight8: int8 weights, float math (default); dynamic: int8 activations too")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from laya.train import load_checkpoint
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(args.out, exist_ok=True)
    fp32_path = os.path.join(args.out, "laya_fp32.onnx")
    int8_path = os.path.join(args.out, "laya.onnx")

    print(f"[*] Loading {os.path.relpath(args.src)} ...")
    model, tok, cfg = load_checkpoint(args.src)
    model.eval()
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    wrapper = Wrapper(model).eval()
    # nn.TransformerEncoderLayer (the Laya head) otherwise takes a fused fast path in eval mode,
    # aten::_transformer_encoder_layer_fwd, which has no ONNX export
    torch.backends.mha.set_fastpath_enabled(False)

    with open(CAR_QUESTIONS, encoding="utf-8") as f:
        qcfg = json.load(f)
    questions = {"control": qcfg["control"], **{f"p_{g}": s["question"] for g, s in qcfg["payloads"].items()}}
    texts = [l.split("\t")[0] for l in open(EVAL_HELDOUT, encoding="utf-8") if "\t" in l and not l.startswith("#")]
    batches = build_batches(tok, cfg, questions, texts[:6])
    example = tuple(torch.from_numpy(batches[len(questions) + 1][k]) for k in INPUTS)

    print(f"[*] Exporting FP32 ONNX (opset {args.opset}) ...")
    t0 = time.time()
    dyn = {"input_ids": {0: "rows", 1: "seq"}, "attention_mask": {0: "rows", 1: "seq"},
           "marker_pos": {0: "rows", 1: "options"}, "marker_mask": {0: "rows", 1: "options"},
           "qtype": {0: "rows"}, "logits": {0: "rows", 1: "options"}, "act_logits": {0: "rows"}}
    with torch.no_grad():
        torch.onnx.export(wrapper, example, fp32_path, input_names=INPUTS, output_names=OUTPUTS,
                          dynamic_axes=dyn, opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print(f"    {os.path.getsize(fp32_path) / MB:.0f} MB in {time.time() - t0:.0f}s")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    w32, f32 = compare("FP32 vs torch", ort.InferenceSession(fp32_path, so, providers=["CPUExecutionProvider"]),
                       model, batches)
    if f32 or w32 > 1e-2:
        raise SystemExit("[!] FP32 ONNX does not match torch; not quantizing")

    t0 = time.time()
    if args.quant == "dynamic":
        print("[*] Quantizing to INT8 (dynamic: int8 weights and activations) ...")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    else:
        import logging
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
        print("[*] Quantizing weights to 8 bits (MatMulNBits, block 128; embedding int8 per row) ...")
        q = MatMulNBitsQuantizer(fp32_path, bits=8, block_size=128, is_symmetric=True, accuracy_level=None,
                                 op_types_to_quantize=("MatMul",))
        q.process()
        tmp = int8_path + ".matmul.onnx"
        q.model.save_model_to_file(tmp, use_external_data_format=False)
        quantize_dynamic(tmp, int8_path, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Gather"])
        os.remove(tmp)
    print(f"    {os.path.getsize(int8_path) / MB:.0f} MB in {time.time() - t0:.0f}s")
    data_path = int8_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)          # onnx appends to an existing data file
    if not args.single_file:
        print("[*] Moving the weights to laya.onnx.data (memory-mapped at load) ...")
        m = onnx.load(int8_path)
        onnx.save_model(m, int8_path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=os.path.basename(data_path), size_threshold=1024)
        os.chmod(data_path, 0o644)    # onnx creates it owner-only; a web server may run as another user
        del m
    levels = {}
    for name, lvl in (("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC), ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
        so_q = ort.SessionOptions()
        so_q.graph_optimization_level = lvl
        levels[name] = ort.InferenceSession(int8_path, so_q, providers=["CPUExecutionProvider"])
    w8, f8 = compare(f"{args.quant} vs torch", levels["all"], model, batches)
    drift = max(float(np.abs(levels["basic"].run(OUTPUTS, b)[0] - levels["all"].run(OUTPUTS, b)[0])[b["marker_mask"]].max())
                for b in batches)
    print(f"   {args.quant}: max logit change between ORT optimization levels {drift:.4f}"
          + ("  (runtime-dependent: browser / device may answer differently)" if drift > 0.05 else ""))

    # files ONNXAgent reads next to the graph
    for name in ("rl_agent_config.json", "config.json", "finetune_summary.json", "README.md"):
        if os.path.exists(os.path.join(args.src, name)):
            shutil.copy(os.path.join(args.src, name), args.out)
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(os.path.join(args.src, sub), os.path.join(args.out, sub), dirs_exist_ok=True)
    shutil.copy(CAR_QUESTIONS, args.out)
    if not args.keep_fp32:
        os.remove(fp32_path)

    files = [p for p in (int8_path, data_path) if os.path.exists(p)]
    report = {"src": os.path.relpath(args.src), "opset": args.opset, "quant": args.quant,
              "files": [os.path.basename(p) for p in files],
              "onnx_mb": round(sum(os.path.getsize(p) for p in files) / MB, 1),
              "fp32_vs_torch_max_logit_diff": w32, "quant_vs_torch_max_logit_diff": w8, "quant_flips": f8,
              "quant_drift_between_ort_levels": drift}
    with open(os.path.join(args.out, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] {' + '.join(report['files'])} ({args.quant}): {report['onnx_mb']} MB. Bundle in {os.path.relpath(args.out)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
