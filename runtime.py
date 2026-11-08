"""
runtime.py  --  the decision logic a car runs, on the INT8 ONNX bundle (export/)

One command in, one car action out:

  1. ask the control question                      -> e.g. tailgate (label shown to the model may differ)
  2. numeric controls: read the number by rule      -> climate/temperature = 23
     otherwise ask only that control's payload       -> tailgate = OPEN
  3. decide whether to act or ask the driver again (the "gate"):
       margin (default)  ask again when the best option leads the runner-up by less than MARGIN,
                         i.e. when the model is torn between two answers
       laya              ask again when laya's confidence is below the threshold it fitted at
                         calibration time (finetune_summary.json)

  from runtime import CarAssistant
  car = CarAssistant()                 # export/laya.onnx, margin gate
  car.decide("เปิดฝาท้ายหน่อย")["action"]  # {"tailgate": "OPEN"}

Why margin by default: laya's thresholds were fitted on calibration items drawn from the same templates
as the training data, which the model gets ~100% right, so they came out very strict (0.98 for the
control question). After temperature scaling every confidence sits between ~0.97 and ~0.99, and right
answers fell below the line on noise: 15 of 52 held-out commands asked again. The margin gate asks
again only when the model hesitates between two options (1 of 141 eval commands). Neither gate catches
the current mistakes, which the model makes confidently; those need training data (train_phrases.py).

web_test/server.py serves this over HTTP; eval_laya.py measures both gates in bulk.
"""
import json
import os
import time

os.environ.setdefault("USE_TF", "0")

from common import ASK_AGAIN_MARGIN, CAR_QUESTIONS, FINETUNED_DIR, HERE  # noqa: E402
from eval_laya import NUMERIC, Questions, read_number  # noqa: E402

EXPORT_DIR = os.path.join(HERE, "export")
MARGIN = ASK_AGAIN_MARGIN   # ask again when p(best) - p(runner-up) < MARGIN, control and payload question
GATES = ("margin", "laya")


def option_bucket(k):
    return "choice:2" if k == 2 else "choice:3-5" if k <= 5 else "choice:6-10" if k <= 10 else "choice:11+"


def margin_of(probabilities):
    p = sorted(probabilities.values(), reverse=True)
    return p[0] - (p[1] if len(p) > 1 else 0.0)


class CarAssistant:
    def __init__(self, path=os.path.join(EXPORT_DIR, "laya.onnx"), questions=None, gate="margin", margin=MARGIN):
        """`path` is an .onnx file inside a bundle folder (default export/laya.onnx) or a checkpoint folder."""
        import laya

        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}")
        self.gate, self.margin = gate, margin
        t0 = time.perf_counter()
        if path.endswith(".onnx"):
            self.folder = os.path.dirname(os.path.abspath(path))
            self.agent = laya.load(self.folder, backend="onnx", onnx_path=path)
            self.backend = "ONNX Runtime, 8-bit" if "fp32" not in os.path.basename(path) else "ONNX Runtime, FP32"
            self.model_mb = sum(os.path.getsize(f) for f in (path, path + ".data") if os.path.exists(f)) / 2**20
        else:
            self.folder = path
            self.agent = laya.load(path, device="cpu")
            self.backend = "torch, FP32"
            self.model_mb = os.path.getsize(os.path.join(path, "model.safetensors")) / 2**20
        self.load_s = time.perf_counter() - t0

        qpath = questions or next((p for p in (os.path.join(self.folder, "car_questions.json"), CAR_QUESTIONS)
                                   if os.path.exists(p)))
        with open(qpath, encoding="utf-8") as f:
            self.qcfg = json.load(f)
        self.Q = Questions(self.qcfg)
        summary = os.path.join(self.folder, "finetune_summary.json")
        self.thresholds = (json.load(open(summary)).get("abstention_thresholds", {})
                           if os.path.exists(summary) else {})
        if gate == "laya" and not self.thresholds:
            raise ValueError("gate='laya' needs finetune_summary.json with abstention_thresholds")

    # ------------------------------------------------------------------ helpers
    def _ask(self, text, qid, q):
        t0 = time.perf_counter()
        a = self.agent.predict(text, {qid: q})["answers"][qid]
        ms = (time.perf_counter() - t0) * 1000
        probs = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
        thr = self.thresholds.get(option_bucket(len(probs)))
        margin = margin_of(a["probabilities"])
        low_laya = thr is not None and a["confidence"] < thr
        low_margin = margin < self.margin
        return {"choice": a["choice"], "confidence": a["confidence"], "threshold": thr, "margin": margin,
                "low_laya": low_laya, "low_margin": low_margin,
                "low": low_margin if self.gate == "margin" else low_laya, "top": probs[:3], "ms": ms}

    # ------------------------------------------------------------------ main entry
    def decide(self, text):
        text = text.strip()
        t0 = time.perf_counter()
        c = self._ask(text, "control", self.Q.all["control"])
        control = self.Q.key(c["choice"])
        c["top"] = [(self.Q.key(k), round(p, 4)) for k, p in c["top"]]

        payload, source, p = None, None, None
        if control in NUMERIC:
            n = read_number(text, control)
            if n is not None:
                payload, source = n, "number read by rule"
        if payload is None and control in self.Q.payload_of:
            qid = self.Q.payload_of[control]
            p = self._ask(text, qid, self.Q.payload_qs[qid])
            payload, source = p["choice"], f"Laya question {qid}"

        def why(q, name):
            if not q or not q["low"]:
                return None
            if self.gate == "margin":
                return f"{name}: best option leads by {q['margin']:.3f} < {self.margin}"
            return f"{name}: confidence {q['confidence']:.4f} < threshold {q['threshold']}"

        reasons = [r for r in (why(c, "control"), why(p, "payload")) if r]
        action = None if control == "none" else {control: payload}
        return {
            "text": text,
            "action": action,
            "ask_again": bool(reasons), "ask_again_reasons": reasons, "gate": self.gate,
            "laya_gate_would_ask": c["low_laya"] or bool(p and p["low_laya"]),
            "margin_gate_would_ask": c["low_margin"] or bool(p and p["low_margin"]),
            "control": control, "control_label": c["choice"],
            "control_confidence": c["confidence"], "control_threshold": c["threshold"],
            "control_margin": round(c["margin"], 4), "control_top": c["top"],
            "payload": payload, "payload_source": source,
            "payload_confidence": p and p["confidence"], "payload_threshold": p and p["threshold"],
            "payload_margin": p and round(p["margin"], 4),
            "payload_top": p and [(k, round(v, 4)) for k, v in p["top"]],
            "ms": {"control": round(c["ms"], 1), "payload": round(p["ms"], 1) if p else 0.0,
                   "total": round((time.perf_counter() - t0) * 1000, 1)},
            "tokens": self.agent.tok.tokenize(text),
        }

    def info(self):
        return {"backend": self.backend, "folder": os.path.relpath(self.folder, HERE), "model_mb": round(self.model_mb, 1),
                "vocab": len(self.agent.tok), "load_s": round(self.load_s, 2),
                "controls": [self.Q.key(k) for k in self.Q.all["control"]["criteria"]],
                "gate": self.gate, "margin": self.margin, "thresholds": self.thresholds}


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    car = CarAssistant(FINETUNED_DIR if "--torch" in sys.argv else os.path.join(EXPORT_DIR, "laya.onnx"),
                       gate="laya" if "--laya-gate" in sys.argv else "margin")
    for t in args or ["เปิดฝาท้ายหน่อย", "เปิดฝาหลัง", "I'm cold", "ข้ามเพลงนี้"]:
        r = car.decide(t)
        print(f"{t!r:30} -> {r['action']}  margin {r['control_margin']:.3f}  conf {r['control_confidence']:.4f}  "
              f"{'ASK AGAIN ' if r['ask_again'] else 'ACT '}{r['ms']['total']:.0f} ms")
