#!/usr/bin/env python3
"""
eval_laya.py  --  run Laya on labeled car commands, original vs pruned checkpoint

Each command in the eval sets (eval_commands.tsv = tuning set, eval_heldout.tsv = paraphrases
never used to tune questions) gets every question in the question file in one forward pass.
Laya scores each question as its own sequence, so this gives the same answers as asking them
one at a time. Two question layouts are supported:

  flat     "control": one choice question over every control
  grouped  "group": a choice question over groups, plus "controls": {group: choice question}.
           A group that has no entry under "controls" is a control itself (settings, none).
Optional "label_to_key" maps the option label the model sees to the real control key, so the
prompt can say "ac" while the car still gets "climate/mode".

Reported per eval set: control accuracy (overall, per language, confusions), group accuracy,
payload accuracy given the right control, numbers read by a simple rule (not Laya),
end-to-end (control and payload right). Latency is timed on the runtime path: flat = a call for
the control, then one for its payload (skipped when a numeric control has a number in the text);
grouped = one call for the group, then one for that group's control + payload question.

Hard check (exit 1): every command that tokenizes the same in both checkpoints must get the
same answers (probabilities within --tol). Commands whose tokens changed are listed.

Usage:
  python eval_laya.py                                          # car_questions.json, both checkpoints, CPU
  python eval_laya.py --questions questions/v1_controls.json --only pruned --device mps
  python eval_laya.py --models pruned finetuned                # after step2_finetune.py
  python eval_laya.py --models finetuned onnx                   # after step3_export_onnx.py (INT8)
  python eval_laya.py --langs th,en                            # only Thai and English commands
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("USE_TF", "0")  # laya README: TF's runtime can deadlock model construction

from common import (ASK_AGAIN_MARGIN, CACHE_DIR, CAR_QUESTIONS, EVAL_COMMANDS, EVAL_HELDOUT, FINETUNED_DIR,  # noqa: E402
                    PRUNED_DIR, is_cjk,
                    is_thai, load_manifest)

NUMERIC = {"climate/temperature": (17, 33), "climate/fan_mode": (0, 7)}
MAX_WORDS = ("max", "maximum", "highest", "สูงสุด", "แรงสุด")


# --------------------------------------------------------------------------- helpers
def lang_of(text):
    if any(is_thai(c) for c in text):
        return "th"
    if any(is_cjk(c) for c in text):
        return "zh"
    return "en"


def read_number(text, control):
    """Rule-based number for numeric controls: digits (Arabic/Thai), 'max'."""
    t = text.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    m = re.search(r"\d+", t)
    if m:
        return int(m.group(0))
    if any(w in t.lower() for w in MAX_WORDS):
        return NUMERIC[control][1]
    return None


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                text, control, payload = line.rstrip("\n").split("\t")
                rows.append({"text": text, "control": control, "payload": payload})
    return rows


# --------------------------------------------------------------------------- question layout
class Questions:
    def __init__(self, cfg):
        self.grouped = "group" in cfg
        self.payload_of = {}          # control -> payload qid
        self.payload_qs = {}
        # "label_to_key": the option label shown to the model -> the real control key (e.g. "ac" -> "climate/mode").
        # Labels not listed are the key itself. Payload "controls" lists always use real keys.
        self.key_of = cfg.get("label_to_key", {})
        for g, spec in cfg["payloads"].items():
            self.payload_qs[f"p_{g}"] = spec["question"]
            for c in spec["controls"]:
                self.payload_of[c] = f"p_{g}"
        if self.grouped:
            self.group_q = cfg["group"]
            self.control_qs = {f"c_{g}": q for g, q in cfg["controls"].items()}
            self.group_of = {}        # real control key -> group
            for g in cfg["group"]["criteria"]:
                q = cfg["controls"].get(g)
                for c in (q["criteria"] if q else [g]):
                    self.group_of[self.key(c)] = g
            self.all = {"group": self.group_q, **self.control_qs, **self.payload_qs}
        else:
            self.all = {"control": cfg["control"], **self.payload_qs}

    def key(self, label):
        return self.key_of.get(label, label)

    def decide(self, ans):
        """answers -> (group or None, real control key)"""
        if not self.grouped:
            return None, self.key(ans["control"]["choice"])
        g = ans["group"]["choice"]
        q = ans.get(f"c_{g}")
        return g, self.key(q["choice"] if q else g)

    def runtime(self, agent, text):
        """The calls a car would make: the control question, then only that control's payload question
        (skipped when a numeric control already has a number in the text).
        Returns (real control key, payload)."""
        if not self.grouped:
            c = self.key(agent.predict(text, {"control": self.all["control"]})["answers"]["control"]["choice"])
            if c in NUMERIC:
                n = read_number(text, c)
                if n is not None:
                    return c, n
            p = self.payload_of.get(c)
            if not p:
                return c, None
            return c, agent.predict(text, {p: self.payload_qs[p]})["answers"][p]["choice"]
        g = agent.predict(text, {"group": self.group_q})["answers"]["group"]["choice"]
        cq = self.control_qs.get(f"c_{g}")
        if not cq:
            return self.key(g), None
        pq = {self.payload_of[self.key(c)] for c in cq["criteria"] if self.key(c) in self.payload_of}
        ans = agent.predict(text, {"control": cq, **{p: self.payload_qs[p] for p in pq}})["answers"]
        c = self.key(ans["control"]["choice"])
        p = self.payload_of.get(c)
        return c, ans[p]["choice"] if p else None


def head_budget_report(agent, qs):
    from laya.common import build_sequence
    hm = agent.cfg.get("head_max_len", 256)
    for qid, q in qs.items():
        ids, _, stats = build_sequence(agent.tok, "x", agent._to_internal(q), max_len=agent.cfg.get("max_len", 1024),
                                       head_max_len=hm, return_stats=True)
        capped = stats.get("tokens_per_option")
        print(f"   head {qid:16} {stats['options']:2} options, {len(ids) - 3:3} tokens"
              + (f"  OVER BUDGET: options cut to {capped} tokens each" if capped else ""))


# --------------------------------------------------------------------------- eval
def predict_all(agent, rows, qs):
    out = []
    for r in rows:
        res = agent.predict(r["text"], qs)
        out.append({qid: {"choice": a["choice"], "probabilities": a["probabilities"], "confidence": a["confidence"]}
                    for qid, a in res["answers"].items()})
    return out


def score(label, rows, preds, Q, thresholds=None):
    by_lang = defaultdict(lambda: [0, 0])
    grp = [0, 0]
    confusions = Counter()
    conf_right, conf_wrong = [], []
    pay = defaultdict(lambda: [0, 0])
    per_control = defaultdict(lambda: [0, 0])
    e2e = [0, 0]
    gate = {"right": 0, "right_asked": 0, "wrong": 0, "wrong_caught": 0}
    mgate = {"right": 0, "right_asked": 0, "wrong": 0, "wrong_caught": 0}
    for r, p in zip(rows, preds):
        g, c = Q.decide(p)
        ok = c == r["control"]
        lang = lang_of(r["text"])
        by_lang[lang][0] += ok
        by_lang[lang][1] += 1
        if Q.grouped:
            grp[0] += g == Q.group_of[r["control"]]
            grp[1] += 1
        cq = p["control"] if not Q.grouped else (p.get(f"c_{g}") or p["group"])
        (conf_right if ok else conf_wrong).append(cq["confidence"])
        pr = sorted(cq["probabilities"].values(), reverse=True)
        mlow = (pr[0] - (pr[1] if len(pr) > 1 else 0.0)) < ASK_AGAIN_MARGIN
        mgate["right" if ok else "wrong"] += 1
        mgate["right_asked" if ok else "wrong_caught"] += mlow
        if thresholds:
            k = len(cq["probabilities"])
            b = "choice:2" if k == 2 else "choice:3-5" if k <= 5 else "choice:6-10" if k <= 10 else "choice:11+"
            low = cq["confidence"] < thresholds.get(b, 0.0)
            gate["right" if ok else "wrong"] += 1
            gate["right_asked" if ok else "wrong_caught"] += low
        per_control[r["control"]][0] += ok
        per_control[r["control"]][1] += 1
        if not ok:
            confusions[(r["control"], c)] += 1
        pq = Q.payload_of.get(r["control"])
        if r["control"] in NUMERIC and r["payload"].isdigit():
            # a number in the text is read by rule; the direction question is only for commands without one
            pok = read_number(r["text"], r["control"]) == int(r["payload"])
            name = "number (rule)"
        elif pq:
            pok = p[pq]["choice"] == r["payload"]
            name = pq[2:]
        else:
            pok, name = True, None
        if name:
            pay[name][0] += pok
            pay[name][1] += 1
        e2e[0] += ok and pok
        e2e[1] += 1
    n_ok, n = sum(v[0] for v in by_lang.values()), sum(v[1] for v in by_lang.values())
    print(f"   [{label}] control {n_ok}/{n} = {n_ok / n * 100:5.1f}%   "
          + "  ".join(f"{k} {a}/{b} ({a / b * 100:.0f}%)" for k, (a, b) in sorted(by_lang.items()))
          + (f"   | group {grp[0]}/{grp[1]} = {grp[0] / grp[1] * 100:.1f}%" if Q.grouped else ""))
    print("      payload given right control: "
          + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(pay.items())))
    print(f"      end-to-end {e2e[0]}/{e2e[1]} = {e2e[0] / e2e[1] * 100:.1f}%"
          + (f"   | confidence right {statistics.mean(conf_right):.2f}, wrong {statistics.mean(conf_wrong):.2f}"
             if conf_right and conf_wrong else ""))
    print(f"      margin gate (< {ASK_AGAIN_MARGIN}): would re-ask {mgate['right_asked']}/{mgate['right']} right "
          f"controls, catches {mgate['wrong_caught']}/{mgate['wrong']} wrong ones")
    if thresholds:
        print(f"      laya gate (fitted thresholds): would re-ask {gate['right_asked']}/{gate['right']} right "
              f"controls, catches {gate['wrong_caught']}/{gate['wrong']} wrong ones")
    print("      per control: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in per_control.items()))
    if confusions:
        print("      confusions: " + ", ".join(f"{g}->{c} x{k}" for (g, c), k in confusions.most_common(6)))
    return {"control": n_ok / n, "by_lang": {k: a / b for k, (a, b) in by_lang.items()},
            "end_to_end": e2e[0] / e2e[1], "payload": {k: a / b for k, (a, b) in pay.items()},
            "per_control": {k: a / b for k, (a, b) in per_control.items()}}


def time_runtime(agent, rows, Q, n=30, warmup=3):
    for r in rows[:warmup]:
        Q.runtime(agent, r["text"])
    ts = []
    for r in rows[:n]:
        t0 = time.perf_counter()
        Q.runtime(agent, r["text"])
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def same_tokenization(orig_dir, pruned_dir, texts):
    from tokenizers import Tokenizer
    o = Tokenizer.from_file(os.path.join(orig_dir, "tokenizer", "tokenizer.json"))
    p = Tokenizer.from_file(os.path.join(pruned_dir, "tokenizer", "tokenizer.json"))
    return [o.encode(t).tokens == p.encode(t).tokens for t in texts]


def compare(rows, a, b, tol, same_tokens):
    diff, max_dp, moved = [], 0.0, []
    for r, pa, pb, same in zip(rows, a, b, same_tokens):
        if not same:
            moved.append(r["text"])
            continue
        for qid in pa:
            if pa[qid]["choice"] != pb[qid]["choice"]:
                diff.append((r["text"], qid, pa[qid]["choice"], pb[qid]["choice"]))
            for k, v in pa[qid]["probabilities"].items():
                max_dp = max(max_dp, abs(v - pb[qid]["probabilities"][k]))
    ok = not diff and max_dp <= tol
    print(f"\n== original vs pruned: {'PASS' if ok else 'FAIL'}  {sum(same_tokens)}/{len(rows)} commands tokenize "
          f"identically; among them {len(diff)} answers differ, max probability change {max_dp:.4f} (tol {tol})")
    for d in diff[:10]:
        print(f"   DIFF {d[0]!r} {d[1]}: {d[2]} -> {d[3]}")
    if moved:
        print(f"   info  tokens changed by pruning (add the words to keep_text.txt to keep them): {moved}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=CAR_QUESTIONS)
    ap.add_argument("--evals", nargs="*", default=[EVAL_COMMANDS, EVAL_HELDOUT])
    ap.add_argument("--models", nargs="*", default=["original", "pruned"],
                    help="original | pruned | finetuned | onnx | NAME=PATH, PATH a checkpoint dir or an .onnx file "
                         "(default: original pruned)")
    ap.add_argument("--only", help="same as --models with one name")
    ap.add_argument("--device", default="cpu", help="cpu (fp32, default) | mps | cuda")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed probability change original -> pruned")
    ap.add_argument("--langs", default="th,en", help="only eval commands in these languages (default th,en)")
    args = ap.parse_args()

    import laya

    with open(args.questions, encoding="utf-8") as f:
        Q = Questions(json.load(f))
    langs = set(args.langs.split(","))
    sets = {os.path.basename(p): [r for r in load_rows(p) if lang_of(r["text"]) in langs] for p in args.evals}
    rows = [r for rs in sets.values() for r in rs]
    man = load_manifest()
    known = {"original": man["model_dir"], "pruned": PRUNED_DIR, "finetuned": FINETUNED_DIR,
             "onnx": os.path.join(os.path.dirname(PRUNED_DIR), "export", "laya.onnx")}
    models = {}
    for m in ([args.only] if args.only else args.models):
        name, _, path = m.partition("=")
        models[name] = path or known[name]
    print(f"questions: {os.path.relpath(args.questions)} ({'grouped' if Q.grouped else 'flat'}), "
          + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))

    preds, summary = {}, {}
    for name, path in models.items():
        t0 = time.time()
        if path.endswith(".onnx"):          # an exported graph: the bundle folder holds tokenizer + config
            agent = laya.load(os.path.dirname(os.path.abspath(path)), backend="onnx", onnx_path=path)
            path = os.path.dirname(os.path.abspath(path))
        else:
            agent = laya.load(path, device=args.device)
        print(f"\n== {name} ({os.path.relpath(path)}, {args.device}, vocab {len(agent.tok):,}, "
              f"loaded in {time.time() - t0:.1f}s)")
        head_budget_report(agent, Q.all)
        ft = os.path.join(path, "finetune_summary.json")
        thresholds = json.load(open(ft)).get("abstention_thresholds") if os.path.exists(ft) else None
        preds[name] = predict_all(agent, rows, Q.all)
        summary[name], i = {}, 0
        for label, rs in sets.items():
            summary[name][label] = score(label, rs, preds[name][i:i + len(rs)], Q, thresholds)
            i += len(rs)
        summary[name]["runtime_median_ms"] = time_runtime(agent, rows, Q)
        print(f"   runtime path latency: median {summary[name]['runtime_median_ms']:.0f} ms per command "
              "(control call + payload call)")
        del agent

    ok = True
    if "original" in preds and "pruned" in preds:
        same = same_tokenization(models["original"], models["pruned"], [r["text"] for r in rows])
        ok = compare(rows, preds["original"], preds["pruned"], args.tol, same)
    if len(summary) > 1:
        print("\n== summary (control / end-to-end)")
        for name, s in summary.items():
            print(f"   {name:10} " + "  ".join(f"{k}: {v['control'] * 100:5.1f}% / {v['end_to_end'] * 100:5.1f}%"
                                              for k, v in s.items() if isinstance(v, dict)))

    tag = os.path.splitext(os.path.basename(args.questions))[0]
    out = os.path.join(CACHE_DIR, f"eval_laya_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"questions": args.questions, "summary": summary, "rows": rows, "preds": preds}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n[+] Details in {os.path.relpath(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
