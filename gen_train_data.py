#!/usr/bin/env python3
"""
gen_train_data.py  --  build Laya fine-tuning data from train_phrases.py

For every control and payload it expands the phrase bank (each object word x each verb template,
politeness prefixes / suffixes, number values, Thai spacing and English casing variants), dedupes,
and samples up to a cap per (control, payload, language) so no class drowns the others.

Each row asks what the car runtime asks: the control question from car_questions.json, plus the
payload question of that control (none for numbers, which are read by rule, and for `none`):

  {"state": "เปิดฝาท้ายหน่อย", "questions": {"control": {...}, "p_motion": {...}},
   "expected": {"control": "tailgate", "p_motion": "OPEN"}, "meta": {...}}

Leak check: no generated sentence may equal (after lower-casing and dropping spaces / punctuation)
a sentence in eval_commands.tsv or eval_heldout.tsv. Those are dropped and counted. Sentences that
contain an eval sentence (or the reverse) are counted too, so you can see how close training is.

Writes data/train.jsonl (generated) and data/val.jsonl (eval_commands.tsv in the same format).

Usage:
  python gen_train_data.py
  python gen_train_data.py --cap 60 --seed 7
"""
import argparse
import json
import os
import random
import re
from collections import Counter

import train_phrases as P
from common import CAR_QUESTIONS, EVAL_COMMANDS, EVAL_HELDOUT, HERE

DATA_DIR = os.path.join(HERE, "data")
TH_VERBS = ("เปิด", "ปิด", "ลด", "ขึ้น", "ยก", "หยุด", "เลื่อน", "แง้ม", "ดึง", "กาง", "เก็บ", "ตั้ง", "ปรับ", "เพิ่ม",
            "เร่ง", "เล่น", "ข้าม", "เปลี่ยน", "ย้อน", "กลับ", "เบา", "ใช้", "เข้า", "ไป", "เชื่อม", "สตาร์ท", "ทำ")
EN_VERBS = ("open", "close", "shut", "roll", "lower", "raise", "put", "stop", "hold", "freeze", "pop", "lift", "slide",
            "tilt", "pull", "retract", "draw", "turn", "start", "switch", "set", "kill", "make", "change", "adjust",
            "increase", "decrease", "speed", "slow", "enable", "disable", "activate", "deactivate", "play", "resume",
            "pause", "skip", "go", "replay", "mute", "launch", "connect", "show", "warm", "cool", "blow", "continue",
            "unpause", "silence")


# --------------------------------------------------------------------------- variants
def th_variants(base):
    out = set()
    verb = next((v for v in TH_VERBS if base.startswith(v)), None)
    prefixes = P.TH_PREFIX if verb else [""]
    for pre in prefixes:
        for suf in P.TH_SUFFIX:
            out.add(f"{pre}{base}{suf}")
    if verb and not base[len(verb):].startswith(" "):            # ASR sometimes puts a space after the verb
        out.add(f"{verb} {base[len(verb):]}")
    return out


def en_variants(base):
    out = set()
    is_verb = base.split()[0].lower().strip(",") in EN_VERBS
    for pre in (P.EN_PREFIX if is_verb else [""]):
        for suf in P.EN_SUFFIX:
            if "please" in pre and "please" in suf:
                continue
            t = f"{pre}{base}{suf}"
            out.add(t)
            out.add(t[0].upper() + t[1:])
    return out


def variants(lang, base):
    return th_variants(base) if lang == "th" else en_variants(base)


def fill(templates, **slots):
    """Expand templates over every combination of slot values."""
    out = []
    for t in templates:
        combos = [{}]
        for k, vals in slots.items():
            if "{" + k + "}" in t:
                combos = [dict(c, **{k: v}) for c in combos for v in vals]
        out += [t.format(**c) for c in combos]
    return out


# --------------------------------------------------------------------------- phrase bank -> (text, control, payload, lang)
def expand():
    pools = {}  # (control, payload, lang) -> set(text)

    def add(control, payload, lang, bases):
        s = pools.setdefault((control, payload, lang), set())
        for b in bases:
            s |= variants(lang, b)

    for control, spec in P.MOTION.items():
        for lang in ("th", "en"):
            for payload, tmpl in spec[lang].items():
                add(control, payload, lang, fill(tmpl, o=spec[f"{lang}_obj"]))
    for lang in ("th", "en"):
        for payload, bases in P.AIRCON[lang].items():
            add("climate/mode", payload, lang, bases)
        for n in P.TEMP_NUMBER["numbers"]:
            add("climate/temperature", str(n), lang, fill(P.TEMP_NUMBER[lang], n=[n]))
        for payload, bases in P.TEMP_DIR[lang].items():
            add("climate/temperature", payload, lang, bases)
        for n in P.FAN_NUMBER["numbers"]:
            add("climate/fan_mode", str(n), lang, fill(P.FAN_NUMBER[lang], n=[n]))
        for payload, bases in P.FAN_DIR[lang].items():
            add("climate/fan_mode", payload, lang, bases)
        sv = P.SEAT_VENT
        for control, seats in sv["seats"].items():
            for level, words in sv["levels"][lang].items():
                add(control, level, lang, fill(sv[lang]["on"], v=sv["vent"][lang], s=seats[lang], l=words))
            add(control, "off", lang, fill(sv[lang]["off"], v=sv["vent"][lang], s=seats[lang]))
        for control, spec in P.SWITCH.items():
            for payload, tmpl in P.SWITCH_VERBS[lang].items():
                add(control, payload, lang, fill(tmpl, o=spec[f"{lang}_obj"]))
        for payload, bases in P.MEDIA[lang].items():
            add("media", payload, lang, bases)
        for payload, bases in P.APPS[lang].items():
            add("open_app", payload, lang, bases)
        add("none", "-", lang, P.NONE[lang])
    return pools


# --------------------------------------------------------------------------- rows
def norm(t):
    return re.sub(r"[\s.,!?'\"’\-]", "", t.lower())


def load_eval(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            t, c, p = line.rstrip("\n").split("\t")
            rows.append((t, c, p))
    return rows


class RowMaker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.label_of = {v: k for k, v in cfg.get("label_to_key", {}).items()}
        self.group_of = {c: g for g, spec in cfg["payloads"].items() for c in spec["controls"]}

    def __call__(self, text, control, payload, lang):
        qs = {"control": self.cfg["control"]}
        exp = {"control": self.label_of.get(control, control)}
        g = self.group_of.get(control)
        if g and not payload.isdigit() and payload != "-":
            spec = self.cfg["payloads"][g]["question"]
            if payload in spec["criteria"]:
                qs[f"p_{g}"] = spec
                exp[f"p_{g}"] = payload
        return {"state": text, "questions": qs, "expected": exp,
                "meta": {"control": control, "payload": payload, "lang": lang}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=40, help="max sentences per (control, payload, language)")
    ap.add_argument("--number-cap", type=int, default=6, help="max sentences per number value and language")
    ap.add_argument("--none-cap", type=int, default=150, help="max 'none' sentences per language")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cfg = json.load(open(CAR_QUESTIONS, encoding="utf-8"))
    make = RowMaker(cfg)
    evals = load_eval(EVAL_COMMANDS) + load_eval(EVAL_HELDOUT)
    eval_norm = {norm(t) for t, _, _ in evals}
    held_norm = [norm(t) for t, _, _ in load_eval(EVAL_HELDOUT)]

    pools = expand()
    rows, dropped, stats = [], 0, Counter()
    for (control, payload, lang), texts in sorted(pools.items()):
        texts = sorted(texts)
        clean = [t for t in texts if norm(t) not in eval_norm]
        dropped += len(texts) - len(clean)
        rng.shuffle(clean)
        cap = args.none_cap if control == "none" else args.number_cap if payload.isdigit() else args.cap
        for t in clean[:cap]:
            rows.append(make(t, control, payload, lang))
            stats[(control, lang)] += 1
    rng.shuffle(rows)

    near = [(r["state"], h) for r in rows for h in held_norm
            if len(h) >= 4 and (h in norm(r["state"]) or norm(r["state"]) in h)]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(DATA_DIR, "val.jsonl"), "w", encoding="utf-8") as f:
        for t, c, p in load_eval(EVAL_COMMANDS):
            lang = "th" if re.search(r"[฀-๿]", t) else "en"
            f.write(json.dumps(make(t, c, p, lang), ensure_ascii=False) + "\n")

    controls = sorted({c for c, _ in stats})
    print(f"[+] data/train.jsonl: {len(rows):,} rows ({sum(len(r['questions']) for r in rows):,} questions)")
    print(f"    {'control':22} {'th':>5} {'en':>5}")
    for c in controls:
        print(f"    {c:22} {stats[(c, 'th')]:5} {stats[(c, 'en')]:5}")
    print(f"[+] data/val.jsonl: {len(load_eval(EVAL_COMMANDS))} rows (eval_commands.tsv)")
    print(f"[*] leak check: dropped {dropped} generated sentences equal to an eval sentence")
    print(f"[*] {len(near)} training sentences contain a held-out sentence or sit inside one, e.g. "
          + ", ".join(f"{a!r}~{b!r}" for a, b in near[:4]))
    print("[*] samples:")
    for r in rows[:12]:
        print(f"    {r['state']!r:50} -> {r['expected']}")


if __name__ == "__main__":
    main()
