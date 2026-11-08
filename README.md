# Lyra Car Assistant

**Thai and English voice commands for a car, understood offline by a slimmed-down Laya model.**

Say "เปิดแอร์ 23 องศา", "Open the tailgate" or "เพลงถัดไป". Lyra takes the text from whatever speech recognizer you
use and turns it into a car action — a control key plus a payload: `climate/temperature = 23`, `tailgate = OPEN`,
`media = next`. No cloud call. The model runs on the device, or right in a browser tab.

```
speech -> text -> Laya: "which control?" -> Laya: "which payload?" (or read the number) -> {control, payload} -> car
```

Where it stands: fine-tuned, it gets 96% of controls and 94% of whole commands right on sentences it never saw
(zero-shot it managed 54% and 39%). Torch, ONNX and the browser all give the same answers. Those test sentences
still share a lot of vocabulary with the training data, though, so real recordings from real drivers are the test
that counts — that and latency/RAM on the actual hardware.

## Try it in your browser

**[gh.zugetor.com/Lyra-Car-Assistant](https://gh.zugetor.com/Lyra-Car-Assistant/)** — nothing to install. The
page pulls the model from [Hugging Face](https://huggingface.co/zugetor/Lyra-Car-Assistant-enth) (~218 MB, once;
after that it loads from the browser's cache in under a second) and runs it in the tab.

Or run the same page against your own `export/`:

```bash
.venv/bin/python web_browser/fetch_vendor.py      # once — ONNX Runtime Web into web_browser/vendor/
.venv/bin/python web_browser/serve.py             # then open http://127.0.0.1:8081/
```

The page — *Lyra Car Assistant · Running locally in the browser* — downloads the 212 MB model from the local server
(~2 s) and does everything else in the tab: tokenizing, building the Laya prompt, running the model, deciding.
`serve.py` only hands out files. Type a command, or click one of the examples. They're split into "Unseen sentences"
(`eval_heldout.tsv`, worded differently from anything in training) and "Familiar sentences" (`eval_commands.tsv`,
close to it), and each one turns green or red against its expected answer.

No model yet? Build the whole thing first (below), then come back.

### Publishing

```bash
cd export && hf upload zugetor/Lyra-Car-Assistant-enth .
```

That uploads the bundle and leaves the model card alone: the card lives only on Hugging Face, the code only here.

The demo on GitHub Pages is built by `.github/workflows/pages.yml` on every push to `master` that touches
`web_browser/` (one-time setup: Settings → Pages → Source: GitHub Actions). It runs `web_browser/build_pages.py`,
which copies the page and ONNX Runtime Web into `_site/` and writes a `config.js` pointing at the Hugging Face repo.
The model isn't part of the site. `build_pages.py` pins a Hugging Face commit (`HF_REVISION`), because the browser
caches by URL; after a new upload, put the new commit id there. To try the site the way Pages serves it:

```bash
python3 web_browser/build_pages.py && python3 -m http.server 8082 --bind 127.0.0.1 --directory _site
```

GitHub Pages can't send the COOP/COEP headers ONNX Runtime Web needs for threads, so `web_browser/coi-sw.js`, a
service worker, adds them and the page reloads once on the first visit. Hugging Face sends CORS headers, which is
all the cross-origin fetches need.

## Build it

```bash
./run.sh                      # venv -> download -> prune vocab -> check -> zero-shot eval
./run.sh --sweep              # just print vocab size / model size per --min-count
./run.sh --min-count 100      # smaller model, more rare English words split

.venv/bin/python gen_train_data.py                      # data/train.jsonl + data/val.jsonl
.venv/bin/python step2_finetune.py                      # ~1 h on Apple MPS -> laya_finetuned_enth/
.venv/bin/python eval_laya.py --models pruned finetuned # before vs after
.venv/bin/python step3_export_onnx.py                   # export/laya.onnx + laya.onnx.data (8-bit) + bundle
.venv/bin/python eval_laya.py --models finetuned onnx   # torch vs ONNX
.venv/bin/python web_browser/make_reference.py          # Python answers for the page's ?selftest
.venv/bin/python web_test/server.py                     # http://127.0.0.1:8080 — same page, Python runs the model
```

You need ~3 GB of free disk (the venv with torch alone is ~1.1 GB) and Python 3.11+. `run.sh` uses `python3` —
override with `PYTHON=...` — and rebuilds `.venv` whenever it's missing, broken, or built with a different Python.
One catch: `onnxruntime` has no Python 3.15 wheel yet (Nov 2026), so run step 3 with 3.14.

## What it understands

| control | key | payloads |
| :--- | :--- | :--- |
| Climate mode | `climate/mode` | off, auto |
| Climate temperature | `climate/temperature` | 17–33 (°C), or warmer / cooler when no number |
| Climate fan | `climate/fan_mode` | 0–7, or up / down when no number |
| Windows | `windows_all` | OPEN, CLOSE, STOP |
| Tailgate | `tailgate` | OPEN, CLOSE, STOP |
| Sunroof | `sunroof` | OPEN, CLOSE, STOP |
| Sunshade | `sunshade` | OPEN, CLOSE, STOP |
| Driver / passenger seat ventilation | `seat_vent_driver`, `seat_vent_passenger` | off, low, medium, high |
| Daytime running lights | `drl` | on / off (or 1 / 0) |
| Phone wireless charger | `wireless_charging` | 1 / 0 |
| Media | `media` | play, pause, next, previous, volume_up, volume_down, mute |
| Open app | `open_app` | carplay, android_auto |

Plus everyday car words — CarPlay, Android Auto, Bluetooth, Wi-Fi, battery, charging current, AC / DC charging.
Charge limit, settings, seat memory, seat heating, child lock and speed-limit warning were here once and got cut as
unused. More controls will come.

## Files

| file | what it does |
| :--- | :--- |
| `step0_download.py` | Fetches the Laya model (pinned revision, ~650 MB), wikitext-103 English text (~315 MB) and 500 Thai Wikipedia articles for testing (`--langs en,th,zh` adds Chinese). Writes `cache/manifest.json`. |
| `step1_prune_vocab.py` | Builds the English + Thai vocab (`--langs`), writes a pruned tokenizer and slices the embedding table into `laya_pruned_enth/` — same layout as the Hub repo, so `laya.load("laya_pruned_enth")` just works. |
| `check_step1.py` | Pass/fail checks on the pruned tokenizer and weights, plus RAM and size. Exit code 1 on failure. |
| `eval_laya.py` | Runs any checkpoint (`--models original pruned finetuned onnx`, default the first two) on both eval sets with a question file. Accuracy, confusions, latency, both ask-again gates. Exits 1 if the pruned model answers differently on a command that tokenizes the same. |
| `train_phrases.py` | The phrase bank: object words, verb templates and whole sentences for every control and payload, in Thai and English. **New ways of saying things go here.** |
| `gen_train_data.py` | Expands the phrase bank (politeness words, numbers, spacing and casing), caps each class, drops anything identical to an eval sentence, writes `data/train.jsonl` and `data/val.jsonl`. |
| `step2_finetune.py` | Fine-tunes `laya_pruned_enth/` with `laya.train.finetune` (4 epochs, options shuffled every epoch), calibrates, writes `laya_finetuned_enth/`. |
| `step3_export_onnx.py` | Exports the whole decision model (encoder + head) to ONNX, checks it against torch on real questions of several lengths, stores the weights in 8 bits, and writes the bundle to `export/`. |
| `runtime.py` | `CarAssistant` — the decision logic a car would run. `car.decide("เปิดฝาท้ายหน่อย")["action"]` → `{"tailgate": "OPEN"}`. |
| `web_browser/` | **The main page.** `lyra.js` (tokenizer, Laya prompt, car runtime on ONNX Runtime Web), `index.html`, `config.js` (where the model is), `serve.py`, `fetch_vendor.py` (pinned onnxruntime-web, SHA-256 checked), `make_reference.py`, and for GitHub Pages `build_pages.py` + `coi-sw.js`. |
| `web_test/` | *Lyra Car Assistant · Web test* — the same kind of page, but Python runs the model. `server.py` (standard library, 127.0.0.1 only) + `index.html`. |
| `car_questions.json` | The questions asked at runtime: one `control` choice question, one payload question per payload group, and `label_to_key` (the model sees `aircon`, the car gets `climate/mode`). |
| `questions/` | `v1_controls.json` (11 controls + none) and `v2_media_apps.json` (v1 + media, apps, warmer/cooler — what `car_questions.json` is now). |
| `eval_commands.tsv` | 89 labeled Thai / English commands ("familiar"). Used to pick the questions. |
| `eval_heldout.tsv` | 52 labeled paraphrases ("unseen"). Never used to tune anything. |
| `keep_text.txt` | English text whose tokens survive pruning no matter what — prompt words, control keys, commands, car words. **Add new English words here.** |
| `test_commands.txt` | Thai / English commands per control. Their tokens are kept too, and the check insists each one tokenizes exactly as before. |
| `common.py` | Shared paths, constants, script ranges, text loaders. |

## How it works

### Pruning the vocabulary

Laya has exactly one vocab-sized weight: `encoder.embeddings.tok_embeddings.weight` [256000, 768]. No MLM decoder,
and the head scores options at `<mask>` positions — so cutting the vocab means slicing that table and the tokenizer,
nothing else.

What stays: special and added tokens, the 255 byte-fallback tokens, bare `▁`. Every token containing a Thai
character. Single characters — ASCII, Latin letters (`é ü ñ`), a short symbol list (`° ℃ ’ “ ” … ×`, and the
zero-width space U+200B that Thai text uses as a word break). ASCII tokens that show up at least `--min-count` times
in wikitext-103. Every token produced by `keep_text.txt`, `test_commands.txt` and the question files — as written,
lower-cased, upper-cased, and glued to Thai with no space (`ต่อWiFi`, which tokenizes without the leading `▁`).

Then the closure: every piece a BPE merge needs to build a kept token. Skip it and words quietly fall apart into
bytes.

Everything else goes — Chinese and other CJK (unless `--langs en,th,zh`), Cyrillic, Arabic, Hangul, Devanagari,
accented Latin words, non-English ASCII, code, emoji. Nothing crashes on dropped words; byte fallback still covers
them. Which is exactly why checking for `<unk>` proves nothing. The checks compare token sequences against the
original tokenizer instead.

### Fine-tuning

`gen_train_data.py` expands the phrase bank into 3,708 sentences, half Thai and half English (6,828 questions). Each
control gets several object words — ฝาท้าย / ประตูหลัง / ท้าย / ท้ายรถ / ประตูท้าย / ฝากระโปรงท้าย, the tailgate /
trunk / boot / rear hatch / liftgate — times several verbs, politeness words and numbers. Also warmer/cooler
phrasings, all 7 media actions, both apps, and 300 `none` sentences, some of them car controls we don't support
("เปิดไฟหน้า", "lock the doors"). The model has to learn to say "not mine".

`step2_finetune.py` then runs `laya.train.finetune` from the pruned checkpoint: 4 epochs, 404 updates, effective
batch 64, options shuffled every epoch, 400 items held back for calibration. About 55 minutes on Apple MPS. The
result is the same size as the pruned model, 393 MB in FP16.

### Exporting to ONNX

`step3_export_onnx.py` exports the full `DecisionModel` with the five inputs laya's `ONNXAgent` expects —
`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype` -> `logits`, `act_logits`. By default
(`--quant weight8`) weights are stored as int8 — MatMulNBits, 8 bits, blocks of 128, the embedding table int8 per row
— and the math stays in float. `--quant dynamic` is onnxruntime's `quantize_dynamic`, which also squeezes
activations into int8 at run time. That one is kept only for comparison; see the results.

`export/` ends up holding `laya.onnx` (the graph, ~1 MB), `laya.onnx.data` (the 8-bit weights, ~211 MB), the
tokenizer, `rl_agent_config.json` (calibrated temperatures), `finetune_summary.json` (laya's ask-again thresholds),
`car_questions.json` and `export_report.json`. Keep the two model files side by side: `laya.onnx` points at
`laya.onnx.data` by name. ONNX Runtime maps the weight file instead of copying it, which halves the RAM (see the
results). `--single-file` puts everything back into one `laya.onnx`; `--keep-fp32` keeps the FP32 graph as well.

### Running in the browser

`web_browser/lyra.js` is a JavaScript port of the whole pipeline:

- **Tokenizer** — about 100 lines of Gemma BPE. Added tokens first, `" "` -> `▁`, Metaspace (prepend, split),
  merges by rank, byte fallback. Builds from `tokenizer.json` in ~60 ms.
- **Prompt** — laya's sequential layout, `[CLS] choice question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 …
  [SEP] <command> [SEP]`, same 256-token head budget, same 48-token cap per option.
- **Model** — ONNX Runtime Web 1.30.0 on WASM. Four threads when the page is cross-origin isolated; `serve.py` sends
  the headers that allow it. Any other static server works too, just single-threaded. The page fetches `laya.onnx`
  and `laya.onnx.data` and hands the weights over as `externalData`; a `--single-file` export loads too.
- **Runtime** — the same rules as `runtime.py`.

Want proof it matches Python? Open http://127.0.0.1:8081/web_browser/?selftest. It compares the page against
`reference.json` from `make_reference.py`.

### Deciding, and when to ask again

Every command takes two calls at most. First the control question. Then, for temperature or fan, the number gets
read straight from the text by a rule; anything else gets only that control's payload question.

Then the car decides whether to act or ask the driver again. There are two gates in `runtime.py`:

- **margin** (default) — ask again when the best option beats the runner-up by less than 0.5 (`ASK_AGAIN_MARGIN` in
  `common.py`), for control or payload. In other words, only when the model is genuinely torn.
- **laya** — ask again when laya's confidence falls below the threshold it fitted during calibration (0.98 for the
  control question, 0.995 for 3–5 option payload questions). Available in `runtime.py` and with
  `web_test/server.py --gate laya`.

Why margin wins by default: laya fitted those thresholds on calibration items drawn from the very same templates as
the training data — items the model gets right almost every time — so they came out absurdly strict. After
temperature scaling every confidence lands somewhere between 0.97 and 0.99, and perfectly good answers got re-asked
on noise. "เปิดฝาหลัง" scored 0.9804 against a 0.9816 line.

## Results

### Pruning (English + Thai, `--min-count 20`, macOS, tokenizers 0.22)

| | original | pruned |
| :--- | ---: | ---: |
| vocab | 256,000 | 104,761 |
| BPE merges | 580,604 | 303,214 |
| params | 322 M | 206 M |
| `model.safetensors` (FP16) | 644 MB | 393 MB |
| `tokenizer.json` | 34 MB | 7.0 MB |
| tokenizer RAM, load peak | +405 MB | +194 MB |
| tokenizer RAM, live heap | 77 MB | 31 MB |

Nothing the car cares about changed. All 107 test commands (221 strings counting lower/upper case) tokenize exactly as
before — though those are force-kept, so that mostly proves the merges still work. The independent checks:
11,327 / 11,327 numbers (0–10,000, Thai digits, glued to Thai, decimals, `%`, `°C`, `kW`), 47,622 / 47,622 pure-Thai
runs from Wikipedia, and 99.8% of English word occurrences in wikitext-103 test. What changed there? Rare names and
Spanish words. Every kept embedding row is byte-identical to its source.

Want it smaller? Raise `--min-count` and pay in English:

| `--min-count` | vocab | est. 8-bit | English words that tokenize differently |
| ---: | ---: | ---: | ---: |
| 3 | 123k | ~210 MB | 0.04% |
| **20** (default) | **105k** | **~196 MB** | **0.22%** |
| 100 | 79k | ~178 MB | 1.4% |
| 300 | 56k | ~160 MB | 4.3% |
| 1000 | 34k | ~144 MB | 11.6% |

There's a floor, though. The non-embedding part alone is ~125 M params, about 120 MB in 8 bits. Getting to ~140 MB
would take a vocab under ~30k, and by then most English words split.

### Fine-tuned vs zero-shot (CPU, fp32)

| | pruned, zero-shot | fine-tuned |
| :--- | ---: | ---: |
| familiar sentences: control | 69.7% | 100% |
| familiar sentences: whole command | 62.9% | 100% |
| **unseen sentences: control** | 53.8% | **96.2%** (50 / 52) |
| **unseen sentences: whole command** | 38.5% | **94.2%** (49 / 52) |
| unseen: Thai / English control | 50% / 58% | 96% / 96% |
| latency, runtime path | ~90 ms | ~95 ms |

Three unseen sentences still go wrong:

| command | expected | got |
| :--- | :--- | :--- |
| ช่วยเปิดแอร์ให้หน่อย ร้อนจัง | `climate/mode` auto | `climate/temperature` |
| Blow harder | `climate/fan_mode` up | `media` |
| It's too loud | `media` volume_down | `media` volume_up |

And all three are confident mistakes — the wrong option wins by a margin of 0.98–0.99. The margin gate lets them
through. The laya gate caught the two control misses, but only because their confidence happened to dip a little
(0.975, 0.978), and it re-asked 7 of the 50 right answers along the way:

| (ONNX, Python) | margin gate | laya gate |
| :--- | ---: | ---: |
| unseen commands that ask again (control or payload) | 0 / 52 | 15 / 52 |
| unseen right controls re-asked | 0 / 50 | 7 / 50 |
| unseen wrong controls caught | 0 / 2 | 2 / 2 |
| familiar right controls re-asked | 1 / 89 | 10 / 89 |

So neither gate is a safety net yet. Mistakes like these get fixed with training data, not thresholds.

### ONNX and the browser (Mac)

| | torch FP32 | ONNX FP32 | **ONNX 8-bit weights** (default) | ONNX dynamic INT8 |
| :--- | ---: | ---: | ---: | ---: |
| file | 393 MB (FP16) | 786 MB | **212 MB** | 208 MB |
| max logit difference vs torch | — | 0.0002 | 2.9 | 5.9 |
| logit change between ORT optimization levels | — | 0 | **0.0002** | up to ~9 |
| unseen control / whole command | 96.2% / 94.2% | 96.2% / 94.2% | 96.2% / 94.2% | 96.2% / 94.2% |
| browser vs Python, 141 eval commands | — | — | **141 / 141 same** | 6 differ |
| latency, runtime path, native CPU | ~78 ms | ~121 ms | ~213 ms | ~101 ms |

Dynamic INT8 started out as the default. It lost the job because its answers depend on which runtime you ask.
Quantizing activations on the fly makes the low-ranked logits swing: one option on one control question came out as
-3.3, -9.6 or -4.1 depending on ORT build and optimization level, against -12.2 in FP32. The top answer usually
survived. Margins, confidences and 6 of 141 browser answers didn't.

8-bit weights stay close to FP32 and identical everywhere — at a price. On this Mac they're the slowest of the
four, because MatMulNBits does its math in float. Measure on the device before worrying; if it is too slow,
`accuracy_level=4` (int8 math inside MatMulNBits) is the next thing to try.

RAM, the whole Python runtime (`runtime.py`: tokenizer + session), 8-bit weights, fresh process:

| | weights inside `laya.onnx` | weights in `laya.onnx.data` (default) |
| :--- | ---: | ---: |
| private RAM after load | +640 MB | **+300 MB** |
| private RAM after 24 commands | +660 MB | **+330 MB** |
| resident (RSS) after 24 commands | +830 MB | +620 MB |
| peak RSS of the process | 880 MB | 670 MB |

With the weights inside the graph, ONNX Runtime reads the whole protobuf into memory and then copies every weight
into its own buffers. With a separate weight file it maps the file instead. Speed is the same. Mapped pages count toward RSS once they're touched, but they're clean pages of a file on
disk that the OS can drop and read again. They aren't private memory.

Of the ~300 MB that's left, ~160 MB is torch + transformers, which `laya.load` imports even for the ONNX backend.
The tokenizer takes ~90 MB and the ONNX Runtime session ~50 MB. A device build has no torch, so expect roughly
140 MB private plus the 211 MB mapped file. Converting to the ORT format didn't help on macOS. The real number has
to come from ONNX Runtime Mobile on the real thing.

The browser gains as well. ONNX Runtime Web's WASM memory ends at 269 MB with the separate weight file, against 550 MB
with one file, and WASM memory never shrinks once it has grown.

In the browser, `?selftest` reports 233 / 233 token sequences, 216 / 216 prompts and 141 / 141 answers identical to
Python, with probabilities at most 0.0001 apart. Same 96.2% / 94.2% on the unseen set, at a median of ~455 ms per
command on WASM.

### Take these numbers with salt

The familiar set is nearly a copy of the training data — 100% there means little. The unseen paraphrases never
appear in training verbatim, but the phrase bank teaches their words on purpose (ท้ายรถ, boot, moonroof), and 73
training sentences contain an unseen sentence inside them ("เปิดหลังคาซันรูฟเลย"). The unseen score is optimistic.

Worse: everything here was typed by one person. Real recognizer output has odd spacing, numbers spelled out as words,
and outright mishearings. "เบาแอ" (a typo for "เบาแอร์") currently comes out as `media volume_down`, and even the
correctly spelled "เบาแอร์" lands on `climate/mode off`. Record 100–200 real commands, keep them as a new unseen set,
and only then trust this in a car.

## How we got here (zero-shot experiments)

Before fine-tuning we spent a while on the question wording. The old question files are gone; this is what they
taught us. Numbers marked "with Chinese" were measured before Chinese was removed, on Thai / English / Chinese sets.

- **Fewer options beat more.** With 20 controls the 256-token head budget overflowed and laya cut every option to 12
  tokens. Dropping 6 unused controls (20 -> 14 options) took unseen control accuracy from 45% to 59% (with Chinese).
  Every extra control costs something. Adding many more later means fine-tuning again, or laya's
  `predict_tournament` (options answered in small groups, winners in a final round).
- **Thai hints in the option text help.** Thai went from 22% to 38% on the familiar set (Chinese hints were in there
  too). They lifted the level question but hurt on/off, so on/off stayed English.
- **Asking for the group first hurt.** "Group, then control" scored 36% vs 40% with English options and 35% vs 49%
  with hints (with Chinese). A `settings` group swallowed almost everything, and the second question can't rescue a
  wrong group. `eval_laya.py` still supports the layout (`group` + `controls`), in case it does better after
  fine-tuning.
- **Two extra controls cost little.** Adding `media` and `open_app` barely moved the old intents: 65.5% vs 60.7% on
  familiar, 59.1% vs 61.4% on unseen (with Chinese).
- **Dropping Chinese helped Thai and English a bit.** Control went from 60.7% to 69.7% (familiar) and 51.9% to 53.8%
  (unseen), whole command on unseen from 34.6% to 38.5%, and it saved ~18 MB plus ~22 MB of tokenizer RAM. Oddly,
  the level question needed its Chinese hints even for Thai and English (5/6 -> 1/6 without them). Thai wording
  ("ระดับต่ำ / กลาง / สูง") brought it back to 4/6.
- **Label wording matters, a little.** `aircon` beat `climate/mode` on familiar (control 56% -> 63%, with Chinese)
  and tied on unseen. `ac` was close behind. `qi_charging` lost to `wireless_charging` every time.

## Gotchas

- `AutoModel` / `AutoTokenizer` on the Laya repo root fail. It's a custom `laya` checkpoint (`LayaTypedDecisions`)
  with the tokenizer in `tokenizer/` and the encoder config in `encoder/`.
- Gemma marks word starts with `▁` (U+2581), not a space. Filter on `" "` and you drop almost every English word.
- Prune the tokenizer too, not just the embedding — otherwise token ids run off the end of the new table.
- The decision comes from the Laya head (2 transformer layers + option scorer + act head), which needs
  `input_ids, attention_mask, marker_pos, marker_mask, qtype`. Export only the encoder and you can't decide anything.
- The head is `nn.TransformerEncoderLayer`. In eval mode it takes a fused fast path
  (`aten::_transformer_encoder_layer_fwd`) that ONNX can't export; `step3_export_onnx.py` switches it off with
  `torch.backends.mha.set_fastpath_enabled(False)`.
- Dynamic INT8 gives runtime-dependent answers. Use 8-bit weights.
- The tokenizer splits on added tokens (`<bos>`, `\n`, runs of tabs …) before normalizing, and every piece in between
  gets its own `▁`. A port that skips this goes wrong on text with newlines.
- ONNX Runtime Web `import()`s its `.mjs` from `env.wasm.wasmPaths`. Make that an absolute URL, not `vendor/`.
- Download the bundle with `snapshot_download(..., local_dir=...)`. In the default Hugging Face cache every file is a
  symlink into its own blob folder, and ONNX Runtime refuses `laya.onnx.data` because it "escapes" the folder
  `laya.onnx` resolves to.
- Weights inside the `.onnx` cost ~2x their size in RAM at load. Keep them in a separate file
  (`save_as_external_data=True`). ONNX Runtime Web won't find that file by itself: pass it as `externalData`.

## Next

- [ ] record 100–200 real driver commands through the real speech recognizer, as a new unseen set
- [ ] add the phrasings that fail today — "เบาแอร์ / ลดแอร์ / หรี่แอร์" (decide first: weaker fan or warmer?), common
  misspellings like "แอ", "blow harder", "it's too loud" — then retrain
- [ ] refit the ask-again gate on real commands
- [ ] measure latency and RAM on the target device with ONNX Runtime Mobile
- [ ] port `lyra.js`'s tokenizer and prompt builder to Kotlin / Java for the device

Done so far: vocab pruned with identical answers, questions tuned (v2), Chinese dropped (`--langs en,th,zh` brings it
back), fine-tuned to 96% / 94% on unseen sentences, exported to ONNX with 8-bit weights, running 100% in the browser
with the same answers as Python, a margin gate that stops re-asking right answers, and a public demo: the model on
Hugging Face, the page on GitHub Pages, cached in the browser after the first visit.

## Layout

```
lyra-car-assistant/
  run.sh  requirements.txt  common.py
  step0_download.py  step1_prune_vocab.py  check_step1.py  eval_laya.py
  train_phrases.py  gen_train_data.py  step2_finetune.py  step3_export_onnx.py  runtime.py
  web_browser/           # lyra.js + index.html + config.js + serve.py + fetch_vendor.py + make_reference.py
                         #   + build_pages.py + coi-sw.js (GitHub Pages)
  .github/workflows/     # pages.yml: builds and deploys the GitHub Pages demo
  web_test/              # server.py + index.html
  keep_text.txt  test_commands.txt  car_questions.json  eval_commands.tsv  eval_heldout.tsv
  questions/             # v1_controls.json, v2_media_apps.json
  cache/                 # git-ignored: HF downloads, test text, token counts, manifest.json
  laya_pruned_enth/      # git-ignored: pruned checkpoint
  data/                  # git-ignored: train.jsonl, val.jsonl (re-create with gen_train_data.py)
  laya_finetuned_enth/   # git-ignored: fine-tuned checkpoint + finetune_summary.json
  export/                # git-ignored: laya.onnx + laya.onnx.data (8-bit) + tokenizer + configs — the on-device bundle
  web_browser/vendor/    # git-ignored: onnxruntime-web (fetch_vendor.py); web_browser/reference.json too
  _site/                 # git-ignored: the GitHub Pages build (build_pages.py)
```

The model builds on Laya (Apache-2.0, Convai Innovations); the pruned checkpoint keeps the upstream README.
