// lyra.js -- lyra-car-assistant running 100% in the browser.
//
//   GemmaBPE      the pruned tokenizer (tokenizer.json): Replace " " -> "▁", Metaspace (prepend always,
//                 split), BPE merges by rank, byte fallback. Same ids as the Python `tokenizers` library;
//                 the page's self-test checks that against reference.json.
//   buildSequence laya's sequential prompt layout:
//                 [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] text [SEP]
//   CarAssistant  the same rules as runtime.py: control question, then the number rule or that control's
//                 payload question, then the ask-again gate (margin by default, or laya's thresholds).
//
// Needs ONNX Runtime Web loaded as the global `ort` (vendor/ort.wasm.min.js).

export const META = "▁";
export const NUMERIC = { "climate/temperature": [17, 33], "climate/fan_mode": [0, 7] };   // = eval_laya.py
export const MAX_WORDS = ["max", "maximum", "highest", "สูงสุด", "แรงสุด"];                    // = eval_laya.py
export const ASK_AGAIN_MARGIN = 0.5;                                                         // = common.py
const QTYPES = { choice: 0, score: 1, noul: 2 };

// --------------------------------------------------------------------------- tokenizer
export class GemmaBPE {
  constructor(tokenizerJson, tokenizerConfig) {
    const m = tokenizerJson.model;
    if (m.type !== "BPE") throw new Error("expected a BPE tokenizer, got " + m.type);
    this.vocab = new Map(Object.entries(m.vocab));
    this.idToToken = [];
    for (const [t, i] of this.vocab) this.idToToken[i] = t;
    this.ranks = new Map();
    m.merges.forEach((pair, rank) => {
      const [a, b] = Array.isArray(pair) ? pair : [pair.slice(0, pair.indexOf(" ")), pair.slice(pair.indexOf(" ") + 1)];
      this.ranks.set(a + "\u0000" + b, rank);
    });
    this.byteFallback = !!m.byte_fallback;
    this.unkId = this.vocab.get(m.unk_token);
    const id = (key) => {
      const v = this.vocab.get(tokenizerConfig[key]);
      if (v === undefined) throw new Error(`${key} ${tokenizerConfig[key]} not in vocab`);
      return v;
    };
    this.clsId = id("cls_token");
    this.sepId = id("sep_token");
    this.maskId = id("mask_token");
    this.padId = id("pad_token");
    this.maskToken = tokenizerConfig.mask_token;
    // added tokens (<bos>, <mask>, "\n", "\t\t" ...) are matched on the raw text, longest first, before
    // normalizing; each stretch of text between them is normalized and pre-tokenized on its own
    this.added = new Map(tokenizerJson.added_tokens.map((a) => [a.content, a.id]));
    const esc = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    this.addedRe = new RegExp([...this.added.keys()].sort((a, b) => b.length - a.length).map(esc).join("|"), "gu");
    this.utf8 = new TextEncoder();
    this.cache = new Map();
  }

  get size() { return this.vocab.size; }

  // normalizer Replace(" " -> "▁") + Metaspace(prepend_scheme "always", split: merged with next)
  words(text) {
    let s = text.replaceAll(" ", META);
    if (!s.startsWith(META)) s = META + s;
    const out = [];
    let cur = "";
    for (const ch of s) {
      if (ch === META && cur !== "") { out.push(cur); cur = ""; }
      cur += ch;
    }
    if (cur) out.push(cur);
    return out;
  }

  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let syms = [];
    for (const ch of word) {
      if (this.vocab.has(ch)) { syms.push(ch); continue; }
      const bytes = [...this.utf8.encode(ch)].map((b) => "<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">");
      if (this.byteFallback && bytes.every((b) => this.vocab.has(b))) syms.push(...bytes);
      else syms.push(null);                                           // unknown
    }
    while (syms.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        if (syms[i] === null || syms[i + 1] === null) continue;
        const r = this.ranks.get(syms[i] + "\u0000" + syms[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      syms.splice(at, 2, syms[at] + syms[at + 1]);
    }
    // fuse_unk: consecutive unknowns collapse into one <unk>
    const ids = [];
    for (const s of syms) {
      const v = s === null ? this.unkId : this.vocab.get(s);
      if (v === this.unkId && ids[ids.length - 1] === this.unkId) continue;
      ids.push(v);
    }
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }

  encodePlain(text) {
    const ids = [];
    for (const w of this.words(text)) ids.push(...this.bpe(w));
    return ids;
  }

  encode(text) {
    if (!text) return [];
    const ids = [];
    let last = 0;
    for (const m of text.matchAll(this.addedRe)) {
      if (m.index > last) ids.push(...this.encodePlain(text.slice(last, m.index)));
      ids.push(this.added.get(m[0]));
      last = m.index + m[0].length;
    }
    if (last < text.length) ids.push(...this.encodePlain(text.slice(last)));
    return ids;
  }

  tokens(text) { return this.encode(text).map((i) => this.idToToken[i]); }
}

// --------------------------------------------------------------------------- laya prompt
export function renderOptions(q) {
  return Object.entries(q.criteria).map(([k, v]) => (v === null || v === undefined || v === "" ? String(k)
    : `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`));
}

export function buildSequence(tok, stateIds, q, maxLen = 1024, headMaxLen = 256) {
  if (q.type !== "choice") throw new Error("only choice questions are ported");
  const opts = renderOptions(q);
  const ins = String(q.instructions).replaceAll(tok.maskToken, " ");
  let head = tok.encode(`${q.type} question: ${ins}`);
  let optIds = opts.map((o) => [tok.maskId, ...tok.encode(" " + o.replaceAll(tok.maskToken, " ")).slice(0, 48)]);
  const total = () => optIds.reduce((n, o) => n + o.length, 0);
  let budget = headMaxLen - total();
  if (budget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    budget = headMaxLen - total();
  }
  head = head.slice(0, Math.max(8, budget));
  const ids = [tok.clsId, ...head, tok.sepId];
  const markers = [];
  for (const o of optIds) { markers.push(ids.length); ids.push(...o); }
  ids.push(tok.sepId);
  const room = Math.max(0, maxLen - ids.length - 1);
  const seq = [...ids, ...stateIds.slice(0, room), tok.sepId].slice(0, maxLen);
  return { ids: seq, markers: markers.filter((m) => m < maxLen) };
}

// --------------------------------------------------------------------------- helpers
const round4 = (x) => Math.round(x * 1e4) / 1e4;
export function optionBucket(k) { return k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"; }

export function readNumber(text, control) {
  const t = text.replace(/[๐-๙]/g, (d) => String(d.charCodeAt(0) - 0x0e50));
  const m = t.match(/\d+/);
  if (m) return parseInt(m[0], 10);
  const low = t.toLowerCase();
  if (MAX_WORDS.some((w) => low.includes(w))) return NUMERIC[control][1];
  return null;
}

// --------------------------------------------------------------------------- the car runtime
export class CarAssistant {
  // base: folder or URL with the export/ bundle, e.g. a Hugging Face ".../resolve/<revision>/".
  // cache: a Cache Storage name. Then every file is downloaded once and read from the browser's cache on later
  // visits, so `base` must point at files that never change (a pinned revision, not "main"). Cached files from
  // any other base are deleted.
  static async load({ base = "../export/", onProgress = () => {}, gate = "margin", cache = null } = {}) {
    const baseUrl = new URL(base, location.href).href;
    const store = cache && self.caches ? await caches.open(cache).catch(() => null) : null;
    if (store) for (const req of await store.keys()) if (!req.url.startsWith(baseUrl)) await store.delete(req);
    const open = async (p) => {
      const url = baseUrl + p;
      const hit = store && await store.match(url);
      return hit ? { url, r: hit, cached: true } : { url, r: await fetch(url), cached: false };
    };
    const read = async ({ url, r, cached }, onChunk = () => {}) => {
      const reader = r.body.getReader();
      const chunks = [];
      let size = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        size += value.length;
        onChunk(value.length);
      }
      const bytes = new Uint8Array(size);
      let off = 0;
      for (const c of chunks) { bytes.set(c, off); off += c.length; }
      if (store && !cached) {
        store.put(url, new Response(bytes, { headers: { "Content-Length": String(size) } })).catch(() => {});
      }
      return bytes;
    };
    const getJson = async (p, optional = false) => {
      const o = await open(p);
      if (!o.r.ok) { if (optional) return null; throw new Error(`${o.url}: HTTP ${o.r.status}`); }
      return JSON.parse(new TextDecoder().decode(await read(o)));
    };
    onProgress("tokenizer");
    const [tj, tcfg, qcfg, rl, summary] = await Promise.all([
      getJson("tokenizer/tokenizer.json"), getJson("tokenizer/tokenizer_config.json"), getJson("car_questions.json"),
      getJson("rl_agent_config.json"), getJson("finetune_summary.json", true)]);
    const tok = new GemmaBPE(tj, tcfg);

    // laya.onnx is the graph, laya.onnx.data holds the weights. An export made with --single-file has no
    // .data file; then laya.onnx carries everything and the 404 is expected.
    onProgress("model", 0);
    const [og, od] = await Promise.all([open("laya.onnx"), open("laya.onnx.data")]);
    if (!og.r.ok) throw new Error(`${og.url}: HTTP ${og.r.status}`);
    const parts = od.r.ok ? [og, od] : [og];
    const total = parts.reduce((n, o) => n + (Number(o.r.headers.get("Content-Length")) || 0), 0);
    const fromCache = parts.every((o) => o.cached);
    let got = 0;
    const tick = (n) => { got += n; onProgress("model", total ? got / total : 0, got, fromCache); };
    const graph = await read(og, tick);
    const data = od.r.ok ? await read(od, tick) : null;

    onProgress("session");
    const t0 = performance.now();
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    if (data) opts.externalData = [{ path: "laya.onnx.data", data }];
    const session = await ort.InferenceSession.create(graph, opts);
    return new CarAssistant({ tok, qcfg, rl, summary, session, modelBytes: got, gate,
                              sessionMs: performance.now() - t0 });
  }

  constructor({ tok, qcfg, rl, summary, session, modelBytes, gate, sessionMs }) {
    Object.assign(this, { tok, qcfg, session, modelBytes, gate, sessionMs });
    this.maxLen = rl.max_len || 1024;
    this.headMaxLen = rl.head_max_len || 256;
    this.temperature = rl.temperature || [1, 1, 1];
    this.temperatureByOptions = rl.temperature_by_options || {};
    this.thresholds = (summary && summary.abstention_thresholds) || {};
    this.labelToKey = qcfg.label_to_key || {};
    this.payloadOf = {};
    this.payloadQs = {};
    for (const [g, spec] of Object.entries(qcfg.payloads)) {
      this.payloadQs["p_" + g] = spec.question;
      for (const c of spec.controls) this.payloadOf[c] = "p_" + g;
    }
    this.margin = ASK_AGAIN_MARGIN;
  }

  key(label) { return this.labelToKey[label] || label; }

  async ask(text, q) {
    const t0 = performance.now();
    const stateIds = this.tok.encode(text.replaceAll(this.tok.maskToken, " "));
    const { ids, markers } = buildSequence(this.tok, stateIds, q, this.maxLen, this.headMaxLen);
    const L = ids.length, k = markers.length, kc = Math.max(k, 2);
    const big = (xs) => BigInt64Array.from(xs, (x) => BigInt(x));
    const feeds = {
      input_ids: new ort.Tensor("int64", big(ids), [1, L]),
      attention_mask: new ort.Tensor("int64", big(new Array(L).fill(1)), [1, L]),
      marker_pos: new ort.Tensor("int64", big([...markers, ...new Array(kc - k).fill(0)]), [1, kc]),
      marker_mask: new ort.Tensor("bool", Uint8Array.from([...new Array(k).fill(1), ...new Array(kc - k).fill(0)]), [1, kc]),
      qtype: new ort.Tensor("int64", big([QTYPES[q.type]]), [1]),
    };
    const out = await this.session.run(feeds);
    const logits = Array.from(out.logits.data).slice(0, k);
    const qt = QTYPES[q.type];
    const tScale = this.temperatureByOptions[`${q.type}:${optionBucket(k)}`] ?? this.temperature[qt];
    const z = logits.map((x) => x / tScale);
    const zmax = Math.max(...z);
    const e = z.map((x) => Math.exp(x - zmax));
    const s = e.reduce((a, b) => a + b, 0);
    const p = e.map((x) => x / s);
    const ent = -p.reduce((a, x) => a + x * Math.log(Math.min(1, Math.max(1e-12, x))), 0);
    const confidence = round4(k < 2 ? 1 : Math.min(1, Math.max(0, 1 - ent / Math.log(k))));
    const keys = Object.keys(q.criteria);
    const probs = keys.map((kk, i) => [kk, round4(p[i])]);
    const best = p.indexOf(Math.max(...p));
    const sorted = [...probs].sort((a, b) => b[1] - a[1]);
    const margin = sorted[0][1] - (sorted.length > 1 ? sorted[1][1] : 0);
    const thr = this.thresholds[`choice:${optionBucket(k)}`];
    const lowLaya = thr !== undefined && confidence < thr;
    const lowMargin = margin < this.margin;
    return { choice: keys[best], confidence, margin, threshold: thr ?? null, top: sorted.slice(0, 3),
             probabilities: Object.fromEntries(probs), lowLaya, lowMargin,
             low: this.gate === "margin" ? lowMargin : lowLaya, ms: performance.now() - t0, ids, markers };
  }

  async decide(text) {
    text = text.trim();
    const t0 = performance.now();
    const c = await this.ask(text, this.qcfg.control);
    const control = this.key(c.choice);
    let payload = null, source = null, p = null;
    if (control in NUMERIC) {
      const n = readNumber(text, control);
      if (n !== null) { payload = n; source = "number read by rule"; }
    }
    if (payload === null && control in this.payloadOf) {
      const qid = this.payloadOf[control];
      p = await this.ask(text, this.payloadQs[qid]);
      payload = p.choice;
      source = `Laya question ${qid}`;
    }
    const why = (q, name) => !q || !q.low ? null : this.gate === "margin"
      ? `${name}: best option leads by ${q.margin.toFixed(3)} < ${this.margin}`
      : `${name}: confidence ${q.confidence.toFixed(4)} < threshold ${q.threshold}`;
    const reasons = [why(c, "control"), why(p, "payload")].filter(Boolean);
    return {
      text, action: control === "none" ? null : { [control]: payload },
      ask_again: reasons.length > 0, ask_again_reasons: reasons, gate: this.gate,
      laya_gate_would_ask: c.lowLaya || !!(p && p.lowLaya), margin_gate_would_ask: c.lowMargin || !!(p && p.lowMargin),
      control, control_label: c.choice, control_confidence: c.confidence, control_threshold: c.threshold,
      control_margin: round4(c.margin), control_top: c.top.map(([k, v]) => [this.key(k), v]),
      control_probabilities: c.probabilities,
      payload, payload_source: source,
      payload_confidence: p ? p.confidence : null, payload_threshold: p ? p.threshold : null,
      payload_margin: p ? round4(p.margin) : null, payload_top: p ? p.top : null,
      payload_probabilities: p ? p.probabilities : null,
      ms: { control: round1(c.ms), payload: p ? round1(p.ms) : 0, total: round1(performance.now() - t0) },
      tokens: this.tok.tokens(text),
    };
  }
}

const round1 = (x) => Math.round(x * 10) / 10;
