// speech.js -- Thai speech-to-text in the browser: the zipformer-CarCMD-TH transducer on ONNX Runtime Web.
//
// The model (huggingface.co/zugetor/zipformer-CarCMD-TH) is three ONNX graphs, each with its weights in a .data
// file next to it: a Zipformer encoder (4-bit MatMulNBits), a stateless decoder and a joiner. A clip goes
//   16 kHz samples -> trim the silence (trim.js) -> 0.5 s of faint noise in front -> 80-dim log-mel fbank
//   (Kaldi's, as kaldi-native-fbank computes it) -> encoder -> greedy transducer search -> BPE pieces -> text
// and textForLyra() turns that into what Lyra reads: ำ instead of ํา, filler words in front dropped, numbers
// as digits ("ยี่สิบสี่" -> 24), Thai words joined up.
//
// Works in the browser (global `ort`, vendor/ort.wasm.min.js) and in Node (pass `ort` to SpeechToText.create).

import { trimSilence } from "./trim.js";
import { textForLyra } from "./thai_text.js";
export { textForLyra };

export const SAMPLE_RATE = 16000;

// ------------------------------------------------------------------ fbank (Kaldi, no dither, snip_edges=false)
const WIN = 400, SHIFT = 160, NFFT = 512, BINS = 80, LOW = 20, HIGH = SAMPLE_RATE / 2 - 400, PREEMPH = 0.97;
const EPS = 1.1920928955078125e-7;                        // float epsilon: the floor before log
const POVEY = Float64Array.from({ length: WIN }, (_, i) => Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1)), 0.85));
const mel = (f) => 1127 * Math.log(1 + f / 700);
const MELBANK = (() => {                                  // [{start, weights}] over FFT bins 0..NFFT/2-1
  const lo = mel(LOW), hi = mel(HIGH), delta = (hi - lo) / (BINS + 1), width = SAMPLE_RATE / NFFT;
  return Array.from({ length: BINS }, (_, b) => {
    const left = lo + b * delta, center = lo + (b + 1) * delta, right = lo + (b + 2) * delta;
    const w = [];
    let start = -1;
    for (let i = 0; i < NFFT / 2; i++) {
      const m = mel(width * i);
      if (m > left && m < right) {
        if (start < 0) start = i;
        w.push(m <= center ? (m - left) / (center - left) : (right - m) / (right - center));
      }
    }
    return { start, weights: Float64Array.from(w) };
  });
})();
const FFT = (() => {                                      // radix-2 tables for NFFT points
  const bits = Math.log2(NFFT), rev = new Uint16Array(NFFT);
  for (let i = 0; i < NFFT; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = Float64Array.from({ length: NFFT / 2 }, (_, k) => Math.cos((-2 * Math.PI * k) / NFFT));
  const sin = Float64Array.from({ length: NFFT / 2 }, (_, k) => Math.sin((-2 * Math.PI * k) / NFFT));
  return { rev, cos, sin };
})();

function powerSpectrum(frame, re, im) {                   // frame: NFFT real samples -> re[k]^2 + im[k]^2 into re
  for (let i = 0; i < NFFT; i++) { re[FFT.rev[i]] = frame[i]; im[i] = 0; }
  for (let size = 2; size <= NFFT; size *= 2) {
    const half = size / 2, step = NFFT / size;
    for (let s = 0; s < NFFT; s += size) {
      for (let k = 0; k < half; k++) {
        const c = FFT.cos[k * step], d = FFT.sin[k * step], a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
  for (let k = 0; k <= NFFT / 2; k++) re[k] = re[k] * re[k] + im[k] * im[k];
}

/** 16 kHz samples in [-1, 1] -> {data: Float32Array(frames * 80), frames}. */
export function fbank(samples) {
  const n = samples.length, frames = Math.floor((n + SHIFT / 2) / SHIFT);
  const out = new Float32Array(frames * BINS), buf = new Float64Array(NFFT), re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  for (let f = 0; f < frames; f++) {
    const start = f * SHIFT + SHIFT / 2 - WIN / 2;
    let mean = 0;
    for (let i = 0; i < WIN; i++) {
      let s = start + i;
      while (s < 0 || s >= n) s = s < 0 ? -s - 1 : 2 * n - 1 - s;   // reflect at the edges
      buf[i] = samples[s];
      mean += buf[i];
    }
    mean /= WIN;
    for (let i = 0; i < WIN; i++) buf[i] -= mean;
    for (let i = WIN - 1; i > 0; i--) buf[i] -= PREEMPH * buf[i - 1];
    buf[0] -= PREEMPH * buf[0];
    for (let i = 0; i < WIN; i++) buf[i] *= POVEY[i];
    buf.fill(0, WIN);
    powerSpectrum(buf, re, im);
    for (let b = 0; b < BINS; b++) {
      const { start: s0, weights } = MELBANK[b];
      let e = 0;
      for (let j = 0; j < weights.length; j++) e += weights[j] * re[s0 + j];
      out[f * BINS + b] = Math.log(Math.max(e, EPS));
    }
  }
  return { data: out, frames };
}

// ------------------------------------------------------------------ the model
function leadIn(seconds, seed = 1) {                       // faint Gaussian noise; a fixed seed keeps runs repeatable
  let s = seed >>> 0;
  const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
                       t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const out = new Float32Array(Math.round(seconds * SAMPLE_RATE));
  for (let i = 0; i < out.length; i += 2) {
    const r = Math.sqrt(-2 * Math.log(rand() || 1e-12)), a = 2 * Math.PI * rand();
    out[i] = r * Math.cos(a) * 1e-3;
    if (i + 1 < out.length) out[i + 1] = r * Math.sin(a) * 1e-3;
  }
  return out;
}

// ------------------------------------------------------------------ the microphone (browser only)
const TAP = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; } }
registerProcessor("tap", Tap);`;

/** The microphone, opened once and then kept warm: opening it takes long enough (getUserMedia, an AudioContext,
 *  the worklet) that the first word was gone by the time audio flowed. While warm and not recording it keeps
 *  only the last `preRollMs` of audio, so speech that starts right as the button goes down is kept too. After
 *  `idleMs` without a recording it closes again. The browser's echo cancelling, noise suppression and gain
 *  control stay on: in a car they take out the music and the road. */
export class Microphone {
  constructor({ preRollMs = 500, idleMs = 30000 } = {}) {
    Object.assign(this, { preRollMs, idleMs, ctx: null, ring: [], ringLen: 0, rec: null, idle: null });
  }
  get isOpen() { return !!this.ctx; }

  /** Opens the microphone; resolves once audio is actually coming in. */
  async open() {
    if (this.ctx) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const ctx = new AudioContext();
    try {
      await ctx.resume();
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP], { type: "text/javascript" })));
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      throw e;
    }
    const node = new AudioWorkletNode(ctx, "tap"), source = ctx.createMediaStreamSource(stream);
    let flowing;
    const first = new Promise((r) => { flowing = r; });
    node.port.onmessage = (e) => { flowing(); this._chunk(e.data); };
    source.connect(node);
    Object.assign(this, { ctx, stream, node, source, ring: [], ringLen: 0 });
    stream.getAudioTracks()[0].addEventListener("ended", () => this.close());   // unplugged, or taken away
    await first;
    this._armIdle();
  }

  _chunk(c) {
    if (!this.ctx) return;
    if (this.rec) return void this.rec.push(c);
    this.ring.push(c);
    this.ringLen += c.length;
    const keep = (this.ctx.sampleRate * this.preRollMs) / 1000;
    while (this.ring.length > 1 && this.ringLen - this.ring[0].length >= keep) this.ringLen -= this.ring.shift().length;
  }
  _armIdle() { clearTimeout(this.idle); this.idle = setTimeout(() => this.close(), this.idleMs); }

  /** Starts a recording (the microphone must be open), with the pre-roll in front. */
  start() {
    clearTimeout(this.idle);
    this.rec = this.ring;
    this.ring = [];
    this.ringLen = 0;
  }
  /** Keeps listening for `tailMs` (the last syllable often trails off after the release), then resolves to
   *  16 kHz mono samples. The microphone stays warm. */
  async stop({ tailMs = 300 } = {}) {
    await new Promise((r) => setTimeout(r, tailMs));
    const chunks = this.rec || [], rate = this.ctx ? this.ctx.sampleRate : SAMPLE_RATE;
    this.rec = null;
    if (this.ctx) this._armIdle();
    const raw = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const c of chunks) { raw.set(c, at); at += c.length; }
    return resampleTo16k(raw, rate);
  }
  /** Drops a recording without decoding it. */
  cancel() { this.rec = null; if (this.ctx) this._armIdle(); }
  /** Releases the microphone (the browser's "in use" indicator goes off). */
  close() {
    clearTimeout(this.idle);
    if (!this.ctx) return;
    const { ctx, stream, node, source } = this;
    this.ctx = null;
    this.ring = [];
    this.ringLen = 0;
    source.disconnect();
    node.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
  }
}

/** Any rate -> 16 kHz with the browser's own resampler. */
export async function resampleTo16k(x, rate) {
  if (rate === SAMPLE_RATE || !x.length) return x;
  const off = new OfflineAudioContext(1, Math.ceil((x.length * SAMPLE_RATE) / rate), SAMPLE_RATE);
  const buf = off.createBuffer(1, x.length, rate);
  buf.copyToChannel(x, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

/** What the model hears: the speech with its silence trimmed (500 ms kept each side) and 0.5 s of faint noise in
 *  front; speech that starts on the first sample would otherwise often lose its first word. */
export function prepareAudio(samples, { trim = true } = {}) {
  const speech = trim ? trimSilence(samples, { padMs: 500 }).samples : samples;
  if (!speech.length) return { audio: new Float32Array(0), seconds: 0 };
  const lead = leadIn(0.5), audio = new Float32Array(lead.length + speech.length);
  audio.set(lead);
  audio.set(speech, lead.length);
  return { audio, seconds: speech.length / SAMPLE_RATE };
}

export class SpeechToText {
  /** files: {encoder, encoderData, decoder, decoderData, joiner, joinerData} as Uint8Array, tokens as text. */
  static async create(ort, files) {
    const opts = (name, data) => ({ executionProviders: ["wasm"], graphOptimizationLevel: "all",
                                    externalData: [{ path: `${name}.data`, data }] });
    const [encoder, decoder, joiner] = await Promise.all([
      ort.InferenceSession.create(files.encoder, opts("encoder.int4.onnx", files.encoderData)),
      ort.InferenceSession.create(files.decoder, opts("decoder.int8.onnx", files.decoderData)),
      ort.InferenceSession.create(files.joiner, opts("joiner.int8.onnx", files.joinerData))]);
    const pieces = [];
    for (const line of files.tokens.split("\n")) {
      const at = line.lastIndexOf(" ");
      if (at > 0) pieces[Number(line.slice(at + 1))] = line.slice(0, at);
    }
    return new SpeechToText(ort, encoder, decoder, joiner, pieces);
  }

  /** Browser: fetch the six files from `base` (Hugging Face), kept in Cache Storage `cache` when given. */
  static async load({ base, cache = null, onProgress = () => {} }) {
    const names = ["encoder.int4.onnx", "encoder.int4.onnx.data", "decoder.int8.onnx", "decoder.int8.onnx.data",
                   "joiner.int8.onnx", "joiner.int8.onnx.data", "tokens.txt"];
    const store = cache && self.caches ? await caches.open(cache).catch(() => null) : null;
    const opened = await Promise.all(names.map(async (n) => {
      const url = new URL(n, base).href;
      const hit = store && await store.match(url);
      const r = hit || await fetch(url);
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return { url, r, cached: !!hit };
    }));
    const total = opened.reduce((s, o) => s + (Number(o.r.headers.get("Content-Length")) || 0), 0);
    const fromCache = opened.every((o) => o.cached);
    let got = 0;
    const bytes = await Promise.all(opened.map(async (o) => {
      const reader = o.r.body.getReader(), chunks = [];
      let size = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); size += value.length; got += value.length;
        onProgress(total ? got / total : 0, got, fromCache);
      }
      const all = new Uint8Array(size);
      let at = 0;
      for (const c of chunks) { all.set(c, at); at += c.length; }
      if (store && !o.cached) store.put(o.url, new Response(all, { headers: { "Content-Length": String(size) } })).catch(() => {});
      return all;
    }));
    const [encoder, encoderData, decoder, decoderData, joiner, joinerData, tokens] = bytes;
    const stt = await SpeechToText.create(self.ort, { encoder, encoderData, decoder, decoderData, joiner, joinerData,
                                                      tokens: new TextDecoder().decode(tokens) });
    stt.modelBytes = got;
    return stt;
  }

  constructor(ort, encoder, decoder, joiner, pieces) {
    Object.assign(this, { ort, encoder, decoder, joiner, pieces });
    this.blank = 0;
    this.unk = pieces.indexOf("<unk>");
  }

  async #decoderOut(a, b) {
    const y = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(a), BigInt(b)]), [1, 2]);
    return (await this.decoder.run({ y })).decoder_out;
  }

  /** 16 kHz samples -> {raw, text (for Lyra), ms}. `trim: false` skips the silence trimming. */
  async transcribe(samples, { trim = true } = {}) {
    const t0 = performance.now();
    const { audio, seconds } = prepareAudio(samples, { trim });
    if (!seconds) return { raw: "", text: "", ms: { total: 0 }, seconds: 0 };
    const feats = fbank(audio);
    const t1 = performance.now();
    const { Tensor } = this.ort;
    const enc = await this.encoder.run({ x: new Tensor("float32", feats.data, [1, feats.frames, 80]),
                                         x_lens: new Tensor("int64", BigInt64Array.from([BigInt(feats.frames)]), [1]) });
    const t2 = performance.now();
    const T = Number(enc.encoder_out_lens.data[0]), D = enc.encoder_out.dims[2], E = enc.encoder_out.data;
    const hyp = [this.blank, this.blank];
    let dec = await this.#decoderOut(this.blank, this.blank);
    for (let t = 0; t < T; t++) {                          // greedy search, one symbol per frame
      const logit = (await this.joiner.run({ encoder_out: new Tensor("float32", E.subarray(t * D, (t + 1) * D), [1, D]),
                                             decoder_out: dec })).logit.data;
      let best = 0;
      for (let k = 1; k < logit.length; k++) if (logit[k] > logit[best]) best = k;
      if (best !== this.blank && best !== this.unk) {
        hyp.push(best);
        dec = await this.#decoderOut(hyp[hyp.length - 2], best);
      }
    }
    const raw = hyp.slice(2).map((i) => this.pieces[i]).join("").replace(/▁/g, " ").trim();
    const t3 = performance.now();
    return { raw, text: textForLyra(raw), ms: { features: Math.round(t1 - t0), encoder: Math.round(t2 - t1),
                                                search: Math.round(t3 - t2), total: Math.round(t3 - t0) }, seconds };
  }
}
