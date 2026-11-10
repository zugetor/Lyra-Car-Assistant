// trim.js -- cut the silence around (and optionally inside) a voice command, before speech-to-text.
//
// Energy-based, no model: 20 ms frames, the noise floor is the 10th percentile of frame energy, a frame is
// speech when it is `marginDb` above that floor (and above `minDb`). Runs shorter than `minSpeechMs` are
// treated as clicks. From there the region grows outwards while frames stay `softDb` above the floor, with
// gaps up to `bridgeMs`, so softer words at the start or end of a command stay in (people often trail off,
// and in a noisy room the last words can sit only a few dB above the floor). The kept region then gets
// `padMs` of context on both sides so word edges survive.
// With `squeezeGapsMs`, pauses inside the command longer than that are shortened to that length.
//
// Speech models tend to invent words on long silence; trimming it also makes the clip shorter to decode.
// Works in the browser and in Node (plain ESM, no dependencies).

export const SAMPLE_RATE = 16000;

export function trimSilence(samples, {
  sampleRate = SAMPLE_RATE,
  frameMs = 20,
  padMs = 200,
  marginDb = 12,
  softDb = 6,
  bridgeMs = 300,
  minDb = -55,
  minSpeechMs = 60,
  squeezeGapsMs = 0,
} = {}) {
  const frame = Math.max(1, Math.round((sampleRate * frameMs) / 1000));
  const n = Math.floor(samples.length / frame);
  const none = { samples, start: 0, end: samples.length, removedMs: 0, speech: false, floorDb: null, thresholdDb: null };
  if (n < 3) return none;

  const db = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * frame, e = j + frame; j < e; j++) s += samples[j] * samples[j];
    db[i] = 10 * Math.log10(s / frame + 1e-12);
  }
  const sorted = Float32Array.from(db).sort();
  const floorDb = sorted[Math.floor(n * 0.1)];
  const peakDb = sorted[n - 1];
  const thresholdDb = Math.max(floorDb + marginDb, minDb);
  if (peakDb < thresholdDb) return { ...none, floorDb, thresholdDb };   // nothing stands out: leave it alone

  // speech frames, dropping runs too short to be speech
  const minRun = Math.max(1, Math.round(minSpeechMs / frameMs));
  const speech = new Uint8Array(n);
  for (let i = 0; i < n; ) {
    if (db[i] <= thresholdDb) { i++; continue; }
    let j = i;
    while (j < n && db[j] > thresholdDb) j++;
    if (j - i >= minRun) speech.fill(1, i, j);
    i = j;
  }
  let first = speech.indexOf(1), last = speech.lastIndexOf(1);
  if (first < 0) return { ...none, floorDb, thresholdDb };

  // grow outwards over softer speech: keep going while a frame above floor + softDb turns up within bridgeMs
  const soft = Math.max(floorDb + softDb, minDb);
  const bridge = Math.max(1, Math.round(bridgeMs / frameMs));
  const grow = (from, step) => {
    let edge = from;
    for (let j = from + step, gap = 0; j >= 0 && j < n; j += step) {
      if (db[j] > soft) { edge = j; gap = 0; } else if (++gap > bridge) break;
    }
    return edge;
  };
  first = grow(first, -1);
  last = grow(last, +1);

  const pad = Math.round((sampleRate * padMs) / 1000);
  const start = Math.max(0, first * frame - pad);
  const end = Math.min(samples.length, (last + 1) * frame + pad);

  // optional: shorten long pauses inside the command
  const pieces = [];
  if (squeezeGapsMs > 0) {
    const keep = Math.round((sampleRate * squeezeGapsMs) / 1000);
    let segStart = start;
    for (let i = first; i <= last; ) {
      if (speech[i]) { i++; continue; }
      let j = i;
      while (j <= last && !speech[j]) j++;
      const gapA = i * frame, gapB = j * frame;
      if (gapB - gapA > keep) {
        pieces.push([segStart, gapA + Math.floor(keep / 2)]);
        segStart = gapB - Math.ceil(keep / 2);
      }
      i = j;
    }
    pieces.push([segStart, end]);
  } else {
    pieces.push([start, end]);
  }

  const total = pieces.reduce((s, [a, b]) => s + (b - a), 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const [a, b] of pieces) { out.set(samples.subarray(a, b), off); off += b - a; }
  return {
    samples: out, start, end, speech: true, floorDb, thresholdDb,
    removedMs: Math.round(((samples.length - total) / sampleRate) * 1000),
  };
}
