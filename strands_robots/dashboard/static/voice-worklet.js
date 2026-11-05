/* The voice panel's microphone worklet: resamples to the model's rate and posts 20 ms PCM frames (panels/voice.js). */
class Down extends AudioWorkletProcessor {
  constructor(o) { super(); this.ratio = sampleRate / o.processorOptions.rate; this.acc = []; this.pos = 0;
    this.frame = Math.round(o.processorOptions.rate / 50); this.buf = new Int16Array(this.frame); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
    let sum = 0;
    for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
    // linear-interpolated resample from the context rate to the model's rate
    for (; this.pos < ch.length; this.pos += this.ratio) {
      const i = Math.floor(this.pos), f = this.pos - i, a = ch[i], b = i + 1 < ch.length ? ch[i + 1] : a;
      const v = Math.max(-1, Math.min(1, a + (b - a) * f));
      this.buf[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      if (this.n === this.frame) { this.port.postMessage({ pcm: this.buf.buffer, rms: Math.sqrt(sum / ch.length) }, [this.buf.buffer]); this.buf = new Int16Array(this.frame); this.n = 0; }
    }
    this.pos -= ch.length;
    return true;
  }
}
registerProcessor("strands-down", Down);
