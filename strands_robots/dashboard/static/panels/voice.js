/* Voice: the microphone to WS /ws/voice (a server-side strands BidiAgent holding the chat agent's tools) and its
 * spoken answer back. Transcripts and tool calls land in the chat log, the same rows the typed chat writes.
 *
 *   mic button  start / end the conversation      mute   stop sending the microphone (the socket stays open)
 *   meter       the microphone level being sent
 *
 * The page sends PCM16 mono at the rate the server's "ready" names (24 kHz OpenAI, 16 kHz Nova Sonic) and plays the
 * PCM16 it receives at the output rate. A barge-in ("interrupt") drops whatever is still queued to play.
 * Registers through window.SR.registerSlot("chat-input") when the shell provides it, else mounts as a row above #chatform.
 */

const $ = (s) => document.querySelector(s);
const events = (window.__voiceEvents = []);

const WORKLET = `
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
registerProcessor("strands-down", Down);`;

const MIC = '<svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const wsUrl = (path) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;

function row(cls, text = "") {
  const log = $("#log");
  const el = document.createElement("div");
  el.className = `msg ${cls} voice`;
  el.textContent = text;
  if (log) { log.appendChild(el); log.scrollTop = 1e9; }
  return el;
}

function toolRow(prefix, name, body) {
  const d = row("tool");
  d.innerHTML = `<details><summary>${prefix} ${esc(name)}</summary><pre></pre></details>`;
  d.querySelector("pre").textContent = body;
}

class Voice {
  constructor(ui) { this.ui = ui; this.ws = null; this.muted = false; this.lines = new Map(); this.playAt = 0; this.sources = new Set(); }

  setState(s, why = "") {
    this.state = s;
    this.ui.root.dataset.state = s;
    this.ui.mic.setAttribute("aria-pressed", String(s === "live"));
    this.ui.mic.title = why || { off: "talk to the agent", connecting: "connecting…", live: "end the conversation" }[s] || s;
    this.ui.mute.hidden = s !== "live";
    this.ui.mic.innerHTML = `${MIC}<span>${s === "off" ? "talk" : s === "connecting" ? "connecting" : "end"}</span>`;
  }

  async start() {
    this.setState("connecting");
    try {
      this.ws = new WebSocket(wsUrl("/ws/voice"));
      this.ws.binaryType = "arraybuffer";
      const ready = await new Promise((resolve, reject) => {
        this.ws.onmessage = (e) => { const ev = JSON.parse(e.data); events.push(ev); ev.type === "ready" ? resolve(ev) : reject(new Error(ev.message || ev.type)); };
        this.ws.onerror = () => reject(new Error("voice socket failed"));
        this.ws.onclose = (e) => reject(new Error(e.reason || "voice socket closed"));
      });
      this.ready = ready;
      this.ws.onmessage = (e) => this.onMessage(e);
      this.ws.onclose = () => this.stop();
      this.out = new AudioContext({ sampleRate: ready.output_rate });
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
      this.ctx = new AudioContext();
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await this.ctx.audioWorklet.addModule(url);
      this.node = new AudioWorkletNode(this.ctx, "strands-down", { processorOptions: { rate: ready.input_rate } });
      this.node.port.onmessage = ({ data }) => {
        this.ui.level.style.setProperty("--level", this.muted ? 0 : Math.min(1, data.rms * 4).toFixed(3));
        if (!this.muted && this.ws && this.ws.readyState === 1) this.ws.send(data.pcm);
      };
      this.ctx.createMediaStreamSource(this.stream).connect(this.node);
      row("sys", `voice on · ${ready.provider} · ${ready.model}`);
      this.setState("live");
    } catch (err) {
      row("err", `voice: ${err.message || err}`);
      events.push({ type: "error", message: String(err.message || err) });
      this.stop(String(err.message || err));
    }
  }

  stop(why = "") {
    const was = this.state === "live";
    if (this.ws) { const ws = this.ws; this.ws = null; ws.onclose = null; try { ws.send(JSON.stringify({ type: "bye" })); } catch {} ws.close(); }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close();
    if (this.out) this.out.close();
    this.stream = this.ctx = this.out = this.node = null;
    this.sources.clear();
    this.ui.level.style.setProperty("--level", 0);
    if (was) row("sys", "voice off");
    this.setState("off", why);
  }

  toggleMute() {
    this.muted = !this.muted;
    this.ui.mute.setAttribute("aria-pressed", String(this.muted));
    this.ui.mute.textContent = this.muted ? "unmute" : "mute";
    this.ui.root.dataset.muted = String(this.muted);
  }

  play(buf) {
    if (!this.out) return;
    const pcm = new Int16Array(buf), f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 0x8000;
    const b = this.out.createBuffer(1, f.length, this.ready.output_rate);
    b.copyToChannel(f, 0);
    const src = this.out.createBufferSource();
    src.buffer = b; src.connect(this.out.destination);
    this.playAt = Math.max(this.playAt, this.out.currentTime + 0.03);
    src.start(this.playAt); this.playAt += b.duration;
    this.sources.add(src); src.onended = () => this.sources.delete(src);
    window.__voiceAudioBytes = (window.__voiceAudioBytes || 0) + buf.byteLength;
  }

  onMessage(e) {
    if (typeof e.data !== "string") { this.play(e.data); return; }
    const ev = JSON.parse(e.data);
    events.push(ev);
    if (ev.type === "transcript") {
      let el = this.lines.get(ev.id);
      if (!el) { el = row(ev.role === "user" ? "user" : "assistant"); this.lines.set(ev.id, el); }
      if (ev.final) el.textContent = ev.text; else el.textContent += ev.delta;
    } else if (ev.type === "tool_use") toolRow("⚙", ev.name, JSON.stringify(ev.input, null, 1));
    else if (ev.type === "tool_result") toolRow(`↳`, `${ev.name} · ${ev.status}`, ev.text || "");
    else if (ev.type === "interrupt") { this.sources.forEach((s) => { try { s.stop(); } catch {} }); this.sources.clear(); this.playAt = 0; }
    else if (ev.type === "error") row("err", `voice: ${ev.message}`);
    const log = $("#log"); if (log) log.scrollTop = 1e9;
  }
}

function mount() {
  const root = document.createElement("span");
  root.id = "voice";
  root.dataset.state = "off";
  root.dataset.panel = "voice";
  root.innerHTML = `<button id="voice-mic" type="button" aria-label="voice" aria-pressed="false"></button>`
    + `<span id="voice-level" aria-hidden="true"></span>`
    + `<button id="voice-mute" type="button" aria-pressed="false" hidden>mute</button>`;
  const style = document.createElement("style");
  style.textContent = `
    #voice{display:flex;align-items:center;gap:8px;padding:6px 0;font-size:.85em}
    #voice button{font:inherit;cursor:pointer;display:inline-flex;align-items:center;gap:5px}
    #voice-mic:disabled{opacity:.45;cursor:not-allowed}
    #voice[data-state=live] #voice-mic{background:var(--sr-accent,#1f9d55);color:#fff}
    #voice[data-state=connecting] #voice-mic{opacity:.6}
    #voice-level{flex:1;max-width:140px;height:6px;border-radius:3px;background:var(--sr-line,#8884);overflow:hidden;position:relative;--level:0}
    #voice-level::after{content:"";position:absolute;inset:0;transform-origin:left;transform:scaleX(var(--level));background:var(--sr-accent,#1f9d55);transition:transform 60ms linear}
    #voice[data-state=off] #voice-level{opacity:.35}
    #voice[data-muted=true] #voice-level::after{background:var(--sr-muted,#999)}
    .msg.voice.sys{opacity:.7;font-size:.9em}`;
  document.head.appendChild(style);
  const ui = { root, mic: root.querySelector("#voice-mic"), level: root.querySelector("#voice-level"), mute: root.querySelector("#voice-mute") };
  const v = new Voice(ui);
  window.__voice = v;
  ui.mic.addEventListener("click", () => (v.state === "live" || v.state === "connecting" ? v.stop() : v.start()));
  ui.mute.addEventListener("click", () => v.toggleMute());
  v.setState("off");
  fetch("/api/voice").then((r) => (r.ok ? r.json() : null)).then((info) => {
    if (!info || !info.enabled) { ui.mic.disabled = true; ui.mic.title = (info && info.hint) || "voice is off on this server"; }
    window.__voiceInfo = info;
  }).catch(() => { ui.mic.disabled = true; ui.mic.title = "no dashboard server: voice needs one"; });
  return root;
}

const SR = window.SR;
if (SR && typeof SR.registerSlot === "function") SR.registerSlot("chat-input", () => mount(), { id: "voice" });
else if (window.strandsSlots && typeof window.strandsSlots.register === "function") window.strandsSlots.register("chat-input", mount);
else {
  const form = $("#chatform");  // until the shell's slot API: a row of its own above the input, never over it
  if (form) form.parentNode.insertBefore(mount(), form);
}
