/* Record panel: pick robots and their cameras, name the task, then REC → save / discard episodes → stop.
 * Every request goes to /api/record/* (dashboard/record.py); the recorder only observes, so nothing here moves a
 * robot or needs X-Confirm. It is a side-column tab (window.SR.registerPanel, loaded through GET /api/panels) and
 * puts a REC dot in the top bar slot while a take is live.
 */
import { authed, image } from "../auth.js";

let BASE = (window.__state && window.__state.base) || "";
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const JSON_POST = { method: "POST", headers: { "content-type": "application/json" } };

const CSS = `
#record { padding: .5rem .6rem; border-bottom: 1px solid var(--sr-line, #ccc); font-family: var(--sr-text, sans-serif); color: var(--sr-fg, inherit); }
#record .rec-head { display: flex; align-items: center; gap: .5rem; font-family: var(--sr-mono, monospace); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }
.rec-dot { width: .6rem; height: .6rem; border-radius: 50%; background: var(--sr-muted, #999); flex: none; display: inline-block; }
#record[data-state="recording"] .rec-dot, .rec-bar[data-state="recording"] .rec-dot { background: var(--sr-bad, #d22); animation: rec-blink 1s steps(2, start) infinite; }
#record[data-state="saving"] .rec-dot { background: var(--sr-warn, #b80); }
.rec-bar { display: inline-flex; align-items: center; gap: .35rem; font-family: var(--sr-mono, monospace); font-size: .66rem; color: var(--sr-bad, #d22); cursor: pointer; background: none; border: 0; padding: 0 .4rem; }
.rec-bar[hidden] { display: none; }
#record .rec-meter[data-dropped] { color: var(--sr-warn, #b80); }
@keyframes rec-blink { to { visibility: hidden; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation: none !important; } }
#record .rec-summary { margin-left: auto; color: var(--sr-fg-light, #555); text-transform: none; letter-spacing: 0; }
#record .rec-body { display: grid; gap: .45rem; margin-top: .5rem; font-size: .74rem; }
#record fieldset { border: 1px solid var(--sr-line, #ccc); border-radius: var(--sr-radius, 8px); padding: .35rem .5rem; margin: 0; min-width: 0; }
#record legend { font-family: var(--sr-mono, monospace); font-size: .62rem; color: var(--sr-muted, #777); padding: 0 .25rem; }
#record label { display: flex; align-items: center; gap: .35rem; }
#record .rec-robot + .rec-cams { display: flex; flex-wrap: wrap; gap: .35rem; margin: .25rem 0 .35rem 1.3rem; }
#record .rec-cam { flex-direction: column; align-items: flex-start; gap: .15rem; font-family: var(--sr-mono, monospace); font-size: .62rem; }
#record .rec-cam img { width: 5.4rem; height: auto; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 4px; border: 1px solid var(--sr-line, #ccc); background: var(--sr-viewer-bg, #eee); }
#record .rec-cam input:checked ~ img { outline: 2px solid var(--sr-accent, #0a5); }
#record input[type=text], #record input[type=number] { min-width: 0; width: 100%; box-sizing: border-box; background: var(--sr-bg, #fff); color: var(--sr-fg, #000); border: 1px solid var(--sr-pill-border, #999); border-radius: 999px; padding: .32rem .7rem; font-family: var(--sr-mono, monospace); font-size: .66rem; }
#record .rec-row { display: grid; grid-template-columns: minmax(0, 1fr) 4.5rem; gap: .35rem; }
#record .rec-btns { display: flex; gap: .35rem; flex-wrap: wrap; }
#record button { font-family: var(--sr-mono, monospace); font-size: .66rem; border-radius: 999px; padding: .32rem .8rem; border: 1px solid var(--sr-pill-border, #999); background: var(--sr-pill-bg, #fff); color: var(--sr-fg, #000); cursor: pointer; }
#record button.rec-go { background: var(--sr-accent, #0a5); color: var(--sr-on-accent, #fff); border-color: transparent; }
#record button.rec-stop { border-color: var(--sr-bad, #d22); color: var(--sr-bad, #d22); }
#record button:disabled { opacity: .45; cursor: default; }
#record .rec-meter { font-family: var(--sr-mono, monospace); font-size: .66rem; color: var(--sr-fg-light, #444); }
#record .rec-error { color: var(--sr-bad, #d22); font-family: var(--sr-mono, monospace); font-size: .64rem; white-space: pre-wrap; }
#record ol { margin: 0; padding-left: 1.2rem; font-family: var(--sr-mono, monospace); font-size: .62rem; max-height: 7rem; overflow: auto; }
#record ol:empty::before { content: "no episodes yet"; color: var(--sr-muted, #777); margin-left: -1.2rem; }
`;

const HTML = `
<div class="rec-head"><span class="rec-dot" aria-hidden="true"></span><span>Record a dataset</span><span class="rec-summary" aria-live="polite">idle</span></div>
<div class="rec-body">
  <fieldset class="rec-robots"><legend>robots · cameras</legend><div class="rec-list"></div></fieldset>
  <input type="text" class="rec-task" placeholder="task: pick up the cube" aria-label="task" autocomplete="off">
  <div class="rec-row">
    <input type="text" class="rec-target" aria-label="dataset target (directory or hf://owner/name)" autocomplete="off" spellcheck="false">
    <input type="number" class="rec-fps" min="1" max="60" step="1" value="30" aria-label="fps" title="frames per second">
  </div>
  <div class="rec-btns">
    <button type="button" class="rec-go">● rec</button>
    <button type="button" class="rec-save" disabled>save episode</button>
    <button type="button" class="rec-discard" disabled>discard</button>
    <button type="button" class="rec-stop" title="end the recording; the take in progress is kept as the last episode (discard it first to drop it)" disabled>stop</button>
  </div>
  <div class="rec-meter"></div>
  <div class="rec-error" role="alert"></div>
  <fieldset><legend>recorded episodes</legend><ol class="rec-episodes"></ol></fieldset>
</div>`;

function panel() {
  const el = document.createElement("div");
  el.id = "record";
  el.dataset.state = "idle";
  el.innerHTML = HTML;
  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);
  return el;
}

const ui = { el: null, opts: null, status: null, busy: false, thumbs: null, ctx: null, bar: null };
const shown = () => !!ui.el && ui.el.offsetParent !== null; // its tab is the open one
const q = (s) => ui.el.querySelector(s);

async function api(path, init) {
  const res = await authed(`${BASE}${path}`, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

function renderOptions(opts) {
  ui.opts = opts;
  if (!q(".rec-target").value) q(".rec-target").value = opts.target;
  const was = new Set([...ui.el.querySelectorAll(".rec-robot input:checked")].map((i) => i.value));
  const camsWas = new Set([...ui.el.querySelectorAll(".rec-cam input:checked")].map((i) => `${i.dataset.robot}\u0000${i.value}`));
  const first = !ui.el.querySelector(".rec-robot");
  q(".rec-list").innerHTML = opts.robots.map((r, i) => {
    const on = first ? i === 0 : was.has(r.id);
    // a robot's own cameras first, then the world's unattached ones (any world camera can be recorded with a robot)
    const cams = opts.cameras.filter((c) => c.robot === r.id || !c.robot);
    const camHtml = cams.map((c) => {
      const checked = first ? c.robot === r.id : camsWas.has(`${r.id}\u0000${c.name}`);
      return `<label class="rec-cam"><span><input type="checkbox" data-robot="${esc(r.id)}" value="${esc(c.name)}" ${checked ? "checked" : ""}> ${esc(c.name)}</span>` +
        `<img alt="camera ${esc(c.name)}" data-src="${esc(BASE)}${esc(c.url)}" width="86" height="64"></label>`;
    }).join("");
    return `<label class="rec-robot"><input type="checkbox" value="${esc(r.id)}" ${on ? "checked" : ""}> ${esc(r.id)} <small>${esc(r.name)} · ${r.is_real ? "real" : "sim"} · ${esc(r.units)}</small></label>` +
      `<div class="rec-cams">${camHtml || '<small>no cameras in this world</small>'}</div>`;
  }).join("") || "<small>no robots in this world</small>";
  refreshThumbs();
}

function refreshThumbs() {
  if (!shown() || document.hidden) return;
  for (const img of ui.el.querySelectorAll(".rec-cam img")) image(img, `${img.dataset.src}?t=${Date.now()}`);
}

function chosen() {
  const robots = [...ui.el.querySelectorAll(".rec-robot input:checked")].map((i) => i.value);
  const cameras = {};
  for (const rid of robots) cameras[rid] = [...ui.el.querySelectorAll(".rec-cam input:checked")].filter((i) => i.dataset.robot === rid).map((i) => i.value);
  return { robots, cameras, target: q(".rec-target").value.trim(), task: q(".rec-task").value.trim() || null, fps: +q(".rec-fps").value };
}

function renderStatus(st) {
  ui.status = st;
  const live = st.state === "recording" || st.state === "saving";
  ui.el.dataset.state = st.state;
  const n = st.episodes.length;
  q(".rec-summary").textContent = live
    ? `${st.state === "saving" ? "saving" : "REC"} ep ${st.episode} · ${st.episode_seconds.toFixed(1)} s · ${st.frames} f`
    : n ? `idle · ${new Set(st.episodes.map((e) => e.index)).size} episodes saved` : "idle";
  const dropped = live && st.dropped > 0 ? ` · dropped ${st.dropped}` : "";
  q(".rec-meter").textContent = live
    ? `episode ${st.episode} · ${st.frames} frames${dropped} · ${st.fps_achieved} / ${st.fps} fps · ${st.elapsed.toFixed(0)} s total → ${st.target}`
    : st.target ? `last: ${st.target}` : "";
  if (dropped) q(".rec-meter").dataset.dropped = ""; else delete q(".rec-meter").dataset.dropped;
  ui.ctx?.setBadge(live ? "REC" : null);
  if (ui.bar) {
    ui.bar.hidden = !live;
    ui.bar.dataset.state = st.state;
    ui.bar.querySelector(".rec-bar-text").textContent = live ? `REC ep ${st.episode} · ${st.episode_seconds.toFixed(0)} s` : "";
  }
  q(".rec-error").textContent = st.error || "";
  for (const s of [".rec-save", ".rec-discard", ".rec-stop"]) q(s).disabled = !live || ui.busy;
  q(".rec-go").disabled = live || ui.busy;
  for (const i of ui.el.querySelectorAll(".rec-body fieldset.rec-robots input, .rec-task, .rec-target, .rec-fps")) i.disabled = live;
  q(".rec-episodes").innerHTML = st.episodes.map((e) =>
    `<li data-index="${esc(e.index)}">ep ${esc(e.index)} · ${esc(e.robot)} · ${esc(e.frames)} f · ${esc(e.seconds)} s <small title="${esc(e.path)}">${esc(e.path)}</small></li>`).join("");
}

async function act(fn, label) {
  ui.busy = true;
  if (ui.status) renderStatus(ui.status);
  if (label) q(".rec-summary").textContent = label;
  q(".rec-error").textContent = "";
  try { renderStatus(await fn()); } catch (e) { q(".rec-error").textContent = String(e.message || e); } finally { ui.busy = false; if (ui.status) renderStatus(ui.status); }
}

function wire() {
  q(".rec-go").addEventListener("click", () => act(() => {
    const body = chosen();
    if (!body.robots.length) throw new Error("tick at least one robot");
    return api("/api/record/start", { ...JSON_POST, body: JSON.stringify(body) });
  }, "starting…"));
  q(".rec-save").addEventListener("click", () => act(() => api("/api/record/episode/save", { ...JSON_POST, body: "{}" }), "saving episode…"));
  q(".rec-discard").addEventListener("click", () => act(() => api("/api/record/episode/discard", { ...JSON_POST, body: "{}" })));
  q(".rec-stop").addEventListener("click", () => act(() => api("/api/record/stop", { ...JSON_POST, body: JSON.stringify({ save: true }) }), "saving…"));
  ui.ctx?.on("tab", ({ id } = {}) => { if (id === "record") refreshThumbs(); });
}

async function poll() {
  try {
    const st = await api("/api/record/status");
    if (!ui.busy) renderStatus(st);
  } catch { /* no server (static/standalone page): the panel stays idle */ }
}

async function loadOptions() {
  try {
    const opts = await api("/api/record/options");
    const sig = JSON.stringify([opts.robots, opts.cameras.map((c) => [c.name, c.robot])]);
    if (sig !== ui.sig) { ui.sig = sig; renderOptions(opts); }
  } catch { /* no recorder on this server */ }
}

export async function mount(host, ctx = null) {
  ui.ctx = ctx;
  if (ctx) BASE = ctx.base || "";
  ui.el = panel();
  host.appendChild(ui.el);
  wire();
  await loadOptions();
  await poll();
  setInterval(poll, 500);
  setInterval(loadOptions, 3000); // a robot or camera added by a script or the agent shows up
  setInterval(refreshThumbs, 1500);
  window.__record = ui;
}

/** The bar slot: a REC dot + episode clock while a take is live; a click opens the Record tab. */
function mountBar(el, ctx) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "rec-bar";
  b.hidden = true;
  b.title = "recording: open the Record tab";
  b.innerHTML = '<span class="rec-dot" aria-hidden="true"></span><span class="rec-bar-text"></span>';
  b.addEventListener("click", () => window.SR?.select("record"));
  el.appendChild(b);
  ui.bar = b;
  if (ui.status) renderStatus(ui.status);
}

async function boot() {
  BASE = (window.__state && window.__state.base) || "";
  if (!(await authed(`${BASE}/api/record/status`).then((r) => r.ok, () => false))) return; // static / standalone page
  const SR = window.SR;
  if (!SR || typeof SR.registerPanel !== "function") return; // a page without the panel slot API has no Record tab
  SR.registerPanel({ id: "record", title: "Record", order: 30, mount: (el, ctx) => { mount(el, ctx); } });
  try { SR.registerSlot("bar", mountBar, { id: "record-bar" }); } catch { /* no bar slot on this page */ }
}

boot();
