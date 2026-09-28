/*
 * <robot-viewer name="so101" [autoload]></robot-viewer>
 *
 * Renders any robot in the strands-robots registry in the browser, on MuJoCo
 * itself (the official @mujoco/mujoco WebAssembly build) with three.js for the
 * pixels. The MJCF and meshes stream from jsDelivr in front of the model's
 * public git repository at view time, so the docs ship no mesh files.
 *
 * One finger orbits, two fingers pan and zoom. Sliders move joints through
 * mj_forward; the Physics switch steps mj_step at real time with the sliders as
 * actuator targets. The code panel mirrors the sliders as robot.act({...}).
 *
 * Dependencies resolve through the import map in overrides/main.html:
 *   three, three/addons/, @mujoco/mujoco  (all pinned on cdn.jsdelivr.net)
 *
 * No build step. ES2022. Every embind handle is deleted on unload.
 */

const MANIFEST_URL = new URL("robots.json", import.meta.url);
const HIDDEN_GROUPS = new Set([3, 4, 5]); // MuJoCo convention: 3+ is collision geometry
const MAX_GEOMS = 20000;

let mujocoPromise = null;
let threePromise = null;
let manifestPromise = null;

/** The engine URL the page's import map names, handed to the worker (workers cannot read import maps). */
function mujocoUrl() {
  if (!mujocoPromise) {
    const map = document.querySelector('script[type="importmap"]');
    const imports = map ? JSON.parse(map.textContent).imports || {} : {};
    mujocoPromise = Promise.resolve(imports["@mujoco/mujoco"] || "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js");
  }
  return mujocoPromise;
}

/**
 * MuJoCo in a Worker (mujoco-worker.js), so compiling a model never blocks the page.
 * One engine per viewer; `dispose()` terminates it.
 */
class Engine {
  constructor(onPose) {
    this._worker = new Worker(new URL("mujoco-worker.js", import.meta.url), { type: "module" });
    this._pending = new Map();
    this._onPose = onPose;
    this._ready = new Promise((resolve, reject) => {
      this._pending.set("init", { resolve, reject });
    });
    this._worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "ready") { this._pending.get("init")?.resolve(); this._pending.delete("init"); return; }
      if (msg.type === "pose") { this._onPose(msg); return; }
      if (msg.type === "compiled") { this._pending.get(msg.id)?.resolve(msg); this._pending.delete(msg.id); return; }
      if (msg.type === "error") {
        const p = msg.id !== undefined ? this._pending.get(msg.id) : this._pending.get("init");
        (p || this._pending.get("init"))?.reject(new Error(msg.message));
        this._pending.delete(msg.id ?? "init");
      }
    };
    this._worker.onerror = (e) => { for (const p of this._pending.values()) p.reject(new Error(e.message || "worker failed")); this._pending.clear(); };
    mujocoUrl().then((url) => this._worker.postMessage({ type: "init", url }));
  }

  ready() { return this._ready; }

  /** Compile in the worker. Buffers are transferred, so `files` is unusable afterwards. */
  compile(sceneXml, fetched) {
    const id = "c" + Math.random().toString(36).slice(2);
    const files = [];
    const transfer = new Set();
    for (const [path, buf] of fetched) {
      // The 1x1 PNG stand-in is one buffer shared by every texture key; a transferable must be unique.
      const own = transfer.has(buf.buffer) ? buf.slice() : buf;
      files.push([path, own]);
      transfer.add(own.buffer);
    }
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({ type: "compile", id, sceneXml, files }, [...transfer]);
    });
  }

  setQpos(qadr, value, act) { this._worker.postMessage({ type: "setQpos", qadr, value, act }); }
  reset() { this._worker.postMessage({ type: "reset" }); }
  physics(on) { this._worker.postMessage({ type: "physics", on }); }
  dispose() { try { this._worker.postMessage({ type: "dispose" }); } catch { /* already gone */ } this._worker.terminate(); }
}
function loadThree() {
  if (!threePromise) {
    threePromise = Promise.all([import("three"), import("three/addons/controls/OrbitControls.js"), import("three/addons/environments/RoomEnvironment.js")]).then(
      ([THREE, { OrbitControls }, { RoomEnvironment }]) => ({ THREE, OrbitControls, RoomEnvironment })
    );
  }
  return threePromise;
}
function loadManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(MANIFEST_URL).then((r) => {
      if (!r.ok) throw new Error(`robots.json ${r.status}`);
      return r.json();
    });
  }
  return manifestPromise;
}

const fmtMB = (n) => `${(n / 1e6).toFixed(1)} MB`;
// A valid 1x1 white RGB PNG (69 bytes, Pillow), used in place of every texture file.
const PNG_1X1 = Uint8Array.from(atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4//8/AAX+Av4zEpUUAAAAAElFTkSuQmCC"), (c) => c.charCodeAt(0));

/** Parse one MJCF file for what it references. Regex is enough: attribute order varies, tags do not. */
function scanXml(xml) {
  const attr = (tag, name) => {
    const m = xml.match(new RegExp(`<${tag}\\b[^>]*\\b${name}="([^"]*)"`));
    return m ? m[1] : null;
  };
  const includes = [...xml.matchAll(/<include\s+[^>]*file="([^"]+)"/g)].map((m) => m[1]);
  const models = [...xml.matchAll(/<model\b[^>]*\bfile="([^"]+)"/g)].map((m) => m[1]);
  const assetdir = attr("compiler", "assetdir");
  const meshdir = attr("compiler", "meshdir") ?? assetdir;
  const texturedir = attr("compiler", "texturedir") ?? assetdir;
  const files = [];
  const textures = [];
  for (const m of xml.matchAll(/<(mesh|hfield|skin)\b[^>]*\bfile="([^"]+)"/g)) files.push(m[2]);
  for (const m of xml.matchAll(/<texture\b[^>]*\bfile="([^"]+)"/g)) textures.push(m[1]);
  for (const m of xml.matchAll(/<texture\b[^>]*\bfile(?:right|left|up|down|front|back)="([^"]+)"/g)) textures.push(m[1]);
  return { includes, models, files, textures, meshdir, texturedir };
}

/** Join path segments the way MuJoCo does and collapse "." and "..". A ".." with nothing
 *  left to pop is kept, so a meshdir of "../meshes" seen from "xml/" stays "../meshes" as a
 *  VFS key and resolves to "meshes" once joined with the scene directory for the download. */
function joinPath(...parts) {
  const out = [];
  for (const seg of parts.filter(Boolean).join("/").split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (out.length && out[out.length - 1] !== "..") out.pop(); else out.push("..");
    } else out.push(seg);
  }
  return out.join("/");
}
const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");

const TEMPLATE = `
<style>
  :host { display:block; position:relative; font-family: Inter, system-ui, sans-serif; color: var(--sr-fg, #15171a); --_card: var(--sr-card-bg, #fff); --_border: var(--sr-card-border, #e3ded2); --_muted: var(--sr-muted, #6b6760); --_accent: var(--sr-accent, #00c46a); --_btn-bg: var(--sr-btn-bg, #15171a); --_btn-fg: var(--sr-btn-fg, #fff); }
  canvas { display:block; width:100%; height:100%; outline:none; }
  .poster, .status { position:absolute; inset:0; display:grid; place-items:center; text-align:center; padding:1rem; }
  .poster[hidden], .status[hidden], .chrome[hidden] { display:none; }
  .poster img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:.6; }
  .poster .card, .status .card { position:relative; background: color-mix(in srgb, var(--_card) 94%, transparent); backdrop-filter: blur(8px); border:1px solid var(--_border); border-radius:12px; padding:.9rem 1.1rem; max-width:22rem; box-shadow: 0 8px 30px rgba(0,0,0,.12); }
  .poster h4, .status h4 { margin:0 0 .25rem; font-size:.95rem; color: var(--sr-fg, #15171a); }
  .poster p, .status p { margin:0 0 .6rem; font-size:.75rem; color: var(--_muted); }
  button { font: inherit; font-size:.75rem; font-weight:600; padding:.45rem .9rem; border-radius:8px; border:1px solid var(--_btn-bg); background: var(--_btn-bg); color: var(--_btn-fg); cursor:pointer; }
  button:hover { border-color: var(--_accent); background: var(--_accent); color: var(--sr-on-accent, #06210f); }
  .bar { height:4px; border-radius:2px; background: var(--_border); overflow:hidden; margin-top:.5rem; }
  .bar i { display:block; height:100%; width:0; background: var(--_accent); transition: width 120ms linear; }
  .chrome { position:absolute; left:0; right:0; bottom:0; display:flex; gap:.4rem; align-items:center; padding:.5rem .6rem; pointer-events:none; }
  .chrome > * { pointer-events:auto; }
  .chrome .spacer { flex:1; }
  .pill { font-size:.65rem; font-weight:500; padding:.3rem .6rem; border-radius:999px; border:1px solid var(--_border); background: color-mix(in srgb, var(--_card) 90%, transparent); backdrop-filter: blur(6px); color: var(--sr-fg, #15171a); cursor:pointer; }
  .pill[aria-pressed="true"] { border-color: var(--_accent); color: var(--_accent); }
  .pill:hover { border-color: var(--_accent); background: color-mix(in srgb, var(--_card) 90%, transparent); color: var(--_accent); }
  .joints { position:absolute; top:.6rem; right:.6rem; width: 13.5rem; max-height: calc(100% - 3.6rem); overflow:auto; background: color-mix(in srgb, var(--_card) 88%, transparent); backdrop-filter: blur(8px); border:1px solid var(--_border); border-radius:10px; padding:.55rem .7rem .5rem; font-size:.68rem; scrollbar-width: thin; }
  .joints[hidden] { display:none; }
  .joints h5 { margin:0 0 .35rem; font-size:.62rem; letter-spacing:.06em; text-transform:uppercase; color: var(--_muted); font-weight:600; }
  .joint { display:grid; gap:.05rem; margin-bottom:.25rem; }
  .joint label { display:flex; justify-content:space-between; font-family: "JetBrains Mono", ui-monospace, monospace; font-size:.6rem; color: var(--sr-fg, #15171a); }
  .joint label output { color: var(--_muted); }
  .joint input[type=range] { width:100%; accent-color: var(--_accent); margin:0; height: .9rem; }
  .code { position:absolute; top:.6rem; left:.6rem; max-width: calc(100% - 15.5rem); font-family: "JetBrains Mono", ui-monospace, monospace; font-size:.62rem; line-height:1.45; color: var(--sr-fg, #15171a); background: color-mix(in srgb, var(--_card) 88%, transparent); backdrop-filter: blur(8px); border:1px solid var(--_border); border-radius:10px; padding:.5rem .7rem; white-space:pre; overflow:auto; max-height: 40%; }
  .code[hidden] { display:none; }
  .code b { color: var(--_accent); font-weight:600; }
  .sheet-head { display:none; }
  /* Narrow stage (a phone, or a small column): the panels become bottom sheets over the chrome,
     the pills grow to finger size and wrap onto a second row, and nothing sits over the robot by default. */
  :host([narrow]) .pill { min-height: 40px; min-width: 40px; font-size:.72rem; padding:.45rem .8rem; }
  :host([narrow]) .chrome { flex-wrap: wrap; padding:.45rem .5rem; gap:.35rem; }
  :host([narrow]) .chrome .spacer { display:none; }
  :host([narrow]) .joints, :host([narrow]) .code { top:auto; right:0; left:0; bottom:0; width:auto; max-width:none; max-height: 52%; z-index:2; border-radius: 14px 14px 0 0; border-bottom:0; padding-bottom: max(.6rem, env(safe-area-inset-bottom)); box-shadow: 0 -8px 30px rgba(0,0,0,.18); }
  :host([narrow]) .chrome { z-index:1; }
  /* While a sheet is open the stage slides up so the robot stays in the visible half. */
  :host([narrow][sheet]) canvas { transform: translateY(-24%); }
  @media (prefers-reduced-motion: no-preference) { canvas { transition: transform 160ms ease; } }
  :host([narrow]) .joints { font-size:.72rem; }
  :host([narrow]) .joint { margin-bottom:.45rem; }
  :host([narrow]) .joint label { font-size:.66rem; }
  :host([narrow]) .joint input[type=range] { height: 1.6rem; }
  :host([narrow]) .code { font-size:.62rem; }
  :host([narrow]) .sheet-head { display:flex; justify-content:space-between; align-items:center; margin:0 0 .4rem; white-space:normal; font-family: Inter, system-ui, sans-serif; }
  :host([narrow]) .sheet-head h5 { font-size:.62rem; letter-spacing:.06em; text-transform:uppercase; color: var(--_muted); font-weight:600; }
  :host([narrow]) .sheet-head h5 { margin:0; }
  :host([narrow]) .sheet-head button { min-height: 40px; min-width: 40px; padding:.3rem .7rem; border-radius:999px; font-size:.72rem; }
  :host([narrow]) .joints > h5 { display:none; }
  @media (prefers-reduced-motion: reduce) { .bar i { transition: none; } }
</style>
<canvas tabindex="0" aria-label="3D robot viewer"></canvas>
<div class="poster" part="poster"></div>
<div class="code" hidden></div>
<div class="joints" hidden></div>
<div class="chrome" hidden>
  <button class="pill" data-act="reset" title="Reset pose (R)">Reset</button>
  <button class="pill" data-act="physics" aria-pressed="false" title="Step MuJoCo at real time">Physics</button>
  <button class="pill" data-act="collision" aria-pressed="false" title="Show collision geometry">Collision</button>
  <span class="spacer"></span>
  <button class="pill" data-act="joints" aria-pressed="true">Joints</button>
  <button class="pill" data-act="code" aria-pressed="false">Code</button>
  <button class="pill" data-act="full" title="Fullscreen">Full</button>
</div>
`;

class RobotViewer extends HTMLElement {
  static get observedAttributes() { return ["name"]; }

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).innerHTML = TEMPLATE;
    this.$ = (s) => this.shadowRoot.querySelector(s);
    this._state = "idle";
    this._physics = false;
    this._showCollision = false;
    this._three = null;
    this._raf = null;
    this._joints = [];
    this._onKey = (e) => { if (e.key === "r" || e.key === "R") this.resetPose(); };
  }

  connectedCallback() {
    this._entry = null;
    loadManifest()
      .then((m) => {
        this._entry = m.robots[this.getAttribute("name")] ?? null;
        this._renderPoster();
        // A phone, a save-data connection or a reduced-motion preference gets the poster: the 10 MB engine
        // and the meshes download only after the reader presses Load 3D.
        if (this.hasAttribute("autoload") && !this._preferPoster()) this._whenVisible(() => this.load());
      })
      .catch((e) => this._fail(`Could not read the robot manifest (${e.message}).`));
    if (this._wired) return; // a name change re-enters here; the listeners below are per element, not per robot
    this._wired = true;
    this.shadowRoot.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-act]");
      if (b) this._action(b.dataset.act, b);
    });
    this.$("canvas").addEventListener("keydown", this._onKey);
    this._ro = new ResizeObserver(() => { this._syncNarrow(); this._resize(); });
    this._ro.observe(this);
    this._syncNarrow();
    this._onFsChange = () => {
      const on = document.fullscreenElement === this;
      this.toggleAttribute("fullscreen", on);
      this.$('[data-act="full"]').setAttribute("aria-pressed", String(on));
      this._syncNarrow();
      this._resize();
    };
    document.addEventListener("fullscreenchange", this._onFsChange);
    // Material for MkDocs toggles data-md-color-scheme on <body>; recolour the stage when it does.
    this._mo = new MutationObserver(() => this._applyTheme());
    this._mo.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  disconnectedCallback() {
    this._ro?.disconnect();
    this._mo?.disconnect();
    document.removeEventListener("fullscreenchange", this._onFsChange);
    if (this.hasAttribute("fullscreen-fallback")) this._toggleFallbackFullscreen();
    this.unload();
  }

  /** Under 640 CSS px the stage is "narrow": panels are bottom sheets and pills are finger sized. */
  _isNarrow() { return (this.clientWidth || this.getBoundingClientRect().width) < 640 || this.hasAttribute("compact"); }
  _syncNarrow() { this.toggleAttribute("narrow", this._isNarrow()); }
  _preferPoster() {
    const mm = (q) => typeof matchMedia === "function" && matchMedia(q).matches;
    return mm("(max-width: 40em)") || mm("(prefers-reduced-motion: reduce)") || navigator.connection?.saveData === true;
  }
  _reducedMotion() { return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches; }
  _toggleFallbackFullscreen() {
    const on = !this.hasAttribute("fullscreen-fallback");
    this.toggleAttribute("fullscreen-fallback", on);
    document.body.classList.toggle("sr-viewer-fullscreen", on);
    this.$('[data-act="full"]').setAttribute("aria-pressed", String(on));
    if (on) {
      this._onFsKey = (e) => { if (e.key === "Escape") this._toggleFallbackFullscreen(); };
      document.addEventListener("keydown", this._onFsKey);
    } else if (this._onFsKey) {
      document.removeEventListener("keydown", this._onFsKey);
      this._onFsKey = null;
    }
    this._syncNarrow();
    this._resize();
  }
  /** Close whichever bottom sheet is open (the sheet's own close button, narrow stages only). */
  _closeSheets() {
    for (const [sel, act] of [[".joints", "joints"], [".code", "code"]]) {
      const el = this.$(sel);
      if (!el.hidden) { el.hidden = true; this.$(`[data-act="${act}"]`).setAttribute("aria-pressed", "false"); }
    }
    this._syncSheet();
  }
  _syncSheet() { this.toggleAttribute("sheet", !this.$(".joints").hidden || !this.$(".code").hidden); }

  attributeChangedCallback(n, oldV, newV) {
    if (n === "name" && oldV && oldV !== newV) { this.unload(); this._clearStatus(); this.connectedCallback(); }
  }

  _whenVisible(fn) {
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
    }, { rootMargin: "200px" });
    io.observe(this);
  }

  _renderPoster() {
    const p = this.$(".poster");
    const e = this._entry;
    if (this._state !== "idle") return;
    if (!e) return this._fail(`No robot named "${this.getAttribute("name")}" in the registry.`);
    if (!e.sim) return this._fail(`${e.description} has no simulation model, so there is nothing to render.`);
    if (!e.viewer) return this._fail(e.viewer_note ? `${e.description} simulates locally, but ${e.viewer_note}.` : `${e.description} simulates locally, but its model has no public source to stream from.`);
    const thumb = e.thumbnail ? `<img alt="" src="${new URL("../../" + e.thumbnail, import.meta.url)}">` : "";
    p.innerHTML = `${thumb}<div class="card"><h4>${e.description}</h4><p>${e.joints ?? "?"} joints. Runs MuJoCo in your browser. Meshes stream from ${this._sourceLabel()}.</p><button data-act="load">Load 3D</button></div>`;
    p.hidden = false;
  }

  _sourceLabel() {
    const m = (this._entry?.base_url || "").match(/\/gh\/([^/]+\/[^/@]+)@/);
    return m ? m[1] : "jsDelivr";
  }

  _status(title, text, progress) {
    let s = this.$(".status");
    if (!s) { s = document.createElement("div"); s.className = "status"; this.shadowRoot.appendChild(s); }
    s.innerHTML = `<div class="card"><h4>${title}</h4><p>${text}</p>${progress == null ? "" : `<div class="bar"><i style="width:${(progress * 100).toFixed(1)}%"></i></div>`}</div>`;
  }
  _clearStatus() { this.$(".status")?.remove(); }

  _fail(msg) {
    this._state = "error";
    this.$(".poster").hidden = true;
    this._status("Viewer unavailable", msg);
  }

  _action(act, btn) {
    switch (act) {
      case "load": this.load(); return;
      case "close": this._closeSheets(); return;
      case "reset": this.resetPose(); return;
      case "physics":
        this._physics = !this._physics; btn.setAttribute("aria-pressed", String(this._physics));
        this._engine?.physics(this._physics);
        return;
      case "collision":
        this._showCollision = !this._showCollision; btn.setAttribute("aria-pressed", String(this._showCollision)); this._applyVisibility(); return;
      case "joints": case "code": {
        const el = this.$(`.${act}`);
        el.hidden = !el.hidden;
        btn.setAttribute("aria-pressed", String(!el.hidden));
        // On a narrow stage the two panels are sheets over the same spot: opening one closes the other.
        if (!el.hidden && this.hasAttribute("narrow")) {
          const other = act === "joints" ? "code" : "joints";
          this.$(`.${other}`).hidden = true;
          this.$(`[data-act="${other}"]`).setAttribute("aria-pressed", "false");
        }
        this._syncSheet();
        return;
      }
      case "full":
        if (document.fullscreenElement === this) { document.exitFullscreen(); return; }
        if (this.requestFullscreen) { this.requestFullscreen().catch(() => this._toggleFallbackFullscreen()); return; }
        this._toggleFallbackFullscreen();
        return;
    }
  }

  /** Fetch the model, compile it in MuJoCo, build the three.js scene. Idempotent. */
  async load() {
    if (this._state !== "idle") return;
    this._state = "loading";
    const gen = (this._gen = (this._gen || 0) + 1);
    const stale = () => gen !== this._gen || this._state !== "loading";
    try {
      if (!this._entry) {
        const m = await loadManifest();
        this._entry = m.robots[this.getAttribute("name")] ?? null;
        if (!this._entry?.viewer) throw new Error(`no streamable model for "${this.getAttribute("name")}"`);
      }
      const e = this._entry;
      this.$(".poster").hidden = true;
      this._status("Loading MuJoCo", "The WebAssembly engine is 10 MB and cached after the first robot.");
      this._engine?.dispose();
      this._engine = new Engine((pose) => this._onPose(pose));
      const [three] = await Promise.all([loadThree(), this._engine.ready()]);
      if (stale()) return;
      const files = await this._fetchAssets(e);
      if (stale()) return;
      this._status("Compiling model", `${files.count} files, ${fmtMB(files.bytes)}. The page stays yours meanwhile.`);
      await this._compile(files);
      if (stale()) return;
      await this._buildScene(three);
      if (stale()) return;
      this._buildJoints();
      this._clearStatus();
      this.$(".poster").hidden = true;
      this.$(".chrome").hidden = false;
      const narrow = this._isNarrow();
      // Joints open on a wide stage; the code card stays behind its pill so the robot is never covered twice.
      this.$(".joints").hidden = narrow;
      this.$(".code").hidden = true;
      this.$('[data-act="joints"]').setAttribute("aria-pressed", String(!narrow));
      this.$('[data-act="code"]').setAttribute("aria-pressed", "false");
      this._syncSheet();
      this._state = "ready";
      this._loop();
      this.dispatchEvent(new CustomEvent("robot-loaded", { detail: { name: this._entry.name } }));
    } catch (err) {
      if (stale()) return;
      console.error(err);
      this._fail(this._explain(err));
    }
  }

  _explain(err) {
    const m = String(err?.message || err);
    if (/Failed to fetch|NetworkError|Load failed/.test(m)) return "cdn.jsdelivr.net is unreachable from this network. The viewer needs it for the engine and the meshes.";
    if (/ 404/.test(m)) return `A model file is missing upstream: ${m}.`;
    return `MuJoCo refused the model: ${m.slice(0, 240)}`;
  }

  async _fetchAssets(e) {
    const base = e.base_url;
    const dec = new TextDecoder();
    const fetched = new Map();
    let bytes = 0, done = 0, total = 1;
    const progress = (label) => this._status("Streaming meshes", `${label} ${fmtMB(bytes)}, ${done}/${total} files`, done / total);
    const LFS = new Uint8Array([118, 101, 114, 115, 105, 111, 110, 32, 104, 116, 116, 112, 115, 58, 47, 47, 103, 105, 116, 45, 108, 102, 115]); // "version https://git-lfs"
    const isLfsPointer = (b) => b.length < 400 && LFS.every((c, i) => b[i] === c);
    const tryFetch = async (url) => {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const r = await fetch(url);
          if (r.ok) return r;
          if (r.status === 404 || r.status === 403) return r;
        } catch { /* network hiccup: retry */ }
        await new Promise((res) => setTimeout(res, 300 * (attempt + 1)));
      }
      return null;
    };
    const get = async (path) => {
      // jsDelivr first (fast, cached), raw.githubusercontent.com when jsDelivr refuses
      // (files over 20 MB), GitHub's LFS media host when the blob is an LFS pointer.
      let r = await tryFetch(new URL(path, base).href);
      if ((!r || !r.ok) && e.raw_url) r = await tryFetch(new URL(path, e.raw_url).href);
      if (!r || !r.ok) throw new Error(`${path} ${r ? r.status : "unreachable"}`);
      let buf = new Uint8Array(await r.arrayBuffer());
      if (isLfsPointer(buf) && e.lfs_url) {
        r = await fetch(new URL(path, e.lfs_url).href);
        if (!r.ok) throw new Error(`${path} (lfs) ${r.status}`);
        buf = new Uint8Array(await r.arrayBuffer());
      }
      bytes += buf.length; done += 1; progress(path.split("/").pop());
      return buf;
    };
    // MJCF path rules, as MuJoCo applies them (checked against models that load from
    // disk): the file named by an <include> or a <model> is relative to the directory
    // of the file that names it; a compiler meshdir/texturedir is relative to the MAIN
    // model file's directory even when an included file declares it (ability_hand:
    // hands/abh_right_large.xml says meshdir="./assets" and means mujoco_xml/assets);
    // an asset with no meshdir is relative to its own file's directory (lekiwi:
    // lekiwi/lekiwi.xml names meshes/base_plate.stl and means lekiwi/meshes). A
    // <model> is a model of its own, so it starts a new root at its directory, and an
    // included file inherits the includer's compiler dirs when it declares none. VFS
    // keys are the paths MuJoCo will compute, relative to the main model file's
    // directory; downloads join the scene dir.
    const sceneDir = dirOf(e.scene);
    const queue = [{ file: joinPath(e.scene.slice(sceneDir ? sceneDir.length + 1 : 0)), root: "", meshdir: null, texturedir: null }];
    const assets = new Set();
    const seen = new Set();
    let sceneXml = null;
    while (queue.length) {
      const { file, root, meshdir: pm, texturedir: pt } = queue.shift();
      if (seen.has(file)) continue;
      seen.add(file);
      total += 1;
      const buf = await get(joinPath(sceneDir, file));
      const text = dec.decode(buf);
      // The main file is handed to from_xml_string, which registers it in the VFS
      // itself; adding it here too fails on a scene literally named model.xml (rby1).
      if (sceneXml === null) sceneXml = text; else fetched.set(file, buf);
      const dir = dirOf(file);
      const s = scanXml(text);
      const meshdir = s.meshdir ?? pm, texturedir = s.texturedir ?? pt;
      for (const inc of s.includes) queue.push({ file: joinPath(dir, inc), root, meshdir, texturedir });
      for (const mdl of s.models) { const f = joinPath(dir, mdl); queue.push({ file: f, root: dirOf(f), meshdir: null, texturedir: null }); }
      for (const f of s.files) assets.add(meshdir ? joinPath(root, meshdir, f) : joinPath(dir, f));
      // Textures are not sampled by this viewer, so a 1x1 PNG stands in and the
      // (often multi-megabyte) images are never downloaded.
      for (const tx of s.textures) { const k = texturedir ? joinPath(root, texturedir, tx) : joinPath(dir, tx); if (!fetched.has(k)) fetched.set(k, PNG_1X1); }
    }
    total = seen.size + assets.size;
    progress("");
    // At most 8 downloads in flight: hundreds of parallel requests trip CDN limits.
    const pending = [...assets];
    await Promise.all(Array.from({ length: 8 }, async () => {
      while (pending.length) { const a = pending.shift(); fetched.set(a, await get(joinPath(sceneDir, a))); }
    }));
    return { sceneXml, fetched, bytes, count: fetched.size };
  }

  /** Compile in the worker; the page keeps its event loop while MuJoCo builds convex hulls. */
  async _compile({ sceneXml, fetched }) {
    const { model, pluginsStripped } = await this._engine.compile(sceneXml, fetched);
    this._model = model;
    this._pluginsStripped = pluginsStripped;
    if (!this._pose) await new Promise((r) => { this._poseWaiter = r; });
    this._qpos0 = Float64Array.from(this._pose.qpos);
  }

  /** A pose from the worker: the latest geom frames, and qpos for the sliders and the code card. */
  _onPose(pose) {
    this._pose = pose;
    if (this._poseWaiter) { const w = this._poseWaiter; this._poseWaiter = null; w(); }
    if (this._state !== "ready") return;
    if (this._physics || this._poseDirty) {
      for (const jt of this._joints) {
        const o = this.shadowRoot.getElementById(`o${jt.j}`);
        if (o) o.textContent = pose.qpos[jt.qadr].toFixed(2);
        if (this._poseDirty) { const inp = this.shadowRoot.getElementById(`j${jt.j}`); if (inp) inp.value = pose.qpos[jt.qadr]; }
      }
      if (this._poseDirty) { this._poseDirty = false; this._renderCode(); }
    }
  }

  async _buildScene({ THREE, OrbitControls, RoomEnvironment }) {
    const canvas = this.$("canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.setClearColor(0x000000, 0); // the stage colour is CSS (--sr-viewer-bg), so it follows the theme
    const scene = new THREE.Scene();
    // A neutral studio environment: specular sheen is what keeps a black robot readable on a dark stage.
    // Rendering the room into a cubemap is the one GPU-bound cost here, so it runs after
    // the first frame has been drawn: the robot appears at once and gains its sheen a frame later.
    this._pendingEnvironment = () => {
      const pmrem = new THREE.PMREMGenerator(renderer);
      scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
      pmrem.dispose();
    };
    const camera = new THREE.PerspectiveCamera(38, 4 / 3, 0.01, 200);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = !this._reducedMotion();
    controls.dampingFactor = 0.08;
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    controls.maxPolarAngle = Math.PI * 0.52;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x8a8f99, 0.7);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(1.5, -2, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.0002;
    key.shadow.normalBias = 0.02;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.7);
    fill.position.set(-2, 1.5, 1.5);
    scene.add(fill);
    // Rim light from behind and above: separates dark silhouettes from a dark stage.
    const rim = new THREE.DirectionalLight(0xdfe8ff, 1.1);
    rim.position.set(-1, 2.5, 2);
    scene.add(rim);
    this._lights = { hemi, key, fill, rim };

    const m = this._model;
    const G = m.enums.geom;
    const geomMeshes = [];
    const meshCache = new Map();
    const theme = this._theme();
    renderer.toneMappingExposure = theme.exposure;
    let sinceYield = performance.now();
    for (let g = 0; g < m.ngeom; g++) {
      // Building geometry is main-thread work; give the page a frame every ~12 ms of it.
      if (performance.now() - sinceYield > 12) { await new Promise((r) => setTimeout(r, 0)); sinceYield = performance.now(); }
      const type = m.geom_type[g];
      const size = [m.geom_size[3 * g], m.geom_size[3 * g + 1], m.geom_size[3 * g + 2]];
      const group = m.geom_group[g];
      let geometry;
      if (type === G.MESH) {
        const id = m.geom_dataid[g];
        geometry = meshCache.get(id) ?? this._meshGeometry(THREE, id);
        meshCache.set(id, geometry);
      } else if (type === G.PLANE) {
        geometry = new THREE.PlaneGeometry(size[0] ? 2 * size[0] : 40, size[1] ? 2 * size[1] : 40);
      } else if (type === G.SPHERE) {
        geometry = new THREE.SphereGeometry(size[0], 24, 16);
      } else if (type === G.CAPSULE) {
        geometry = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 20); geometry.rotateX(Math.PI / 2);
      } else if (type === G.CYLINDER) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 32); geometry.rotateX(Math.PI / 2);
      } else if (type === G.BOX) {
        geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
      } else if (type === G.ELLIPSOID) {
        geometry = new THREE.SphereGeometry(1, 24, 16); geometry.scale(size[0], size[1], size[2]);
      } else {
        continue; // hfield, sdf: not drawn in v1
      }
      let rgba = [m.geom_rgba[4 * g], m.geom_rgba[4 * g + 1], m.geom_rgba[4 * g + 2], m.geom_rgba[4 * g + 3]];
      const matid = m.geom_matid[g];
      if (matid >= 0) rgba = [m.mat_rgba[4 * matid], m.mat_rgba[4 * matid + 1], m.mat_rgba[4 * matid + 2], m.mat_rgba[4 * matid + 3]];
      const isPlane = type === G.PLANE;
      // The floor only catches the shadow; the stage colour behind it is the page's CSS.
      const material = isPlane
        ? new THREE.ShadowMaterial({ color: 0x000000, opacity: theme.shadow, transparent: true, side: THREE.DoubleSide })
        : new THREE.MeshStandardMaterial({
            color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
            roughness: 0.5,
            metalness: 0.1,
            envMapIntensity: theme.env,
            transparent: rgba[3] < 1,
            opacity: rgba[3],
          });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = !isPlane;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.userData = { geom: g, group, collision: HIDDEN_GROUPS.has(group), plane: isPlane };
      scene.add(mesh);
      geomMeshes.push(mesh);
    }
    this._three = { THREE, renderer, scene, camera, controls, geomMeshes };
    this._syncPoses();
    // Frame the robot: bounding box of every visible non-plane geom at the home pose.
    const bbox = new THREE.Box3();
    for (const mesh of geomMeshes) {
      if (mesh.userData.plane || mesh.userData.collision) continue;
      mesh.geometry.computeBoundingBox();
      bbox.union(mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrix));
    }
    if (bbox.isEmpty()) bbox.set(new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 1));
    const center = bbox.getCenter(new THREE.Vector3());
    const radius = Math.max(bbox.getSize(new THREE.Vector3()).length() / 2, 0.15);
    // Fit the bounding sphere to the narrower field of view, then slide the target so the
    // robot sits left of centre, clear of the joints panel.
    const w = this.clientWidth || 4, h = this.clientHeight || 3;
    camera.aspect = w / h;
    const vfov = THREE.MathUtils.degToRad(camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
    const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * (this.hasAttribute("compact") ? 0.92 : 1.08);
    const dir = new THREE.Vector3(1.0, -1.25, 0.55).normalize();
    camera.position.copy(center).addScaledVector(dir, dist);
    camera.near = radius / 100; camera.far = radius * 100; camera.updateProjectionMatrix();
    const right = new THREE.Vector3().crossVectors(dir.clone().negate(), camera.up).normalize();
    const shift = right.multiplyScalar(w >= 640 && !this.hasAttribute("compact") ? radius * 0.45 : 0);
    controls.target.copy(center).add(shift);
    camera.position.add(shift);
    key.shadow.camera.left = key.shadow.camera.bottom = -radius * 1.6;
    key.shadow.camera.right = key.shadow.camera.top = radius * 1.6;
    key.shadow.camera.near = radius * 0.5; key.shadow.camera.far = radius * 8;
    key.shadow.camera.updateProjectionMatrix();
    key.position.copy(center).add(new THREE.Vector3(radius * 1.5, -radius * 2, radius * 3));
    key.target.position.copy(center); scene.add(key.target);
    controls.minDistance = radius * 0.5; controls.maxDistance = radius * 12;
    // A faint grid on the floor for depth; hidden when the model has no ground plane.
    if (geomMeshes.some((x) => x.userData.plane)) {
      const grid = new THREE.GridHelper(Math.max(2, radius * 8), Math.max(8, Math.round(radius * 8 / 0.1)), theme.gridMajor, theme.gridMinor);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.0015;
      grid.material.transparent = true; grid.material.opacity = theme.gridOpacity;
      scene.add(grid);
      this._grid = grid;
    }
    this._applyVisibility();
    this._resize();
  }

  /** Stage colours from the page theme (CSS custom properties on the host), with paper defaults. */
  _theme() {
    const cs = getComputedStyle(this);
    const v = (name, fallback) => (cs.getPropertyValue(name).trim() || fallback);
    const dark = v("--sr-scheme", "paper") === "dark";
    return {
      dark,
      gridMajor: v("--sr-grid-major", dark ? "#3d454f" : "#d6d1c5"),
      gridMinor: v("--sr-grid-minor", dark ? "#2a2f36" : "#e6e1d6"),
      gridOpacity: dark ? 1.0 : 0.7,
      shadow: dark ? 0.55 : 0.22,
      env: dark ? 1.25 : 0.85,
      exposure: dark ? 1.05 : 0.88,
    };
  }

  /** Re-read the theme after a palette toggle and recolour the grid, floor shadow and reflections. */
  _applyTheme() {
    if (!this._three) return;
    const theme = this._theme();
    const { THREE } = this._three;
    this._three.renderer.toneMappingExposure = theme.exposure;
    if (this._grid) {
      const colors = this._grid.geometry.attributes.color;
      const major = new THREE.Color(theme.gridMajor), minor = new THREE.Color(theme.gridMinor);
      // GridHelper stores a colour per vertex: the centre lines get the major colour, every other line the minor one.
      const n = colors.count, lines = n / 4; // two lines (four vertices) per grid step in each direction
      for (let i = 0; i < n; i++) {
        const c = (Math.floor(i / 2) === Math.floor(lines / 2) || Math.floor(i / 2) === Math.floor(lines / 2) + lines) ? major : minor;
        colors.setXYZ(i, c.r, c.g, c.b);
      }
      colors.needsUpdate = true;
      this._grid.material.opacity = theme.gridOpacity;
    }
    for (const mesh of this._three.geomMeshes) {
      if (mesh.userData.plane) mesh.material.opacity = theme.shadow;
      else if (mesh.material.envMapIntensity !== undefined) mesh.material.envMapIntensity = theme.env;
    }
  }

  _meshGeometry(THREE, id) {
    const m = this._model;
    const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id];
    const fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
    const pos = new Float32Array(m.mesh_vert.subarray(3 * va, 3 * (va + vn)));
    const idx = new Uint32Array(m.mesh_face.subarray(3 * fa, 3 * (fa + fn)));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    return geo;
  }

  _applyVisibility() {
    if (!this._three) return;
    for (const mesh of this._three.geomMeshes) {
      if (!mesh.userData.collision) continue;
      mesh.visible = this._showCollision;
      mesh.material.wireframe = true;
      mesh.material.transparent = true;
      mesh.material.opacity = 0.5;
    }
  }

  _syncPoses() {
    if (!this._pose) return;
    const xpos = this._pose.xpos, xmat = this._pose.xmat;
    for (const mesh of this._three.geomMeshes) {
      const g = mesh.userData.geom, p = 3 * g, r = 9 * g;
      mesh.matrix.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1
      );
    }
  }

  _buildJoints() {
    const m = this._model, q = this._pose.qpos;
    const el = this.$(".joints");
    const rows = [];
    this._joints = [];
    const { HINGE, SLIDE, TRN_JOINT } = m.enums;
    for (let j = 0; j < m.njnt; j++) {
      const type = m.jnt_type[j];
      if (type !== HINGE && type !== SLIDE) continue;
      const name = m.jnt_names[j];
      let lo = m.jnt_range[2 * j], hi = m.jnt_range[2 * j + 1];
      // mjtByte arrays (jnt_limited) are not readable in the 3.14 bindings; an unlimited joint has range 0 0.
      if (lo === hi) { lo = -Math.PI; hi = Math.PI; }
      const qadr = m.jnt_qposadr[j];
      let act = -1;
      for (let u = 0; u < m.nu; u++) {
        if (m.actuator_trntype[u] === TRN_JOINT && m.actuator_trnid[2 * u] === j) { act = u; break; }
      }
      this._joints.push({ j, name, lo, hi, qadr, act });
      const v = q[qadr];
      rows.push(`<div class="joint"><label for="j${j}"><span>${name}</span><output id="o${j}">${v.toFixed(2)}</output></label><input id="j${j}" type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 400}" value="${v}" data-j="${j}" aria-label="${name}"></div>`);
    }
    el.innerHTML = `<div class="sheet-head"><h5>${this._joints.length} joints</h5><button class="pill" data-act="close" aria-label="Close joints">Close</button></div><h5>${this._joints.length} joints</h5>${rows.join("")}`;
    el.oninput = (e) => {
      const inp = e.target.closest("input[data-j]");
      if (!inp) return;
      const jt = this._joints.find((x) => x.j === Number(inp.dataset.j));
      const v = Number(inp.value);
      this._pose.qpos[jt.qadr] = v; // optimistic, so the code card follows the finger before the worker answers
      this._engine.setQpos(jt.qadr, v, jt.act);
      this.shadowRoot.getElementById(`o${jt.j}`).textContent = v.toFixed(2);
      this._renderCode();
    };
    this._renderCode();
  }

  _renderCode() {
    const q = this._pose.qpos;
    const moved = this._joints.filter((jt) => Math.abs(q[jt.qadr] - this._qpos0[jt.qadr]) > 1e-3);
    const body = moved.length
      ? moved.map((jt) => `    <b>"${jt.name}"</b>: ${q[jt.qadr].toFixed(3)},`).join("\n")
      : `    <span style="opacity:.55"># move a slider</span>`;
    this.$(".code").innerHTML = `<div class="sheet-head"><h5>robot.act</h5><button class="pill" data-act="close" aria-label="Close code">Close</button></div>from strands_robots import Robot\n\nrobot = Robot(<b>"${this._entry.name}"</b>)\nrobot.act({\n${body}\n})`;
  }

  resetPose() {
    if (!this._engine || !this._pose) return;
    this._poseDirty = true; // sliders and the code card follow the worker's reply
    this._engine.reset();
  }

  _resize() {
    if (!this._three) return;
    const { renderer, camera } = this._three;
    const w = this.clientWidth, h = this.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  _loop() {
    // Physics steps in the worker and arrives as poses; this loop only draws the latest one.
    this._framesDrawn = 0;
    const tick = () => {
      if (this._state !== "ready") return;
      this._three.controls.update();
      this._syncPoses();
      this._three.renderer.render(this._three.scene, this._three.camera);
      if (this._pendingEnvironment && this._framesDrawn++ >= 1) { const build = this._pendingEnvironment; this._pendingEnvironment = null; build(); }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  /** Free every embind handle and GPU resource. Safe to call twice. */
  unload() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this._state = "idle";
    if (this._three) {
      for (const mesh of this._three.geomMeshes) { mesh.geometry.dispose(); mesh.material.dispose(); }
      this._grid?.geometry.dispose(); this._grid?.material.dispose(); this._grid = null;
      this._three.controls.dispose();
      this._three.renderer.dispose();
      this._three = null;
    }
    this._pendingEnvironment = null;
    this._engine?.dispose();
    this._engine = null;
    this._model = null;
    this._pose = null;
    this._poseWaiter = null;
    this._physics = false;
    this._joints = [];
    this.$(".chrome").hidden = true;
    this.$(".joints").hidden = true;
    this.$(".code").hidden = true;
  }
}

if (!customElements.get("robot-viewer")) customElements.define("robot-viewer", RobotViewer);

// Robot picker (index.md): a <select data-robot-pick> next to a viewer lists every streamable
// robot by family; choosing one swaps the viewer's model and loads it (a user gesture, so the
// phone rule that waits for Load 3D is satisfied).
for (const pick of document.querySelectorAll("select[data-robot-pick]")) {
  const viewer = pick.closest(".sr-hero__stage, .sr-hero, body")?.querySelector("robot-viewer");
  if (!viewer) continue;
  loadManifest().then((m) => {
    const families = new Map();
    for (const r of Object.values(m.robots)) {
      if (!r.viewer) continue;
      const fam = r.category || "other";
      if (!families.has(fam)) families.set(fam, []);
      families.get(fam).push(r);
    }
    const current = viewer.getAttribute("name");
    pick.innerHTML = [...families.keys()].sort().map((fam) => {
      const opts = families.get(fam).sort((a, b) => a.name.localeCompare(b.name))
        .map((r) => `<option value="${r.name}"${r.name === current ? " selected" : ""}>${r.name}</option>`).join("");
      return `<optgroup label="${fam}">${opts}</optgroup>`;
    }).join("");
  });
  pick.addEventListener("change", () => {
    viewer.setAttribute("name", pick.value);
    viewer.load();
  });
}

// Catalog filter chips (robots/index.md): .sr-filter button[data-family] toggles .sr-robot[data-family].
document.addEventListener("click", (e) => {
  const b = e.target.closest(".sr-filter button[data-family], .sr-filter-btn[data-family]");
  if (!b) return;
  const fam = b.dataset.family;
  for (const x of b.parentElement.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
  for (const card of document.querySelectorAll(".sr-robot[data-family]")) card.hidden = fam !== "all" && card.dataset.family !== fam;
});
