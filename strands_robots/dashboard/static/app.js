/* The page: one Twin filling the viewport, a mode pill, status chips, a drawer with robot cards + agent chat.
 *   ?scene=<url>   static: draw a scene description, no server (tests/dashboard/fixtures/scene.json)
 *   server         GET /api/scene + WS /ws/state + WS /ws/chat from the Dashboard server (same origin, or
 *                  the reader's own CLI at 127.0.0.1:8090 when this page is served from the docs site)
 *   standalone     no server reachable: the bundled world steps in this browser and an ONNX skill drives it
 *                  (?standalone=<scene.json> picks another bundle). The browser never steps physics while a server
 *                  is connected; a server that answers later wins (re-probe backing off 5 s → 60 s, then a reload).
 */
import { authed, image, signIn, signedOut, socket } from "./auth.js";
import { Standalone } from "./standalone.js";
import { Twin } from "./twin.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
// ?scene= and ?standalone= must be files on this page's own origin, and ?cli= a bare host:port: none may bring
// markup or another site's scene description into the page (a scene's strings end up in the status chips).
const sameOrigin = (raw) => { try { return raw && !/^[a-z][a-z0-9+.-]*:|^[\\/]{2}/i.test(raw) && new URL(raw, location.href).origin === location.origin ? raw : null; } catch { return null; } };
const STATIC = sameOrigin(params.get("scene"));
const CLI_RE = /^(localhost|127\.0\.0\.1|\[::1\]):\d{1,5}$/;
const LOCAL_CLI = CLI_RE.test(params.get("cli") || "") ? params.get("cli") : "127.0.0.1:8090"; // ?cli=host:port — where the reader's CLI listens (tests use a free port)

const state = { mode: "loading", base: "", desc: null, fps: 0, bytes: 0, lastFrame: null, ws: null, picked: null, chips: {} };
const twin = new Twin($("#twin"), { onPick: (robotId) => highlight(robotId) });
window.__twin = twin; // the Playwright gate reads qpos/poses from here
window.__state = state;

/* ---------- chrome ---------- */

function setMode(mode, label) {
  state.mode = mode;
  const pill = $("#mode");
  pill.dataset.mode = mode;
  pill.textContent = label || mode;
  emit("mode", { mode, label });
}
// Every string the server sends (robot ids, names, cameras, policies, chat tool names) is text, never markup:
// an agent chooses some of them (sim.tool add_robot id=, camera name=), and a script in this page could set X-Confirm.
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
function notice(html, bad = false) { const n = $("#notice"); n.innerHTML = html || ""; n.classList.toggle("bad", bad); }
function chip(key, html, cls = "") { state.chips[key] = html ? { html, cls } : null; renderChips(); }
function renderChips() {
  $("#status").innerHTML = Object.entries(state.chips).filter(([, c]) => c).map(([k, c]) => `<span class="chip ${c.cls}" data-chip="${k}">${c.html}</span>`).join("");
}
function highlight(robotId) {
  state.picked = robotId;
  if (robotId && [...$("#run-robot").options].some((o) => o.value === robotId)) { $("#run-robot").value = robotId; renderTasks(state.tasks); }
  for (const el of document.querySelectorAll(".card")) el.classList.toggle("picked", el.dataset.id === robotId);
  if (robotId && matchMedia("(max-width: 56em)").matches && !$("#app").dataset.side) showTab("robots");
}
const small = () => matchMedia("(max-width: 56em)").matches;
/** Open or close the side column (desktop: a grid column; phones: a bottom sheet). It never floats over the chat. */
function showDrawer(on, tab) {
  if (on) delete $("#app").dataset.side; else $("#app").dataset.side = "closed";
  $("#toggle").setAttribute("aria-pressed", String(on));
  if (tab) showTab(tab);
  emit("layout", { open: on });
}
/** Select a tab of the side column: "robots", "chat" (phones), or a registered panel id. */
function showTab(tab) {
  const side = $("#side");
  if (!document.querySelector(`#tabs button[data-tab="${CSS.escape(tab)}"]`)) tab = "robots";
  if (tab === "chat" && !small()) { $("#prompt").focus(); return; } // desktop: the chat is docked, always there
  side.dataset.tab = tab;
  for (const b of document.querySelectorAll("#tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const p of document.querySelectorAll("#panes > .pane")) p.hidden = p.dataset.pane !== tab;
  if ($("#app").dataset.side) showDrawer(true);
  try { sessionStorage.setItem("sr.dashboard.tab", tab); } catch { /* storage blocked */ }
  const panel = SR.panels.get(tab);
  if (panel && !panel.shown) { panel.shown = true; }
  emit("tab", { id: tab });
}
// The stage is its own grid cell now: nothing covers the twin, so no view offset.
function inset() { twin.setInset(0); }
$("#toggle").addEventListener("click", () => showDrawer(Boolean($("#app").dataset.side)));
$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]");
  if (!b) return;
  // phones: the tab row is the sheet's dock; tapping the tab that is already open folds the sheet
  if (small() && !$("#app").dataset.side && $("#side").dataset.tab === b.dataset.tab) { showDrawer(false); return; }
  showTab(b.dataset.tab);
});
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) { e.preventDefault(); showDrawer(true, "chat"); $("#prompt").focus(); }
  else if (e.key === "d" && !typing && !e.metaKey && !e.ctrlKey) showDrawer(Boolean($("#app").dataset.side));
  else if (e.key === "t" && !typing && !e.metaKey && !e.ctrlKey) setScheme(document.documentElement.dataset.scheme === "dark" ? "paper" : "dark", true);
  else if (e.key === "Escape" && typing && !e.shiftKey) document.activeElement.blur();
});

/* Light / dark: the docs site's two schemes (paper · dark). index.html picks one before first paint (?theme= · this
   toggle's own choice · the docs palette · the OS); the toggle stores its choice; the docs palette is followed live
   when the page is embedded there and the reader has not chosen here. */
function setScheme(scheme, remember = false) {
  document.documentElement.dataset.scheme = scheme;
  twin.setScheme?.(scheme);
  if (remember) try { localStorage.setItem("sr.dashboard.scheme", scheme); } catch { /* storage blocked */ }
  emit("scheme", { scheme });
}
// Embedded in the docs hero (another origin, so no shared storage): the docs page posts its palette changes. Only a
// scheme name is read from the message; nothing else in it is used, so any sender can at most flip light/dark.
addEventListener("message", (e) => {
  const m = e.data;
  if (window.top === window || !m || m.type !== "sr:scheme" || (m.scheme !== "paper" && m.scheme !== "dark")) return;
  setScheme(m.scheme);
});
$("#theme").addEventListener("click", () => setScheme(document.documentElement.dataset.scheme === "dark" ? "paper" : "dark", true));
twin.setScheme?.(document.documentElement.dataset.scheme);
addEventListener("storage", (e) => {
  if (!e.key || !e.key.endsWith("__palette") || params.get("theme")) return;
  try { if (localStorage.getItem("sr.dashboard.scheme")) return; const m = JSON.parse(e.newValue || "null")?.color?.scheme; if (m) setScheme(m === "slate" ? "dark" : m); } catch { /* ignore */ }
});

if (window.top !== window) $(".brand").target = "_blank"; // embedded: the wordmark opens the full page, not a nested one
// Embedded: tell the parent "sr:hello" now and "sr:painted" once the twin has drawn a frame with the robots in view, so
// the docs hero keeps its recording until then. Only these two names are sent (no state), to the embedding docs'
// origin when ?docs= names it, else to any parent.
if (window.top !== window) {
  const to = (() => { try { return new URL(params.get("docs") || "").origin; } catch { return "*"; } })();
  const say = (type) => { try { window.parent.postMessage({ type }, to); } catch { /* parent gone */ } };
  document.documentElement.dataset.embed = "yes";
  say("sr:hello");
  const painted = () => {
    if (!(twin.framed && twin.pose)) { requestAnimationFrame(painted); return; }
    requestAnimationFrame(() => requestAnimationFrame(() => { say("sr:painted"); window.__painted = true; })); // drawn, then shown
  };
  requestAnimationFrame(painted);
}

/* Docs: the public site, or the reader's local docs when this page is embedded in them (the hero) or ?docs= names one. */
(() => {
  const local = (u) => { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x : null; } catch { return null; } };
  const explicit = params.get("docs") && local(params.get("docs"));
  const ref = window.top !== window && document.referrer ? local(document.referrer) : null;
  const base = explicit || (ref && ["127.0.0.1", "localhost"].includes(ref.hostname) ? new URL("/", ref) : null);
  if (base) $("#docs-link").href = new URL("dashboard/", base.href.endsWith("/") ? base : `${base.href}/`).href;
})();

/* ---------- panel slot API ----------
 * Other front-end modules add a tab to the side column without editing this file:
 *   window.SR.registerPanel({ id, title, icon?, order?, badge?, mount(el, ctx) })
 * mount runs once, with a fresh <section> and ctx = { state, base, esc, log, chip, notice, on, api, robots, select }.
 * Server mode also imports every module listed by GET /api/panels (static/panels/*.js), so a feature module only adds a file.
 */
const listeners = new Map(); // event -> Set(fn)
function emit(ev, data) { for (const fn of listeners.get(ev) || []) { try { fn(data); } catch (err) { console.error(`SR ${ev} listener`, err); } } }
function on(ev, fn) { if (!listeners.has(ev)) listeners.set(ev, new Set()); listeners.get(ev).add(fn); return () => listeners.get(ev).delete(fn); }
const SR = window.SR = Object.assign(window.SR || {}, {
  version: 1,
  panels: new Map(),
  on, emit,
  registerPanel(spec) {
    const { id, title, icon = "", order = 100, mount } = spec || {};
    if (!/^[a-z][a-z0-9-]{0,31}$/.test(id || "")) throw new Error(`SR.registerPanel: id must be a short lowercase slug, got ${id}`);
    if (id === "robots" || id === "chat" || SR.panels.has(id)) throw new Error(`SR.registerPanel: ${id} is taken`);
    if (typeof mount !== "function") throw new Error(`SR.registerPanel(${id}): mount(el, ctx) is required`);
    const tab = document.createElement("button");
    tab.setAttribute("role", "tab"); tab.dataset.tab = id; tab.dataset.order = String(order); tab.setAttribute("aria-selected", "false");
    tab.textContent = `${icon ? `${icon} ` : ""}${title || id}`;
    const badge = document.createElement("span"); badge.className = "badge"; tab.appendChild(badge);
    const before = [...document.querySelectorAll("#tabs button[data-tab]")].find((b) => b.dataset.tab !== "robots" && (b.dataset.tab === "chat" || +b.dataset.order > order));
    $("#tabs").insertBefore(tab, before || null);
    const el = document.createElement("section");
    el.className = "pane"; el.dataset.pane = id; el.dataset.panel = id; el.setAttribute("role", "tabpanel"); el.hidden = true;
    $("#panes").appendChild(el);
    const panel = { ...spec, el, tab, setBadge: (v) => { badge.textContent = v === null || v === undefined ? "" : String(v); } };
    SR.panels.set(id, panel);
    try { mount(el, ctx(panel)); } catch (err) { el.textContent = `panel ${id} failed: ${err.message}`; console.error(err); }
    let want = null; try { want = sessionStorage.getItem("sr.dashboard.tab"); } catch { /* storage blocked */ }
    if (want === id) showTab(id);
    emit("panel", { id });
    return panel;
  },
  /** A fixed slot outside the tabs: "chat-input" (a row right above the chat input, e.g. a mic) or "bar" (the top
   *  bar, left of Docs, e.g. a REC dot). mount(el, ctx) gets an empty element; it may also return an element to append. */
  registerSlot(slot, mount, { id = slot } = {}) {
    const host = { "chat-input": $("#chat-slot"), bar: $("#bar-slot") }[slot];
    if (!host) throw new Error(`SR.registerSlot: no slot ${slot} (chat-input · bar)`);
    if (typeof mount !== "function") throw new Error(`SR.registerSlot(${slot}): mount(el, ctx) is required`);
    const el = document.createElement("div");
    el.className = "slot"; el.dataset.slot = slot; el.dataset.panel = id;
    host.appendChild(el); host.hidden = false;
    const panel = { id, el, setBadge: () => {} };
    try { const out = mount(el, ctx(panel)); if (out instanceof Element) el.appendChild(out); } catch (err) { el.textContent = `${id} failed: ${err.message}`; console.error(err); }
    emit("panel", { id, slot });
    return panel;
  },
  select: (id) => showTab(id),
  open: (on_ = true) => showDrawer(on_),
});
// Alias names for the same two functions.
window.srPanels = { register: (spec) => SR.registerPanel(spec) };
window.strandsSlots = { register: (slot, mount) => SR.registerSlot(slot, mount) };
function ctx(panel) {
  return {
    get state() { return state; },
    get base() { return state.base; },
    get robots() { return state.desc?.robots || []; },
    esc, log, chip, notice, on,
    setBadge: panel.setBadge,
    select: () => showTab(panel.id),
    /** fetch against this dashboard; JSON in, JSON out; a POST is always application/json (the guard requires it). */
    api: async (path, { method = "GET", body, confirm: confirmed = false } = {}) => {
      const headers = { "content-type": "application/json" };
      if (confirmed) headers["X-Confirm"] = "yes";
      const res = await authed(`${state.base}${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
      const text = await res.text();
      let doc = null; try { doc = text ? JSON.parse(text) : null; } catch { doc = { detail: text }; }
      if (res.status === 401) signedOut(); // the session ended: back to the passkey screen
      if (!res.ok) { const err = new Error(doc?.detail || `HTTP ${res.status}`); err.status = res.status; err.doc = doc; throw err; }
      return doc;
    },
  };
}
/** Server mode: import every panel module the server lists (static/panels/*.js). A missing route is a quiet no-op. */
async function loadPanels() {
  const doc = await authed(`${state.base}/api/panels`).then((r) => (r.ok ? r.json() : null), () => null);
  for (const name of doc?.panels || []) {
    if (!/^(panels\/)?[a-z][a-z0-9_-]*\.js$/.test(name)) continue;
    try { await import(`${state.base || "."}/${name}`); /* "panels/x.js" or a feature module next to app.js */ } catch (err) { console.error(`panel ${name}`, err); chip(`panel-${name}`, `panel ${esc(name)} failed`, "warn"); }
  }
  window.__panelsLoaded = true; // modules may register later (after their own probe): read SR.panels / window.__panels
}
Object.defineProperty(window, "__panels", { get: () => [...SR.panels.keys()] });
if (small()) showDrawer(false); else inset();

/* ---------- cards ---------- */

function fmt(v) { return (Math.abs(v) < 1e-9 ? 0 : v).toFixed(3); }

const CHEV = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 9l5 6 5-6z"/></svg>';
const collapsed = (() => { try { return new Set(JSON.parse(localStorage.getItem("sr.dashboard.collapsed") || "[]")); } catch { return new Set(); } })();
function setCollapsed(card, on) {
  card.dataset.collapsed = on ? "yes" : "no";
  card.querySelector(".chev")?.setAttribute("aria-expanded", String(!on));
  if (on) collapsed.add(card.dataset.key); else collapsed.delete(card.dataset.key);
  try { localStorage.setItem("sr.dashboard.collapsed", JSON.stringify([...collapsed])); } catch { /* storage blocked */ }
}
/** One card per robot. The header carries the robot's own E-STOP / Resume slot (.estop-ctl, painted by safety.js
 *  from the state frames) and a collapse chevron; the body holds one readable slider per joint. */
function renderCards(desc) {
  const root = $("#cards");
  root.innerHTML = "";
  for (const r of desc.robots) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.id = r.id;
    card.dataset.key = `robot:${r.id}`;
    const tag = r.is_twin_of ? `<span class="tag real">twin of ${esc(r.is_twin_of)}</span>` : r.is_real ? '<span class="tag real">real</span>' : '<span class="tag sim">sim</span>';
    card.innerHTML = `<h3><button class="chev" type="button" aria-label="show or hide ${esc(r.id)}'s joints" aria-expanded="true">${CHEV}</button>` +
      `<span class="name"><span>${esc(r.id)}</span><small>${esc(r.name)}</small></span>${tag}` +
      `<span class="estop-ctl" data-robot="${esc(r.id)}" data-latched="no"><span class="estop-status">armed</span>` +
      `<button class="estop" type="button" title="E-stop ${esc(r.id)}: stops it now, no confirm">E-STOP</button>` +
      `<button class="resume" type="button" title="release the latch (needs an operator name)">Resume</button></span></h3><div class="body"></div>`;
    const body = card.querySelector(".body");
    r.joints.forEach((jname, i) => {
      const label = (r.labels && r.labels[i]) || jname.split("/").pop();
      const lim = (r.limits && r.limits[label]) || [-3.14, 3.14];
      const row = document.createElement("div");
      row.className = "joint";
      row.innerHTML = `<span title="${esc(jname)}">${esc(label)}</span><span class="val" data-joint="${esc(jname)}">–</span>` +
        `<input type="range" min="${esc(lim[0])}" max="${esc(lim[1])}" step="0.001" aria-label="${esc(r.id)} ${esc(label)}" data-robot="${esc(r.id)}" data-label="${esc(label)}" ${state.mode === "server" ? "" : "disabled"}>`;
      body.appendChild(row);
    });
    card.querySelector("h3").addEventListener("click", (e) => {
      if (e.target.closest(".estop-ctl")) return; // the e-stop buttons are their own click
      if (e.target.closest(".chev")) { setCollapsed(card, card.dataset.collapsed !== "yes"); return; }
      highlight(r.id);
    });
    body.addEventListener("click", () => highlight(r.id));
    setCollapsed(card, collapsed.has(card.dataset.key));
    root.appendChild(card);
  }
  for (const c of desc.cameras || []) {  // what a policy sees: the world's own render, refreshed once a second
    const card = document.createElement("div");
    card.className = "card cam";
    card.dataset.key = `camera:${c.name}`;
    card.innerHTML = `<h3><button class="chev" type="button" aria-label="show or hide camera ${esc(c.name)}" aria-expanded="true">${CHEV}</button>` +
      `<span class="name"><span>${esc(c.name)}</span><small>camera ${esc(c.size[0])}x${esc(c.size[1])}</small></span><span class="tag sim">eyes</span></h3>` +
      `<div class="body"><img alt="camera ${esc(c.name)}" data-src="${esc(state.base)}${esc(c.url)}"></div>`;
    card.querySelector("h3").addEventListener("click", () => setCollapsed(card, card.dataset.collapsed !== "yes"));
    setCollapsed(card, collapsed.has(card.dataset.key));
    root.appendChild(card);
  }
  $("#robot-count").textContent = desc.robots.length ? String(desc.robots.length) : "";
  refreshCams();
  if (!root.__wired) { root.addEventListener("change", onSlider); root.__wired = true; setInterval(refreshCams, 1000); }
  emit("scene", desc);
}

function refreshCams() {
  if (state.mode !== "server" || document.hidden) return;
  for (const img of document.querySelectorAll('.card.cam:not([data-collapsed="yes"]) img')) image(img, `${img.dataset.src}?t=${Date.now()}`);
}

async function onSlider(e) {
  const el = e.target;
  if (el.tagName !== "INPUT" || el.type !== "range") return;
  const rid = el.dataset.robot;
  const robot = state.desc.robots.find((r) => r.id === rid);
  const headers = { "content-type": "application/json" };
  if (robot.is_real) {
    if (!confirm(`${rid} is a REAL robot. Move ${el.dataset.label} to ${fmt(+el.value)} rad?`)) return;
    headers["X-Confirm"] = "yes";
  }
  const res = await authed(`${state.base}/api/robots/${encodeURIComponent(rid)}/act`, { method: "POST", headers, body: JSON.stringify({ targets: { [el.dataset.label]: +el.value } }) });
  window.__acts = (window.__acts || []).concat([{ rid, label: el.dataset.label, value: +el.value, status: res.status }]);
  if (!res.ok) log(`act ${rid}: HTTP ${res.status} ${await res.text()}`, "err");
}

function applyFrame(frame) {
  // {t, revision, bodies: {id: {q: {name: val}, pose: [x y z qw qx qy qz] | null, root}}}
  const q = {}, poses = {};
  for (const id in frame.bodies) {
    const b = frame.bodies[id];
    Object.assign(q, b.q);
    if (b.pose && b.root) poses[b.root] = b.pose;
  }
  twin.setQposByName({ q, poses });
  for (const el of document.querySelectorAll(".val[data-joint]")) {
    const v = q[el.dataset.joint];
    if (v !== undefined) el.textContent = fmt(v);
  }
  if (frame.tasks !== undefined && JSON.stringify(frame.tasks) !== JSON.stringify(state.tasks || {})) renderTasks(frame.tasks);
  state.lastFrame = frame;
  emit("frame", frame);
}

/* ---------- run a policy · add a robot (server mode) ----------
 * Needs on the server (DOCS-LOG "Needs from features", exact shapes): GET /api/policies, POST /api/robots/{id}/run,
 * POST /api/robots/{id}/stop, GET /api/registry, POST /api/robots. A 404 on /api/policies hides the whole row and says
 * so in a chip — nothing here is faked when the server cannot do it.
 */
async function loadPickers() {
  if (state.mode !== "server") return;
  const get = (p) => authed(`${state.base}${p}`).then((r) => (r.ok ? r.json() : null), () => null);
  const pol = await get("/api/policies");
  state.policies = pol;
  $("#run").hidden = !pol; $("#add").hidden = !pol; // POST /api/robots landed together with /api/policies
  chip("norun", pol ? "" : "this server has no /api/policies yet — update the package to run policies from here", "opt");
  if (!pol) return;
  // {policies: [{name, kind: builtin|onnx|lerobot|remote, robots, ready, hz, source, description}]}
  $("#run-policy").innerHTML = pol.policies.filter((p) => p.kind !== "remote").map((p) =>
    `<option value="${esc(p.name)}" ${p.ready === false ? "disabled" : ""} title="${esc(p.description || "")}">${esc(p.name)} · ${esc(p.kind)}${p.ready === false ? " · extra missing" : ""}</option>`).join("") +
    `<option value="__text">from the text field → (ws://host:port · lerobot/repo · file.onnx)</option>`;
  const names = new Set(["so101", "g1", "microduck", "franka", "go2", "spot"]);
  for (const p of pol.policies) for (const r of p.robots || []) names.add(r);
  $("#add-names").innerHTML = [...names].map((n) => `<option value="${esc(n)}">`).join("");
  fillRunRobots();
  pollTasks();
}
function fillRunRobots() {
  const sel = $("#run-robot"); const was = sel.value;
  sel.innerHTML = (state.desc?.robots || []).map((r) => `<option value="${esc(r.id)}">${esc(r.id)}${r.is_real ? " · REAL" : ""}</option>`).join("");
  if ([...sel.options].some((o) => o.value === was)) sel.value = was;
  else if (state.picked && [...sel.options].some((o) => o.value === state.picked)) sel.value = state.picked;
}
/** Task state: GET /api/robots/{id}/run per robot every 2 s (a frame field `tasks` is used instead when the server sends one). */
async function pollTasks() {
  clearTimeout(state.pollTimer);
  if (state.mode !== "server" || !state.policies || state.lastFrame?.tasks) return;
  const tasks = {};
  await Promise.all((state.desc?.robots || []).map(async (r) => {
    const st = await authed(`${state.base}/api/robots/${encodeURIComponent(r.id)}/run`).then((x) => (x.ok ? x.json() : null), () => null);
    if (st && st.status === "running") tasks[r.id] = { policy: state.runPolicy?.[r.id] || "policy", status: st.status };
    else if (st && st.status === "error") tasks[r.id] = null, state.taskErrors = { ...(state.taskErrors || {}), [r.id]: st.error };
  }));
  renderTasks(tasks);
  window.__tasks = tasks;
  followRunning(tasks);
  state.pollTimer = setTimeout(pollTasks, 2000);
}

/** Server mode: the camera walks with the one robot whose task is running (C-D23); none running → stay put. */
function followRunning(tasks) {
  const running = Object.keys(tasks).filter((id) => tasks[id]);
  const id = running.length === 1 ? running[0] : null;
  const root = id && state.lastFrame?.bodies?.[id]?.root;
  twin.follow = root ? `${root}` : null;
}
async function runOrStop() {
  const rid = $("#run-robot").value, robot = state.desc.robots.find((r) => r.id === rid);
  const running = state.tasks?.[rid];
  const headers = { "content-type": "application/json" };
  if (running) {
    const res = await authed(`${state.base}/api/robots/${encodeURIComponent(rid)}/stop`, { method: "POST", headers });
    log(`stop ${rid}: ${res.status} ${res.ok ? JSON.stringify(await res.json()) : await res.text()}`, res.ok ? "tool" : "err");
    pollTasks();
    return;
  }
  const picked = $("#run-policy").value, text = $("#run-text").value.trim();
  // text field: the whole spec (ws://… · lerobot/repo · file.onnx) · for a preset a query (vx=0.3&wz=0.2) · else the instruction
  const isQuery = /^[a-z_]+=[^\s]+(&[a-z_]+=[^\s]+)*$/.test(text);
  const policy = picked === "__text" ? text : isQuery ? `${picked}?${text}` : picked;
  const instruction = picked === "__text" || isQuery ? null : text || null;
  if (!policy) { log("pick a policy or type ws://… / lerobot/<repo>", "err"); return; }
  if (robot.is_real) {
    if (!confirm(`${rid} is a REAL robot. Run ${policy} on it?`)) return;
    headers["X-Confirm"] = "yes";
  }
  const body = { policy, instruction, seconds: null, hz: null }; // the server paces to wall-clock and picks the policy's hz
  const res = await authed(`${state.base}/api/robots/${encodeURIComponent(rid)}/run`, { method: "POST", headers, body: JSON.stringify(body) });
  window.__runs = (window.__runs || []).concat([{ rid, policy, status: res.status }]);
  if (res.ok) { state.runPolicy = { ...(state.runPolicy || {}), [rid]: policy }; }
  log(`run ${policy} on ${rid}: ${res.status} ${res.ok ? JSON.stringify(await res.json()) : await res.text()}`, res.ok ? "tool" : "err");
  pollTasks();
}
async function addRobot() {
  const name = $("#add-robot").value.trim();
  if (!name) return;
  const res = await authed(`${state.base}/api/robots`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, position: [(state.desc?.robots.length || 0) * 1.0, 0, 0], id: null }) });
  window.__adds = (window.__adds || []).concat([{ name, status: res.status }]);
  log(`add ${name}: ${res.status} ${res.ok ? JSON.stringify(await res.json()) : await res.text()}`, res.ok ? "tool" : "err");
  if (res.ok) $("#add-robot").value = "";
}
function renderTasks(tasks) {
  // tasks: {robotId: {policy, steps}} from the state frame → card tag + run button label
  state.tasks = tasks || {};
  for (const card of document.querySelectorAll(".card")) {
    const t = state.tasks[card.dataset.id];
    let tag = card.querySelector(".running");
    if (t && !tag) { tag = document.createElement("span"); tag.className = "running"; card.querySelector("h3").insertBefore(tag, card.querySelector(".estop-ctl")); }
    if (tag) { if (t) tag.textContent = `▸ ${t.policy}`; else tag.remove(); }
  }
  const rid = $("#run-robot").value, go = $("#run-go");
  go.dataset.running = String(Boolean(state.tasks[rid]));
  go.textContent = state.tasks[rid] ? "■ stop" : "run ▸";
}
$("#run-go").addEventListener("click", runOrStop);
$("#add-go").addEventListener("click", addRobot);
$("#run-robot").addEventListener("change", () => renderTasks(state.tasks));

/* ---------- status ---------- */

function statusLine() {
  const d = state.desc;
  if (d) chip("phys", `<b>${esc(d.physics.name)}</b> dt=${esc(d.physics.timestep)}`);
  const t = state.lastFrame ? state.lastFrame.t : d ? d.time : null;
  chip("t", t === null ? "" : `t=<b>${t.toFixed(2)}</b>s`);
  if (state.mode === "server") chip("net", `<b>${state.fps}</b> fps · ${(state.bytes / 1024).toFixed(1)} KB/s`);
  chip("model", twin.model ? `<b>${twin.model.ngeom}</b> geoms · ${twin.model.njnt} joints · compiled ${Math.round(twin.compileMs || 0)} ms` : "", "opt");
  chip("rev", twin.revision !== null ? `rev ${esc(twin.revision)}` : "", "opt");
  chip("unres", twin.unresolved ? `${twin.unresolved} unresolved names` : "", "warn");
}

/* ---------- scene + server ---------- */

async function loadScene() {
  chip("compile", "compiling…");
  const desc = await twin.load(`${state.base}/api/scene`);
  chip("compile", "");
  state.desc = desc;
  renderCards(desc);
  if (state.policies) { fillRunRobots(); pollTasks(); } else loadPickers();
  renderTasks(state.tasks);
  statusLine();
  window.__sceneLoaded = (window.__sceneLoaded || 0) + 1;
  return desc;
}

async function loadStatic(url) {
  chip("compile", "compiling…");
  const desc = await twin.load(url);
  chip("compile", "");
  state.desc = desc;
  renderCards(desc);
  statusLine();
  window.__sceneLoaded = (window.__sceneLoaded || 0) + 1;
  return desc;
}

function wsUrl(path) {
  if (state.base) return state.base.replace(/^http/, "ws") + path;
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
}

function connectState() {
  const ws = socket(wsUrl("/ws/state"));
  state.ws = ws;
  let frames = 0, bytes = 0;
  const tick = setInterval(() => { state.fps = frames; state.bytes = bytes; frames = 0; bytes = 0; statusLine(); }, 1000);
  ws.onmessage = async (e) => {
    bytes += e.data.length; frames++;
    const frame = JSON.parse(e.data);
    state.latestRevision = frame.revision;
    if (twin.revision !== null && frame.revision !== twin.revision && !state.reloading) {
      state.reloading = true;
      try {
        // Reload until the compiled model matches the newest revision seen (bumps can arrive mid-compile).
        while (state.latestRevision !== twin.revision) await loadScene();
      } finally { state.reloading = false; }
    }
    if (frame.revision === twin.revision) applyFrame(frame);
  };
  ws.onclose = () => { clearInterval(tick); signedOut(); setMode("error", "server gone"); chip("net", "state socket closed — reconnecting…", "bad"); setTimeout(connectState, 1000); };
  ws.onopen = () => setMode("server", state.base ? `server · ${LOCAL_CLI}` : "server");
}

/* ---------- chat ---------- */

function log(text, cls = "") {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  $("#log").appendChild(el);
  $("#log").scrollTop = 1e9;
  return el;
}

function connectChat() {
  let ws = null, current = null;
  authed(`${state.base}/api/chat`).then((r) => (r.ok ? r.json() : null), () => null).then((info) => {
    if (!info || info.enabled) return; // an older server without /api/chat: the first send reports it
    $("#chatform").querySelector("button").disabled = true;
    $("#prompt").placeholder = "chat is off · start with --model <bedrock model id>";
    const el = log("", "sys off");
    el.innerHTML = `<b>Chat is off.</b> Start the dashboard with a model:<code>python -m strands_robots.dashboard --model &lt;bedrock model id&gt;</code>or pass <code>Dashboard(sim, model=...)</code>`;
  });
  const open = () => new Promise((resolve, reject) => {
    ws = socket(wsUrl("/ws/chat"));
    ws.onopen = () => resolve(ws);
    ws.onerror = reject;
    ws.onclose = () => { ws = null; };
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      window.__chatEvents = (window.__chatEvents || []).concat([ev]);
      if (ev.type === "text") { if (!current) current = log("", "assistant"); current.textContent += ev.delta; }
      else if (ev.type === "tool_use") { current = null; const d = log("", "tool"); d.innerHTML = `<details><summary>⚙ ${esc(ev.name)}</summary><pre></pre></details>`; d.querySelector("pre").textContent = JSON.stringify(ev.input, null, 1); }
      else if (ev.type === "tool_result") { current = null; const d = log("", "tool"); d.innerHTML = `<details><summary>↳ ${esc(ev.name)} · ${esc(ev.status)}</summary><pre></pre></details>`; d.querySelector("pre").textContent = ev.text || ""; }
      else if (ev.type === "error") { current = null; log(ev.message, "err"); }
      else if (ev.type === "done") { current = null; }
      $("#log").scrollTop = 1e9;
    };
  });
  $("#chatform").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#prompt").value.trim();
    if (!text) return;
    $("#prompt").value = "";
    log(text, "user");
    current = null;
    if (!ws) await open();
    ws.send(JSON.stringify({ prompt: text }));
  });
}

/* ---------- standalone ---------- */

const STANDALONE_SCENE = sameOrigin(params.get("standalone")) || "./scenes/microduck/scene.json"; // like ?scene=
const CMD = { vx: 0.3, back: -0.2, wz: 1.0 };

async function standalone(why) {
  const desc = await loadStatic(STANDALONE_SCENE);
  chip("compile", "loading policy…");
  const t0 = performance.now();
  const sa = new Standalone(twin, desc, STANDALONE_SCENE, { onStats: onStandalone });
  const ready = await sa.start();
  twin.follow = desc.standalone.root; // the camera walks with the body
  chip("compile", "");
  state.standalone = sa;
  window.__standalone = sa; // the browser gates read stats, ready and errors here
  sa.loadMs = performance.now() - t0;
  policyChip(ready);
  notice(`<b>standalone</b> — physics in your browser (MuJoCo WASM, ${esc(desc.physics.timestep * 1000)} ms × ${esc(desc.physics.substeps)}; ` +
    `the ONNX skill on onnxruntime-web at ${esc(desc.standalone.hz)} Hz). ${why} <span class="keys">W A S D walk · R reset</span>`);
  const pick = $("#skill");
  pick.innerHTML = sa.skills.map((k) => `<option value="${esc(k)}"${k === sa.skill ? " selected" : ""}>${esc(k)}</option>`).join("");
  pick.addEventListener("change", async () => { pick.disabled = true; try { policyChip(await sa.use(pick.value)); } finally { pick.disabled = false; } });
  $("#pad").hidden = false;
  wireCommand(sa);
}

function policyChip(ready) {
  chip("policy", `<b>${esc(ready.skill)}</b> · ${(ready.modelBytes / 1024).toFixed(0)} KB · loaded ${Math.round(ready.loadMs)} ms`, "opt");
}

function onStandalone(m) {
  if (m.stats) {
    chip("sim", `sim <b>${m.stats.hz.toFixed(0)}</b> Hz · policy <b>${m.stats.policyMs.toFixed(1)}</b> ms · <b>${twin.fps ?? "–"}</b> fps · z ${m.stats.z.toFixed(3)} m` + (m.stats.falls ? ` · ${m.stats.falls} falls` : ""));
    chip("t", `t=<b>${m.stats.t.toFixed(1)}</b>s`);
    for (const el of document.querySelectorAll(".val[data-joint]")) { const v = twin.qpos(el.dataset.joint); if (v !== undefined && v !== null) el.textContent = fmt(v); }
  }
  if (m.fell) chip("fall", `fell · reset ${m.fell}`, "warn");
  if (m.error) { chip("sim", esc(m.error), "bad"); notice(`<b>standalone</b> — ${esc(m.error)}. The twin still draws the last pose; physics is off.`, true); }
}

function wireCommand(sa) {
  const held = new Set();
  const send = () => {
    const vx = held.has("w") ? CMD.vx : held.has("s") ? CMD.back : 0;
    const wz = held.has("a") ? CMD.wz : held.has("d") ? -CMD.wz : 0;
    sa.command(vx, 0, wz);
    for (const b of document.querySelectorAll("#pad button")) b.setAttribute("aria-pressed", String(held.has(b.dataset.key)));
  };
  document.addEventListener("keydown", (e) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "")) return;
    const k = e.key.toLowerCase();
    if ("wasd".includes(k) && k.length === 1) { held.add(k); send(); e.preventDefault(); }
    else if (k === "r") twin.engine.reset();
  });
  document.addEventListener("keyup", (e) => { if (held.delete(e.key.toLowerCase())) send(); });
  for (const b of document.querySelectorAll("#pad button")) {
    const down = (e) => { e.preventDefault(); held.add(b.dataset.key); send(); };
    const up = () => { if (held.delete(b.dataset.key)) send(); };
    b.addEventListener("pointerdown", down); b.addEventListener("pointerup", up); b.addEventListener("pointercancel", up); b.addEventListener("pointerleave", up);
  }
}

/* ---------- mode probe ---------- */

/** Is a Dashboard server reachable? Same origin first (the CLI serves this page), then the reader's own CLI.
 *  The server registers GET only (HEAD answers 404), so the probe is a GET of the scene description.
 *  From an https site, Chromium's Local Network Access check gates 127.0.0.1 behind a permission
 *  (ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS, measured on the live site): a fetch with
 *  targetAddressSpace:"loopback" is what may raise that prompt; the WebSocket cannot ask, so it goes second. */
async function probe() {
  const same = await authed("/api/scene").then((r) => r.ok, () => false);
  if (same) return { base: "" };
  if (location.host === LOCAL_CLI) return null;
  const base = `http://${LOCAL_CLI}`;
  const read = () => fetch(`${base}/api/scene`, location.protocol === "https:" ? { targetAddressSpace: "loopback" } : {}).then((r) => r.ok, () => false);
  if (location.protocol === "https:" && (await read())) return { base };
  const ok = await new Promise((resolve) => {
    let done = false;
    const ws = new WebSocket(`ws://${LOCAL_CLI}/ws/state`);
    const finish = (v) => { if (!done) { done = true; resolve(v); try { ws.close(); } catch { /* closed */ } } };
    ws.onopen = () => finish(true); ws.onerror = () => finish(false);
    setTimeout(() => finish(false), 1000);
  });
  if (!ok) return location.protocol === "https:" ? { base, unreachable: true } : null;
  // The socket answered; the page is on another origin, so /api must allow it (CORS) or the scene cannot be read.
  return (await read()) ? { base } : { base, blocked: true };
}

(async () => {
  try {
    if (STATIC) {
      setMode("static", "static scene");
      $("#chatform").querySelector("button").disabled = true;
      $("#prompt").placeholder = "no agent in a static scene";
      await loadStatic(STATIC);
      notice(`static scene · <b>${twin.model.ngeom}</b> geoms · ${twin.model.njnt} joints · ${(twin.assetBytes / 1048576).toFixed(1)} MB assets · compiled in ${Math.round(twin.compileMs)} ms`);
      return;
    }
    if (!(await signIn())) { setMode("server", "sign in"); return; } // sign-in on and no session: the passkey screen
    const server = await probe();
    if (server && !server.blocked && !server.unreachable) {
      state.base = server.base;
      setMode("server", server.base ? `server · ${LOCAL_CLI}` : "server");
      await loadScene();
      connectState();
      connectChat();
      loadPanels();
      return;
    }
    // No server anywhere: standalone, said on screen. A server that answers later wins: re-probe (5 s, doubling to 60 s), stop the
    // browser's physics and reload into server mode, so nothing from the standalone world survives the flip.
    $("#chatform").querySelector("button").disabled = true;
    $("#prompt").placeholder = "start strands-robots dashboard to chat with an agent";
    const why = server && server.blocked
      ? `A <code>strands-robots dashboard</code> answers on <code>${LOCAL_CLI}</code> but does not allow this origin (<code>${location.origin}</code>) to read <code>/api/scene</code>: open <a href="http://${LOCAL_CLI}/">${LOCAL_CLI}</a> directly.`
      : `Start <code>strands-robots dashboard</code> to drive real worlds.`;
    if (server && server.blocked) window.__blocked = true;
    const reprobe = (wait) => { state.reprobe = setTimeout(async () => {
      const again = await probe().catch(() => null);
      if (again && !again.blocked && !again.unreachable) { state.standalone?.stop(); location.reload(); return; }
      reprobe(Math.min(2 * wait, 60_000)); // each failed probe is a browser console error: back off 5 s → 60 s
    }, wait); };
    reprobe(5000);
    const bundled = await fetch(STANDALONE_SCENE, { method: "HEAD" }).then((r) => r.ok, () => false);
    if (!bundled) {
      setMode("offline", "no server");
      notice(`<b>no world connected</b> — ${why} (No standalone bundle here: <code>python -m strands_robots.dashboard.standalone</code> writes one.)`);
      return;
    }
    setMode("standalone", "standalone");
    notice(`<b>standalone</b> — physics in your browser; ${why}`);
    await standalone(why);
  } catch (err) {
    setMode("error", "error");
    notice(esc(err.message), true);
    window.__error = String(err.message || err);
    throw err;
  }
})();
