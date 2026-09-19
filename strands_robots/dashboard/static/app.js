/* strands robots dashboard - app shell. ES modules, no bundler.
   Every request goes through api(); a 401 anywhere routes to the login view. */

import { Twin } from "./twin.js";

const $ = (sel, root = document) => root.querySelector(sel);
const views = ["fleet", "sim", "agent", "settings"];
let authenticated = false;

export async function api(path, init = {}) {
  const res = await fetch(path, { credentials: "same-origin", headers: { "content-type": "application/json", ...(init.headers || {}) }, ...init });
  if (res.status === 401) { showLogin(); throw new Error("sign in required"); }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.error || res.statusText);
  return body;
}

/* ---- WebAuthn helpers: py_webauthn hands out base64url, the browser wants ArrayBuffers ---- */
const b64uToBuf = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(s.length / 4) * 4, "=")), c => c.charCodeAt(0)).buffer;
const bufToB64u = (b) => btoa(String.fromCharCode(...new Uint8Array(b))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

function toCreationOptions(o) {
  return { ...o, challenge: b64uToBuf(o.challenge), user: { ...o.user, id: b64uToBuf(o.user.id) },
    excludeCredentials: (o.excludeCredentials || []).map(c => ({ ...c, id: b64uToBuf(c.id) })) };
}
function toRequestOptions(o) {
  return { ...o, challenge: b64uToBuf(o.challenge), allowCredentials: (o.allowCredentials || []).map(c => ({ ...c, id: b64uToBuf(c.id) })) };
}
function serializeCredential(cred) {
  const r = cred.response;
  const out = { id: cred.id, rawId: bufToB64u(cred.rawId), type: cred.type, response: { clientDataJSON: bufToB64u(r.clientDataJSON) } };
  if (r.attestationObject) out.response.attestationObject = bufToB64u(r.attestationObject);
  if (r.authenticatorData) out.response.authenticatorData = bufToB64u(r.authenticatorData);
  if (r.signature) out.response.signature = bufToB64u(r.signature);
  if (r.userHandle) out.response.userHandle = bufToB64u(r.userHandle);
  if (cred.getClientExtensionResults) out.clientExtensionResults = cred.getClientExtensionResults();
  return out;
}

/* ---- views ---- */
/* Data never becomes markup. A view builds its nodes and hands each one its text
   as text, so a string a route serves - a registry description, a mesh peer's
   name, a refusal - is a text node and not a tag. Script in this page would be
   same-origin: it rides the session cookie and passes origin_is_self by
   construction, so every guard in access.py is behind this one. */
const node = (tag, className, text) => {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
};

function show(view) {
  for (const v of [...views, "login"]) $(`#view-${v}`).hidden = v !== view;
  for (const b of $("#tabs").children) b.classList.toggle("on", b.dataset.view === view);
  if (view !== "login") location.hash = view;
}

function showLogin() { authenticated = false; $("#who").textContent = "signed out"; show("login"); }

async function loadStatus() {
  const s = await fetch("/api/auth/status", { credentials: "same-origin" }).then(r => r.json());
  const hint = $("#login-hint"), enrol = $("#enrol"), login = $("#login");
  enrol.hidden = true; login.hidden = true;
  if (s.warning) hint.textContent = s.warning;
  if (s.setup_required) {
    hint.textContent = s.bootstrap_source === "env"
      ? "First passkey. Paste the STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN you started the dashboard with."
      : "First passkey. Paste the token from the enrol_token file beside ~/.strands_dashboard/auth.json on the machine running the dashboard.";
    enrol.hidden = false;
  } else if (!s.authenticated && !s.open_posture) {
    hint.textContent = "This dashboard is sealed by a passkey.";
    login.hidden = false;
  }
  authenticated = s.authenticated || s.open_posture;
  return s;
}

async function loadWho() {
  try {
    const w = await api("/api/whoami");
    $("#who").textContent = w.via === "passkey" ? `passkey · ${w.name || w.sub}` : w.via === "token" ? "token" : "this machine · no passkey yet";
  } catch { /* routed to login */ }
}

async function loadFleet() {
  const box = $("#fleet"); box.textContent = "";
  try {
    const f = await api("/api/fleet");
    $("#fleet-count").textContent = `${f.robots.length} robots`;
    for (const r of f.robots) {
      const el = node("article", "robot");
      el.append(node("div", "name", r.name), node("div", "desc", r.description || ""));
      const meta = node("div", "meta");
      meta.append(node("span", "pill", r.category || ""), node("span", "pill", `${r.joints ?? "?"} dof`));
      if (r.has_sim) meta.append(node("span", "pill sim", "sim"));
      if (r.has_real) meta.append(node("span", "pill real", "real"));
      el.append(meta);
      box.appendChild(el);
    }
  } catch (e) {
    if (e.message !== "sign in required") box.append(node("p", "muted", e.message === "Not Found" ? "The fleet route arrives with the next slice." : e.message));
  }
}

async function loadSettings() {
  const form = $("#settings-form"); form.textContent = "";
  const { settings, file } = await api("/api/settings");
  $("#settings-file").textContent = file;
  for (const [section, values] of Object.entries(settings)) {
    const h = document.createElement("h2"); h.textContent = section; form.appendChild(h);
    for (const [key, value] of Object.entries(values)) {
      const label = document.createElement("label");
      const isSecret = section === "security" && key === "auth_token";
      const shown = isSecret ? (value ? "(set)" : "(unset)") : Array.isArray(value) ? value.join(", ") : value ?? "";
      label.append(node("span", "mono", `${section}.${key}`));
      const input = document.createElement("input"); input.name = `${section}.${key}`; input.value = shown; input.placeholder = isSecret ? "leave to keep" : "";
      label.appendChild(input); form.appendChild(label);
    }
  }
  const btn = document.createElement("button"); btn.type = "submit"; btn.textContent = "Save"; form.appendChild(btn);
}

async function saveSettings(ev) {
  ev.preventDefault();
  const patch = {};
  for (const input of ev.target.querySelectorAll("input")) {
    const [section, key] = input.name.split(".");
    let v = input.value.trim();
    if (section === "security" && key === "auth_token" && (v === "(set)" || v === "(unset)" || v === "")) continue;
    if (v === "") v = null;
    (patch[section] ||= {})[key] = v;
  }
  try {
    const r = await api("/api/settings", { method: "POST", body: JSON.stringify(patch) });
    $("#settings-msg").textContent = r.changed.length ? `saved: ${r.changed.join(", ")}` : "nothing changed";
  } catch (e) { $("#settings-msg").textContent = e.message; }
}


/* ---- sim ---- */
const sockets = new Map();
const twins = new Map();
/* The lockout as the server last reported it. Only lockoutLine() writes it, and
   only from a server answer - a failed request is not an e-stop, so it is shown
   as a message and the line is re-read from /api/safety rather than painted. */
let lockout = { state: "unknown", reason: "not read yet" };

function lockoutLine(l) {
  lockout = l;
  const el = $("#lockout");
  el.className = `lockout ${l.state}`;
  el.textContent = l.state === "locked" ? `e-stop engaged · ${l.by || "dashboard"} · ${l.reason}` : l.state === "unknown" ? `lockout unknown · ${l.reason}` : `clear · ${l.reason}`;
  // The red button's label and the action a click takes are both read from
  // this state, so they cannot disagree: a page loaded under an engaged e-stop
  // reads RESUME and resumes, and a button that reads E-STOP stops.
  $("#estop").textContent = l.state === "locked" ? "RESUME" : "E-STOP";
}

function simMessage(text) { $("#sim-msg").textContent = text; }

async function simFailed(e) {
  simMessage(e.message);
  try { lockoutLine((await api("/api/safety")).lockout); } catch (again) { simMessage(`${e.message} · ${again.message}`); }
}

async function loadSim() {
  const sel = $("#sim-robot");
  if (!sel.options.length) {
    const f = await api("/api/fleet?mode=sim");
    for (const r of f.robots.filter(r => r.model_local)) {
      const o = document.createElement("option"); o.value = r.name; o.textContent = `${r.name} · ${r.joints} dof`; sel.appendChild(o);
    }
    sel.value = "so101";
  }
  lockoutLine((await api("/api/safety")).lockout);
  const { sessions } = await api("/api/sim");
  const box = $("#sessions");
  for (const el of [...box.children]) if (!sessions.some(s => s.id === el.dataset.id)) { sockets.get(el.dataset.id)?.close(); sockets.delete(el.dataset.id); twins.get(el.dataset.id)?.dispose(); twins.delete(el.dataset.id); el.remove(); }
  for (const s of sessions) if (!box.querySelector(`[data-id="${s.id}"]`)) mountSession(s);
  if (!sessions.length) { if (!box.querySelector("p.muted")) box.appendChild(node("p", "muted", "No session yet. Pick a robot and press Start - it steps in this process and streams here.")); }
  else box.querySelector("p.muted")?.remove();
}

function mountSession(s) {
  const el = node("article", "session"); el.dataset.id = s.id;
  // Built node by node: the robot name, the session id and the joint names all
  // arrive from routes, and each one is handed to the page as text.
  const head = node("div", "head");
  head.append(node("span", "name", s.robot), node("span", "pill mono", s.id), node("span", "pill state", s.state));
  const view = node("div", "view");
  const canvas = node("canvas", "twin");
  const img = node("img", "cam"); img.alt = `${s.robot} camera`; img.hidden = true;
  const viewsel = node("div", "viewsel"); viewsel.setAttribute("role", "tablist");
  const twinBtn = node("button", "on", "Twin"); twinBtn.dataset.view = "twin";
  const camBtn = node("button", "", "Camera"); camBtn.dataset.view = "cam";
  viewsel.append(twinBtn, camBtn);
  view.append(canvas, img, viewsel);
  const joints = node("div", "joints");
  for (const n of s.joint_names) {
    const j = node("div", "joint");
    const bar = node("span", "bar"); bar.appendChild(document.createElement("i"));
    j.append(node("span", "label", n), node("span", "val", "0.000"), bar);
    joints.appendChild(j);
  }
  const foot = node("div", "foot");
  foot.append(node("span", "t", "t=0.00s"), node("span", "fps", ""));
  const reset = node("button", "", "Reset"); reset.dataset.act = "reset";
  const stop = node("button", "", "Stop"); stop.dataset.act = "stop";
  foot.append(reset, stop);
  el.append(head, view, joints, foot);
  $("#sessions").appendChild(el);
  el.querySelector('[data-act="stop"]').onclick = async () => { await api(`/api/sim/${s.id}`, { method: "DELETE" }); loadSim(); };
  el.querySelector('[data-act="reset"]').onclick = async () => { try { await api(`/api/sim/${s.id}/reset`, { method: "POST" }); simMessage(""); } catch (e) { await simFailed(e); } };
  const twin = new Twin(el.querySelector("canvas.twin"), s.id);
  twins.set(s.id, twin);
  twin.load().catch((e) => console.warn("twin", e));
  for (const b of el.querySelectorAll(".viewsel button")) b.onclick = () => {
    el.querySelectorAll(".viewsel button").forEach(x => x.classList.toggle("on", x === b));
    const cam = b.dataset.view === "cam";
    img.hidden = !cam; el.querySelector("canvas.twin").hidden = cam;
    img.src = cam ? `/api/sim/${s.id}/stream.mjpg` : ""; // only stream while shown
  };
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/telemetry/${s.id}?poses=1`);
  ws.binaryType = "arraybuffer";
  sockets.set(s.id, ws);
  const vals = el.querySelectorAll(".joint .val"), bars = el.querySelectorAll(".joint .bar i");
  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) { twin.poses(ev.data); return; }
    const m = JSON.parse(ev.data);
    el.classList.toggle("frozen", m.state === "frozen");
    el.querySelector(".state").textContent = m.state;
    el.querySelector(".t").textContent = `t=${m.sim_time.toFixed(2)}s`;
    el.querySelector(".fps").textContent = m.fps ? `${m.fps} fps` : "";
    m.qpos.forEach((q, i) => { if (vals[i]) { vals[i].textContent = q.toFixed(3); bars[i].style.transform = `translateX(${Math.max(-1, Math.min(1, q / Math.PI)) * 40}px)`; } });
    lockoutLine(m.lockout);
    if (m.state === "stopped" || m.state === "error") ws.close();
  };
}

$("#sim-new").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try { await api("/api/sim", { method: "POST", body: JSON.stringify({ robot: $("#sim-robot").value }) }); simMessage(""); await loadSim(); }
  catch (e) { await simFailed(e); }
});
$("#estop").addEventListener("click", async () => {
  const action = lockout.state === "locked" ? "resume" : "estop";
  try { lockoutLine((await api(`/api/safety/${action}`, { method: "POST" })).lockout); simMessage(""); }
  catch (e) { await simFailed(e); }
});

/* ---- ceremonies ---- */
$("#enrol").addEventListener("submit", async (ev) => {
  ev.preventDefault(); $("#login-error").textContent = "";
  const fd = new FormData(ev.target);
  try {
    const begin = await api("/api/auth/register/begin", { method: "POST", body: JSON.stringify({ bootstrap: fd.get("bootstrap"), label: fd.get("label") }) });
    const cred = await navigator.credentials.create({ publicKey: toCreationOptions(begin.options) });
    await api("/api/auth/register/finish", { method: "POST", body: JSON.stringify({ challenge_id: begin.challenge_id, credential: serializeCredential(cred) }) });
    await boot();
  } catch (e) { $("#login-error").textContent = e.message; }
});
$("#login").addEventListener("click", async () => {
  $("#login-error").textContent = "";
  try {
    const begin = await api("/api/auth/login/begin", { method: "POST" });
    const cred = await navigator.credentials.get({ publicKey: toRequestOptions(begin.options) });
    await api("/api/auth/login/finish", { method: "POST", body: JSON.stringify({ challenge_id: begin.challenge_id, credential: serializeCredential(cred) }) });
    await boot();
  } catch (e) { $("#login-error").textContent = e.message; }
});

/* ---- agent ---- */
let agentWs = null, agentTurn = null;
function agentSocket() {
  if (agentWs && agentWs.readyState <= 1) return agentWs;
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/agent`);
  agentWs = ws;
  ws.onmessage = (ev) => agentEvent(JSON.parse(ev.data));
  ws.onclose = (ev) => { if (ev.code === 4401) showLogin(); agentIdle(); };
  return ws;
}
function agentLine(cls, text) {
  const log = $("#agent-log"), el = document.createElement("div");
  el.className = cls; el.textContent = text; log.appendChild(el); log.scrollTop = log.scrollHeight; return el;
}
function agentIdle() { $("#agent-send").disabled = false; agentTurn = null; }
function agentEvent(m) {
  switch (m.type) {
    case "text":
      if (!agentTurn) agentTurn = agentLine("turn agent", "");
      agentTurn.textContent += m.text; $("#agent-log").scrollTop = 1e9; break;
    case "tool_use":
      agentTurn = null; agentLine("tool", `▸ ${m.name} ${JSON.stringify(m.input)}`); break;
    case "tool_result":
      agentTurn = null; agentLine(`tool${m.status === "error" ? " err" : ""}`, `  ${m.text || m.status}`); break;
    case "interrupt": {
      agentTurn = null;
      const r = m.reason || {}, card = node("div", "consent");
      // The detail is the tool's own words about a move it wants to make; it is
      // shown to the operator as text, and answered by the buttons built here.
      const what = node("div", "what", `${r.tool} · session ${r.session_id || "?"}`);
      what.append(document.createElement("br"), document.createTextNode(r.detail || JSON.stringify(r.positions)));
      const row = node("div", "row");
      for (const [cls, a, label] of [["yes", "once", "Allow once"], ["yes", "always", "Allow for this conversation"], ["no", "no", "Refuse"]]) {
        const b = node("button", cls, label); b.dataset.a = a; row.appendChild(b);
      }
      card.append(node("b", "", "The agent wants to move a robot."), what, row);
      for (const b of card.querySelectorAll("button")) b.onclick = () => {
        card.classList.add("answered");
        card.querySelector(".what").insertAdjacentText("beforeend", b.dataset.a === "no" ? "\n— refused" : b.dataset.a === "always" ? "\n— allowed for this conversation" : "\n— allowed once");
        agentSocket().send(JSON.stringify({ type: "resume", id: m.id, approve: b.dataset.a !== "no", always: b.dataset.a === "always" }));
      };
      $("#agent-log").appendChild(card); $("#agent-log").scrollTop = 1e9; break;
    }
    case "done": agentIdle(); break;
    case "error": agentLine("error", m.message); agentIdle(); break;
  }
}
$("#agent-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = $("#agent-text").value.trim(); if (!text) return;
  const ws = agentSocket();
  const send = () => { agentLine("turn you", text); ws.send(JSON.stringify({ type: "say", text })); $("#agent-text").value = ""; $("#agent-send").disabled = true; agentTurn = null; };
  ws.readyState === 1 ? send() : ws.addEventListener("open", send, { once: true });
});
async function loadAgent() {
  try { const info = await api("/api/agent"); $("#agent-model").textContent = info.model; } catch (e) { $("#agent-model").textContent = ""; }
  agentSocket();
}

$("#tabs").addEventListener("click", (ev) => {
  const v = ev.target.dataset.view; if (!v) return;
  if (!authenticated) return showLogin();
  show(v);
  if (v === "fleet") loadFleet();
  if (v === "sim") loadSim();
  if (v === "agent") loadAgent();
  if (v === "settings") loadSettings();
});
$("#settings-form").addEventListener("submit", saveSettings);

async function boot() {
  const h = await fetch("/api/health").then(r => r.json()).catch(() => ({}));
  $("#version").textContent = h.version ? `strands-robots ${h.version}` : "";
  await loadStatus();
  if (!authenticated) return showLogin();
  await loadWho();
  const v = views.includes(location.hash.slice(1)) ? location.hash.slice(1) : "fleet";
  show(v);
  if (v === "fleet") loadFleet();
  if (v === "sim") loadSim();
  if (v === "agent") loadAgent();
  if (v === "settings") loadSettings();
}
boot();
