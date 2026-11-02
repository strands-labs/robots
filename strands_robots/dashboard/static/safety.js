/* E-stops, placed where they never cover an input:
 *   - STOP ALL in the top bar: one click, no confirm, e-stops every robot in this world and every mesh device;
 *     Shift+Esc does the same from the keyboard (even while typing).
 *   - each robot card's header: that robot's E-STOP (no confirm) and a Resume that needs the operator's name
 *     (a real robot also needs an X-Confirm). app.js renders the slot (.estop-ctl[data-robot]); this file wires it.
 *   - mesh devices (other peers' robots) get a row each with their own E-STOP; resume stays with the device.
 * The latched state comes from the server's state frames (frame.estop), never from a local guess. */

const $ = (s) => document.querySelector(s);
const bar = $("#safety");
const say_ = $("#safety-say");
window.__safety = { notices: [] };

async function post(path, body, confirmed) {
  const headers = { "content-type": "application/json" };
  if (confirmed) headers["X-Confirm"] = "yes";
  const res = await fetch(`${window.__state?.base || ""}${path}`, { method: "POST", headers, body: JSON.stringify(body || {}) });
  const doc = await res.json().catch(() => ({}));
  return { status: res.status, doc };
}

function say(text, bad) {
  window.__safety.notices.push(text);
  say_.textContent = text;
  say_.title = text;
  say_.classList.toggle("bad", Boolean(bad));
}

async function estop(id) {
  const r = await post(`/api/robots/${encodeURIComponent(id)}/estop`);
  say(r.status === 200 ? `${id}: e-stop engaged` : `${id}: ${r.doc.detail || r.status}`, r.status !== 200);
  return r.status === 200;
}

async function resume(id) {
  const who = window.prompt(`Resume ${id}: who cleared the stop? (operator name)`) || "";
  const real = (window.__state?.desc?.robots || []).find((x) => x.id === id)?.is_real;
  if (real && !window.confirm(`${id} is a REAL robot. Release its e-stop (approved by ${who || "nobody"})?`)) return;
  const r = await post(`/api/robots/${encodeURIComponent(id)}/resume`, { approved_by: who }, real);
  say(r.status === 200 ? `${id}: resumed by ${r.doc.resumed_by}` : `${id}: ${r.doc.detail || r.status}`, r.status !== 200);
}

async function meshEstop(id) {
  const r = await post(`/api/mesh/${encodeURIComponent(id)}/estop`);
  say(r.status === 200 ? `${id}: e-stop sent over the mesh` : `${id}: ${r.doc.detail || r.status}`, r.status !== 200);
  return r.status === 200;
}

/** Every robot the page knows, then every remote mesh device, all at once: a stop is never queued behind another. */
async function stopAll() {
  const robots = (window.__state?.desc?.robots || []).map((r) => r.id);
  const devices = remote.map((d) => d.id);
  const got = await Promise.all([...robots.map(estop), ...devices.map(meshEstop)]);
  const ok = got.filter(Boolean).length;
  say(`STOP ALL: ${ok}/${got.length} stopped${got.length - ok ? ` · ${got.length - ok} refused` : ""}`, ok !== got.length);
  window.__safety.stopAll = { robots, devices, ok };
}

// One delegated listener: the cards are re-rendered on every scene revision, the listener is not.
document.addEventListener("click", (e) => {
  const ctl = e.target.closest(".estop-ctl[data-robot]");
  if (!ctl) return;
  if (e.target.closest("button.estop")) { e.stopPropagation(); estop(ctl.dataset.robot); }
  else if (e.target.closest("button.resume")) { e.stopPropagation(); resume(ctl.dataset.robot); }
});
$("#stop-all").addEventListener("click", stopAll);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && e.shiftKey && !bar.hidden) { e.preventDefault(); stopAll(); }
});

function paint(estopMap) {
  let any = false;
  for (const ctl of document.querySelectorAll(".estop-ctl[data-robot]")) {
    const id = ctl.dataset.robot;
    if (!(id in (estopMap || {}))) continue;
    const latched = Boolean(estopMap[id]);
    any ||= latched;
    const flag = latched ? "yes" : "no";
    if (ctl.dataset.latched !== flag) {
      ctl.dataset.latched = flag;
      ctl.querySelector(".estop-status").textContent = latched ? "LATCHED" : "armed";
      ctl.closest(".card")?.setAttribute("data-latched", flag);
    }
  }
  $("#stop-all").dataset.latched = any ? "yes" : "no";
}

/* Only a server has something to stop: in standalone, static or offline mode the bar's stop is hidden and never polls. */
bar.hidden = true;
let meshTimer = 0;
setInterval(() => {
  const st = window.__state;
  bar.hidden = st?.mode !== "server";
  if (!bar.hidden && !meshTimer) { pollMesh(); meshTimer = setInterval(pollMesh, 2000); }
  if (st?.lastFrame?.estop) paint(st.lastFrame.estop);
}, 50);

/* Mesh devices (other peers' robots): listed under the world's cards, each with its own E-STOP. Hidden when the server
 * has no mesh. */
const meshBox = $("#mesh-devices");
let remote = [];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function pollMesh() {
  const doc = await fetch(`${window.__state?.base || ""}/api/mesh`).then((r) => (r.ok ? r.json() : null), () => null);
  remote = (doc?.devices || []).filter((d) => !d.local);
  window.__mesh = doc;
  meshBox.hidden = remote.length === 0;
  const key = JSON.stringify(remote.map((d) => [d.id, d.device_type]));
  if (meshBox.__key === key) return;
  meshBox.__key = key;
  meshBox.innerHTML = remote.length ? `<div class="section-title"><span>mesh · ${esc(doc.tenant)}</span><span>${remote.length}</span></div>` : "";
  for (const d of remote) {
    const row = document.createElement("div");
    row.className = "device";
    row.dataset.device = d.id;
    row.innerHTML = `<span>${esc(d.id)} <small>${esc(d.device_type)}</small></span>`;
    const stop = document.createElement("button");
    stop.className = "estop";
    stop.type = "button";
    stop.textContent = "E-STOP";
    stop.onclick = () => meshEstop(d.id);
    row.append(stop);
    meshBox.appendChild(row);
  }
}
