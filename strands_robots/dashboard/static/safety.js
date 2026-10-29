/* The e-stop panel: one row per robot, an E-STOP button (no confirm: stopping is always allowed) and a Resume that
 * needs the operator's name. The latched state comes from the server's state frames (frame.estop), never from a
 * local guess, so the panel shows what the world says. */

const panel = document.createElement("section");
panel.id = "safety";
panel.setAttribute("aria-label", "e-stop");
panel.style.cssText =
  "position:fixed;right:12px;bottom:12px;z-index:20;min-width:15rem;padding:.5rem .6rem;border-radius:var(--sr-radius);" +
  "background:var(--sr-panel);border:1px solid var(--sr-pill-border);font:12px/1.5 var(--sr-mono);color:var(--sr-fg)";
document.body.appendChild(panel);

const rows = new Map(); // robot id -> {row, status}
window.__safety = { notices: [] };

async function post(path, body, confirmed) {
  const headers = { "content-type": "application/json" };
  if (confirmed) headers["X-Confirm"] = "yes";
  const res = await fetch(path, { method: "POST", headers, body: JSON.stringify(body || {}) });
  const doc = await res.json().catch(() => ({}));
  return { status: res.status, doc };
}

function say(text, bad) {
  window.__safety.notices.push(text);
  let n = panel.querySelector(".safety-notice");
  if (!n) { n = document.createElement("div"); n.className = "safety-notice"; panel.appendChild(n); }
  n.textContent = text;
  n.style.color = bad ? "var(--sr-bad)" : "var(--sr-muted)";
}

function rowFor(id) {
  if (rows.has(id)) return rows.get(id);
  const row = document.createElement("div");
  row.dataset.robot = id;
  row.style.cssText = "display:flex;align-items:center;gap:.4rem;margin:.2rem 0";
  const name = document.createElement("span");
  name.textContent = id;
  name.style.flex = "1";
  const status = document.createElement("span");
  status.className = "estop-status";
  const stop = document.createElement("button");
  stop.className = "estop";
  stop.textContent = "E-STOP";
  stop.style.cssText = "background:var(--sr-bad);color:#fff;border:0;border-radius:4px;padding:.15rem .5rem;font:inherit;cursor:pointer";
  stop.onclick = async () => {
    const r = await post(`/api/robots/${encodeURIComponent(id)}/estop`);
    say(r.status === 200 ? `${id}: e-stop engaged` : `${id}: ${r.doc.detail || r.status}`, r.status !== 200);
  };
  const resume = document.createElement("button");
  resume.className = "resume";
  resume.textContent = "Resume";
  resume.style.cssText = "background:transparent;color:var(--sr-fg);border:1px solid var(--sr-pill-border);border-radius:4px;padding:.15rem .5rem;font:inherit;cursor:pointer";
  resume.onclick = async () => {
    const who = window.prompt(`Resume ${id}: who cleared the stop? (operator name)`) || "";
    const real = (window.__state?.desc?.robots || []).find((x) => x.id === id)?.is_real;
    if (real && !window.confirm(`${id} is a REAL robot. Release its e-stop (approved by ${who || "nobody"})?`)) return;
    const r = await post(`/api/robots/${encodeURIComponent(id)}/resume`, { approved_by: who }, real);
    say(r.status === 200 ? `${id}: resumed by ${r.doc.resumed_by}` : `${id}: ${r.doc.detail || r.status}`, r.status !== 200);
  };
  row.append(name, status, stop, resume);
  panel.prepend(row);
  const entry = { row, status };
  rows.set(id, entry);
  return entry;
}

function paint(estop) {
  for (const [id, latched] of Object.entries(estop || {})) {
    const { row, status } = rowFor(id);
    status.textContent = latched ? "LATCHED" : "armed";
    status.style.color = latched ? "var(--sr-bad)" : "var(--sr-accent)";
    row.dataset.latched = latched ? "yes" : "no";
  }
  for (const [id, { row }] of rows) if (!(id in (estop || {}))) { row.remove(); rows.delete(id); }
}

/* Only a server has something to stop: in standalone, static or offline mode the panel is hidden and never polls. */
panel.hidden = true;
let meshTimer = 0;
setInterval(() => {
  const st = window.__state;
  panel.hidden = st?.mode !== "server";
  if (!panel.hidden && !meshTimer) { pollMesh(); meshTimer = setInterval(pollMesh, 2000); }
  if (st?.lastFrame?.estop) paint(st.lastFrame.estop);
}, 50);

/* Mesh devices (other peers' robots): listed under the world's, each with its own E-STOP (resume stays with the
 * device's resume authority, never this page). Hidden when the server has no mesh. */
const meshBox = document.createElement("div");
meshBox.id = "mesh-devices";
meshBox.hidden = true;
panel.appendChild(meshBox);

async function pollMesh() {
  const doc = await fetch("/api/mesh").then((r) => (r.ok ? r.json() : null), () => null);
  const remote = (doc?.devices || []).filter((d) => !d.local);
  window.__mesh = doc;
  meshBox.hidden = remote.length === 0;
  meshBox.innerHTML = remote.length ? `<div style="color:var(--sr-muted);margin-top:.3rem">mesh · ${String(doc.tenant ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c])}</div>` : "";
  for (const d of remote) {
    const row = document.createElement("div");
    row.dataset.device = d.id;
    row.style.cssText = "display:flex;align-items:center;gap:.4rem;margin:.2rem 0";
    const name = document.createElement("span");
    name.textContent = `${d.id} · ${d.device_type}`;
    name.style.flex = "1";
    const stop = document.createElement("button");
    stop.className = "estop";
    stop.textContent = "E-STOP";
    stop.style.cssText = "background:var(--sr-bad);color:#fff;border:0;border-radius:4px;padding:.15rem .5rem;font:inherit;cursor:pointer";
    stop.onclick = async () => {
      const r = await post(`/api/mesh/${encodeURIComponent(d.id)}/estop`);
      say(r.status === 200 ? `${d.id}: e-stop sent over the mesh` : `${d.id}: ${r.doc.detail || r.status}`, r.status !== 200);
    };
    row.append(name, stop);
    meshBox.appendChild(row);
  }
}
