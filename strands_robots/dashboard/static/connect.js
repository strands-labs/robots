/* Connect panel: other peers' robots over the mesh, policies served elsewhere, hardware cameras.
 *   GET  /api/connect/peers · /api/connect/robots · /api/policies · /api/scene (the world's robots, for the run row)
 *   POST /api/connect/mesh {address} · /api/connect/robot {peer, address} · /api/connect/robot/<id>/{act,detach,resume}
 *        /api/connect/policy {endpoint, kind, deployment} · /api/robots/<id>/{act,run,stop,estop}
 * A remote or hardware card is a real robot to this page: every move, run and resume asks the operator first and goes
 * out with X-Confirm: yes; its E-STOP is one click, never asked. Registers as a side-column tab through the panel slot API (window.SR.registerPanel);
 * on a page without it, it mounts itself as a collapsible panel on the left. */

const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const enc = encodeURIComponent;
const state = { peers: { meshes: [], devices: [] }, robots: [], policies: [], world: [], notices: [], camTick: 0 };
window.__connect = state; // the Playwright gate reads what the panel shows here

const DOCS = "https://strands-labs.github.io/robots";
const STYLE = `
#connect { font: 12px/1.45 var(--sr-text); color: var(--sr-fg); }
#connect.floating { position: absolute; top: 3rem; left: .7rem; width: 21rem; max-height: calc(100% - 4rem); overflow: auto; z-index: 5;
  padding: .6rem; border: 1px solid var(--sr-pill-border); border-radius: var(--sr-radius); background: var(--sr-panel); backdrop-filter: blur(6px); }
#connect.floating.closed { width: auto; max-height: none; overflow: visible; }
#connect.closed .cx-body { display: none; }
#connect header { display: flex; align-items: center; gap: .4rem; }
#connect header h2 { margin: 0; flex: 1; font: 500 .78rem var(--sr-mono); letter-spacing: .03em; }
#connect header a { color: var(--sr-muted); font: .62rem var(--sr-mono); }
#connect h3 a { float: right; text-transform: none; letter-spacing: 0; color: var(--sr-muted); }
#connect .hint { font: .58rem/1.4 var(--sr-mono); color: var(--sr-muted); margin: 0 0 .25rem; }
#connect .hint code { background: var(--sr-chip-bg); padding: 0 .2rem; border-radius: 3px; }
#connect .peers-err { color: var(--sr-bad); }
#connect h3 { margin: .7rem 0 .3rem; font: 500 .64rem var(--sr-mono); color: var(--sr-muted); text-transform: uppercase; letter-spacing: .06em; }
#connect .row { display: flex; gap: .3rem; margin: .25rem 0; }
#connect input, #connect select { min-width: 0; flex: 1; background: var(--sr-bg); color: var(--sr-fg); border: 1px solid var(--sr-pill-border);
  border-radius: 999px; padding: .28rem .65rem; font: .66rem var(--sr-mono); }
#connect button { border-radius: 999px; border: 1px solid var(--sr-pill-border); background: transparent; color: var(--sr-fg); padding: .26rem .7rem;
  font: .64rem var(--sr-mono); cursor: pointer; white-space: nowrap; }
#connect button.go { background: var(--sr-accent); border-color: var(--sr-accent); color: var(--sr-on-accent); }
#connect button.estop { background: var(--sr-bad); border-color: var(--sr-bad); color: #fff; font-weight: 600; }
#connect button:disabled { opacity: .5; cursor: default; }
#connect .peer { display: flex; align-items: center; gap: .4rem; padding: .3rem .1rem; border-top: 1px solid var(--sr-line); font: .66rem var(--sr-mono); }
#connect .peer span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#connect .peer small, #connect .muted { color: var(--sr-muted); }
#connect .rcard { border: 1px solid var(--sr-pill-border); border-radius: var(--sr-radius); padding: .45rem .55rem; margin: .4rem 0; background: var(--sr-card-bg); }
#connect .rcard.latched { border-color: var(--sr-bad); }
#connect .rcard h4 { margin: 0 0 .3rem; display: flex; align-items: center; gap: .35rem; font: 500 .72rem var(--sr-mono); }
#connect .rcard h4 .who { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#connect .via { font: .58rem var(--sr-mono); color: var(--sr-muted); margin-bottom: .2rem; overflow-wrap: anywhere; }
#connect .tag { font-size: .56rem; padding: .15rem .4rem; border-radius: 999px; border: 1px solid var(--sr-warn); color: var(--sr-warn); }
#connect .tag.bad { border-color: var(--sr-bad); color: var(--sr-bad); }
#connect .cams { display: grid; grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); gap: .3rem; margin: .3rem 0; }
#connect .cams figure { margin: 0; } #connect .cams img { width: 100%; display: block; border-radius: 4px; background: var(--sr-viewer-bg); }
#connect .cams figcaption { font: .56rem var(--sr-mono); color: var(--sr-muted); }
#connect .j { display: grid; grid-template-columns: minmax(0, 1fr) 3rem 5.5rem; gap: .3rem; align-items: center; font: .6rem var(--sr-mono); color: var(--sr-muted); }
#connect .j b { color: var(--sr-fg); font-weight: 400; text-align: right; font-variant-numeric: tabular-nums; }
#connect .j input { padding: 0; accent-color: var(--sr-accent); }
#connect .note { margin-top: .4rem; font: .62rem var(--sr-mono); color: var(--sr-muted); overflow-wrap: anywhere; }
#connect .note.bad { color: var(--sr-bad); }
`;

const base = () => window.__state?.base || ""; // "" same origin; the reader's CLI when the page is served by the docs site

async function api(path, { body, confirmed, method } = {}) {
  const headers = { "content-type": "application/json" };
  if (confirmed) headers["X-Confirm"] = "yes";
  const init = body === undefined && !method ? {} : { method: method || "POST", headers, body: JSON.stringify(body || {}) };
  const res = await fetch(base() + path, init).catch((e) => ({ ok: false, status: 0, json: async () => ({ detail: String(e) }) }));
  const doc = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, doc };
}

function note(text, bad = false) {
  state.notices.push(text);
  const n = $("#cx-note");
  if (n) { n.textContent = text; n.classList.toggle("bad", bad); }
}
const fail = (what, r) => note(`${what}: HTTP ${r.status} ${r.doc.detail || ""}`, true);

function mount() {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.appendChild(style);
  const el = document.createElement("section");
  el.id = "connect";
  el.setAttribute("aria-label", "connect");
  el.innerHTML = `
    <header><h2>Connect</h2><a href="${DOCS}/mesh/" target="_blank" rel="noopener">docs ↗</a>
      <button id="cx-toggle" aria-expanded="true" title="show / hide">–</button></header>
    <div class="cx-body">
      <h3>mesh <a href="${DOCS}/mesh/" target="_blank" rel="noopener">how ↗</a></h3>
      <div class="hint">a peer serving robots: <code>Mesh("zenoh://?listen=tcp/0.0.0.0:7447").mount(robot)</code> on that machine; type its <code>host:7447</code>, or <code>lan</code> to scout</div>
      <div class="row"><input id="cx-address" placeholder="host:7447 · lan · zenoh://…" autocomplete="off"><button id="cx-join" class="go">join</button></div>
      <div id="cx-peers"></div>
      <h3>attached robots <a href="${DOCS}/safety/" target="_blank" rel="noopener">safety ↗</a></h3>
      <div id="cx-robots"><div class="muted">attach a peer's robot above; hardware robots passed with robots= show here too</div></div>
      <h3>policy endpoint <a href="${DOCS}/lerobot/" target="_blank" rel="noopener">how ↗</a></h3>
      <div class="hint">lerobot-async: a <code>lerobot-policy-server</code> listening on <code>tcp/host:port</code>, deployment = its config's <code>deployment:</code> name · onnx: a preset (<code>microduck?vx=0.3</code>) · lerobot: <code>owner/repo</code> or a checkpoint dir</div>
      <div class="row"><input id="cx-endpoint" placeholder="host:7450 · zenoh://h:p/dep · preset · repo" autocomplete="off">
        <select id="cx-kind"><option>lerobot-async</option><option>onnx</option><option>lerobot</option><option>dataset</option></select></div>
      <div class="row"><input id="cx-deployment" placeholder="deployment (lerobot-async)" autocomplete="off"><button id="cx-policy" class="go">connect</button></div>
      <h3>run <a href="${DOCS}/policies/" target="_blank" rel="noopener">policies ↗</a></h3>
      <div class="row"><select id="cx-run-robot" aria-label="robot"></select><select id="cx-run-policy" aria-label="policy"></select></div>
      <div class="row"><input id="cx-run-text" placeholder="instruction (optional)" autocomplete="off"><input id="cx-run-seconds" placeholder="s" style="flex:0 0 3rem">
        <button id="cx-run" class="go">run ▸</button><button id="cx-stop">stop</button></div>
      <div id="cx-note" class="note" aria-live="polite"></div>
    </div>`;
  if (window.SR?.registerPanel) { // a side-column tab: never an overlay
    window.SR.registerPanel({ id: "connect", title: "Connect", icon: "⇄", order: 20, mount(pane) { pane.appendChild(el); } });
    $("#cx-toggle", el).hidden = true;
  } else { // a page without the slot API: a collapsible panel on the left
    el.classList.add("floating");
    (document.getElementById("app") || document.body).appendChild(el);
  }
  $("#cx-toggle").onclick = () => {
    const closed = el.classList.toggle("closed");
    $("#cx-toggle").textContent = closed ? "+" : "–";
    $("#cx-toggle").setAttribute("aria-expanded", String(!closed));
  };
  $("#cx-join").onclick = join;
  $("#cx-address").onkeydown = (e) => { if (e.key === "Enter") join(); };
  $("#cx-policy").onclick = addPolicy;
  $("#cx-run").onclick = run;
  $("#cx-stop").onclick = stop;
  el.addEventListener("click", onCardClick);
  el.addEventListener("change", onSlider);
}

async function join() {
  const address = $("#cx-address").value.trim() || "lan";
  const r = await api("/api/connect/mesh", { body: { address } });
  if (!r.ok) return fail("join", r);
  note(`joined ${r.doc.url} (tenant ${r.doc.tenant})`);
  state.joinedAt = Date.now();
  pollPeers();
}

function renderPeers() {
  const { devices, meshes } = state.peers;
  const box = $("#cx-peers");
  let html;
  if (!devices.length) {
    const late = state.joinedAt && Date.now() - state.joinedAt > 4000; // zenoh keeps retrying: silence is the error
    html = meshes.length
      ? `<div class="${late ? "peers-err" : "muted"}">${late ? "no device answers" : "waiting for devices"} on ${meshes.map((m) => `${esc(m.url)} (tenant ${esc(m.tenant)})`).join(", ")}${late ? ": is a peer serving there, in this tenant? (STRANDS_TENANT on both sides)" : " …"}</div>`
      : `<div class="muted">join a mesh: a peer's address, or lan to scout</div>`;
  } else {
    html = devices.map((d) => `<div class="peer" data-device="${esc(d.id)}"><span>${esc(d.id)} <small>${esc(d.name || d.type)} · ${esc(d.mesh)}</small></span>${
      d.local ? `<small>this page</small>` : d.attached ? `<small>attached</small>` : `<button class="go" data-attach="${esc(d.id)}" data-mesh="${esc(d.mesh)}">attach</button>`}</div>`).join("");
  }
  if (box.dataset.html === html) return; // rows stay the same nodes between polls: a button is never pulled from under a click
  box.dataset.html = html;
  box.innerHTML = html;
}

function camUrl(c) { return `${base()}${c.url}?t=${state.camTick}`; }

function renderRobots() {
  const box = $("#cx-robots");
  if (!state.robots.length) return;
  const have = new Set([...box.querySelectorAll(".rcard")].map((n) => n.dataset.robot));
  const want = new Set(state.robots.map((r) => r.id));
  if ([...have].some((x) => !want.has(x)) || [...want].some((x) => !have.has(x))) box.innerHTML = state.robots.map(cardHtml).join("");
  for (const r of state.robots) {
    const card = box.querySelector(`.rcard[data-robot="${CSS.escape(r.id)}"]`);
    if (!card) continue;
    card.classList.toggle("latched", !!r.estop);
    $(".latch", card).hidden = !r.estop;
    $("[data-resume]", card).hidden = !r.estop;
    for (const [k, v] of Object.entries(r.joints || {})) {
      const b = card.querySelector(`b[data-joint="${CSS.escape(k)}"]`);
      if (b) b.textContent = (+v).toFixed(3);
    }
    if (r.error) $(".rerr", card).textContent = r.error;
  }
}

function cardHtml(r) {
  const joints = Object.entries(r.joints || {}).map(([k, v]) => `<div class="j"><span>${esc(k)}</span><b data-joint="${esc(k)}">${(+v).toFixed(3)}</b>
    <input type="range" min="-1.5" max="1.5" step="0.05" value="${(+v).toFixed(2)}" data-robot="${esc(r.id)}" data-joint="${esc(k)}" aria-label="${esc(k)}"></div>`).join("");
  const cams = (r.cameras || []).map((c) => `<figure><img data-cam="${esc(c.url)}" src="${esc(camUrl(c))}" alt="${esc(r.id)} ${esc(c.name)}"><figcaption>${esc(c.name)}</figcaption></figure>`).join("");
  return `<div class="rcard" data-robot="${esc(r.id)}"><h4><span class="who">${esc(r.id)} <small class="muted">${esc(r.name || "")}</small></span>
    <span class="tag">${esc(r.kind)}</span><span class="tag bad latch" hidden>E-STOPPED</span>
    <button class="estop" data-estop="${esc(r.id)}" title="stop now (never asks)">E-STOP</button></h4>
    ${r.mesh ? `<div class="via">via ${esc(r.mesh)}</div>` : ""}
    ${cams ? `<div class="cams">${cams}</div>` : ""}${joints}
    <div class="row"><button data-resume="${esc(r.id)}" data-kind="${esc(r.kind)}" hidden>resume…</button>${r.kind === "remote" ? `<button data-detach="${esc(r.id)}">detach</button>` : ""}</div>
    <div class="note bad rerr"></div></div>`;
}

async function onCardClick(e) {
  const t = e.target.closest("button");
  if (!t) return;
  if (t.dataset.attach) {
    t.disabled = true;
    const r = await api("/api/connect/robot", { body: { peer: t.dataset.attach, address: t.dataset.mesh } });
    if (!r.ok) { t.disabled = false; return fail(`attach ${t.dataset.attach}`, r); }
    note(`attached ${r.doc.id} (${r.doc.kind}): moves ask you first`);
    await Promise.all([pollPeers(), pollRobots(), loadRunPickers()]);
  } else if (t.dataset.estop) {
    const r = await api(`/api/robots/${enc(t.dataset.estop)}/estop`, { body: {} });
    if (!r.ok) return fail(`e-stop ${t.dataset.estop}`, r);
    note(`${t.dataset.estop} e-stopped${t.closest(".rcard").querySelector(".via") ? " over the mesh" : ""}`);
    pollRobots();
  } else if (t.dataset.resume) {
    const rid = t.dataset.resume;
    if (!confirm(`Release the e-stop of ${rid} (a real robot to this page)? It may move again.`)) return;
    const path = t.dataset.kind === "remote" ? `/api/connect/robot/${enc(rid)}/resume` : `/api/robots/${enc(rid)}/resume`;
    const r = await api(path, { body: {}, confirmed: true });
    if (!r.ok) return fail(`resume ${rid}`, r);
    note(`${rid} resumed (approved by ${r.doc.resumed_by || "you"})`);
    pollRobots();
  } else if (t.dataset.detach) {
    const r = await api(`/api/connect/robot/${enc(t.dataset.detach)}/detach`, { body: {} });
    if (!r.ok) return fail("detach", r);
    note(`detached ${t.dataset.detach}`);
    await Promise.all([pollPeers(), pollRobots(), loadRunPickers()]);
  }
}

async function onSlider(e) {
  const el = e.target;
  if (el.type !== "range" || !el.dataset.robot) return;
  const { robot, joint } = el.dataset;
  if (!confirm(`${robot} is a real robot to this page. Move ${joint} to ${(+el.value).toFixed(2)} rad?`)) return;
  const r = await api(`/api/connect/robot/${enc(robot)}/act`, { body: { targets: { [joint]: +el.value }, seconds: 1.0 }, confirmed: true });
  state.acts = (state.acts || []).concat([{ robot, joint, value: +el.value, status: r.status }]);
  if (!r.ok) return fail(`move ${robot}`, r);
  note(`${robot}.${joint} → ${(+r.doc.joints[joint]).toFixed(3)}`);
}

async function addPolicy() {
  const body = { endpoint: $("#cx-endpoint").value.trim(), kind: $("#cx-kind").value, deployment: $("#cx-deployment").value.trim() || null };
  note(`asking ${body.endpoint} …`);
  const r = await api("/api/connect/policy", { body });
  if (!r.ok) return fail("policy", r);
  note(`policy ${r.doc.name}: ${r.doc.description}${r.doc.info?.features?.length ? ` · features ${r.doc.info.features.join(", ")}` : ""}`);
  await loadRunPickers(r.doc.name);
}

function isReal(rid) { return state.robots.some((r) => r.id === rid); }

async function run() {
  const rid = $("#cx-run-robot").value, policy = $("#cx-run-policy").value;
  if (!rid || !policy) return note("pick a robot and a policy", true);
  const real = isReal(rid);
  if (real && !confirm(`Run ${policy} on ${rid}, a real robot to this page?`)) return;
  const secs = parseFloat($("#cx-run-seconds").value);
  const body = { policy, instruction: $("#cx-run-text").value.trim() || null, seconds: Number.isFinite(secs) ? secs : null };
  const r = await api(`/api/robots/${enc(rid)}/run`, { body, confirmed: real });
  state.runs = (state.runs || []).concat([{ rid, policy, status: r.status }]);
  if (!r.ok) return fail(`run ${policy} on ${rid}`, r);
  note(`${rid}: ${policy} ${r.doc.status}`);
  watchRun(rid);
}

async function watchRun(rid) {
  for (let i = 0; i < 600; i++) {
    await new Promise((ok) => setTimeout(ok, 500));
    const r = await api(`/api/robots/${enc(rid)}/run`);
    if (!r.ok) return;
    state.lastRun = r.doc;
    if (r.doc.status !== "running") {
      const res = r.doc.result;
      return note(`${rid}: ${r.doc.status}${res ? ` · ${res.steps} steps at ${res.hz_achieved} Hz, stopped by ${res.stopped_by}` : ""}${r.doc.error ? ` · ${r.doc.error}` : ""}`, r.doc.status === "error");
    }
    if (i % 4 === 0) note(`${rid}: running …`);
  }
}

async function stop() {
  const rid = $("#cx-run-robot").value;
  const r = await api(`/api/robots/${enc(rid)}/stop`, { body: {} });
  if (!r.ok) return fail("stop", r);
  note(`${rid}: ${r.doc.status}`);
}

async function loadRunPickers(pick) {
  const [scene, pols] = await Promise.all([api("/api/scene"), api("/api/policies")]);
  state.world = scene.ok ? scene.doc.robots.map((r) => r.id) : [];
  state.policies = pols.ok ? pols.doc.policies : [];
  const robots = [...new Set([...state.world, ...state.robots.map((r) => r.id)])];
  const rs = $("#cx-run-robot"), ps = $("#cx-run-policy");
  const keepR = rs.value, keepP = pick || ps.value;
  rs.innerHTML = robots.map((id) => `<option value="${esc(id)}">${esc(id)}${isReal(id) ? " (real)" : ""}</option>`).join("");
  ps.innerHTML = state.policies.map((p) => `<option value="${esc(p.name)}">${esc(p.label || p.name)} · ${esc(p.kind)}</option>`).join("");
  if (robots.includes(keepR)) rs.value = keepR;
  if (state.policies.some((p) => p.name === keepP)) ps.value = keepP;
}

async function pollPeers() {
  const r = await api("/api/connect/peers");
  if (r.ok) { state.peers = r.doc; renderPeers(); }
}

async function pollRobots() {
  if (state.polling) return; // one read in flight: a slow peer never stacks requests up
  state.polling = true;
  const r = await api("/api/connect/robots").finally(() => { state.polling = false; });
  if (!r.ok) return;
  const before = state.robots.map((x) => x.id).join();
  state.robots = r.doc.robots;
  renderRobots();
  if (state.robots.map((x) => x.id).join() !== before) loadRunPickers();
}

function refreshCams() {
  state.camTick++;
  for (const img of document.querySelectorAll("#connect img[data-cam]")) {
    if (img.complete) img.src = `${base()}${img.dataset.cam}?t=${state.camTick}`;
  }
}

(async () => {
  const probe = await api("/api/connect/peers"); // static and standalone pages have no server to connect through
  if (!probe.ok) return;
  mount();
  state.peers = probe.doc;
  renderPeers();
  await Promise.all([pollRobots(), loadRunPickers()]);
  setInterval(pollPeers, 2000);
  setInterval(pollRobots, 500);
  setInterval(refreshCams, 400);
  state.ready = true;
})();
