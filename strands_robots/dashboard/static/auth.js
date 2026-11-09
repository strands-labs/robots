/* Passkey sign-in (dashboard/auth/): the first-time setup and sign-in screen, and the two WebAuthn ceremonies.
 *
 * A session has two halves (dashboard/auth/session.py): the token, in an HttpOnly cookie this module never sees, and
 * the proof the sign-in answer returns, kept here in this origin's localStorage. A browser sends a localhost cookie
 * to every localhost port but keeps localStorage per port, so the cookie alone opens nothing: every request to this
 * dashboard goes through authed() (the proof as a header), every socket through socket() (the proof as its
 * subprotocol), and a camera frame through image(). signIn() resolves true when
 * the page may go on (signed in, sign-in off, or no server: standalone); otherwise it shows the screen and the page
 * reloads once a passkey signs in. Server text is only ever set with textContent.
 */

const JSON_POST = (body) => ({ method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body ?? {}) });
const PROOF = "strands-robots-session-proof";
const proof = () => { try { return localStorage.getItem(PROOF); } catch { return null; } };
const keep = (p) => { try { if (p) localStorage.setItem(PROOF, p); else localStorage.removeItem(PROOF); } catch { /* storage off */ } };

/** fetch() that carries this page's half of the session to this dashboard (never to another origin). */
export function authed(url, init = {}) {
  const p = proof();
  if (!p || new URL(url, location.href).origin !== location.origin) return fetch(url, init);
  const headers = new Headers(init.headers || {});
  headers.set("x-strands-session", p);
  return fetch(url, { ...init, headers });
}

/** A WebSocket to this dashboard, offering the proof as its subprotocol (a browser cannot set a socket's headers). */
export function socket(url) {
  const p = proof();
  return p ? new WebSocket(url, [`strands-session.${p}`]) : new WebSocket(url);
}

/** Show a camera frame from ``url`` in ``img`` (an <img src> cannot carry the proof): fetched, then a blob: URL. */
export async function image(img, url) {
  const res = await authed(url).catch(() => null);
  if (!res || !res.ok) return;
  const old = img.src;
  img.src = URL.createObjectURL(await res.blob());
  if (old.startsWith("blob:")) URL.revokeObjectURL(old);
}

const b64 = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64 = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4)), (c) => c.charCodeAt(0)).buffer;

function creationOptions(o) {
  if (window.PublicKeyCredential?.parseCreationOptionsFromJSON) return PublicKeyCredential.parseCreationOptionsFromJSON(o);
  return { ...o, challenge: unb64(o.challenge), user: { ...o.user, id: unb64(o.user.id) }, excludeCredentials: (o.excludeCredentials || []).map((c) => ({ ...c, id: unb64(c.id) })) };
}
function requestOptions(o) {
  if (window.PublicKeyCredential?.parseRequestOptionsFromJSON) return PublicKeyCredential.parseRequestOptionsFromJSON(o);
  return { ...o, challenge: unb64(o.challenge), allowCredentials: (o.allowCredentials || []).map((c) => ({ ...c, id: unb64(c.id) })) };
}
function credentialJSON(cred) {
  const r = cred.response, response = { clientDataJSON: b64(r.clientDataJSON) };
  if (r.attestationObject) response.attestationObject = b64(r.attestationObject);
  if (r.authenticatorData) { response.authenticatorData = b64(r.authenticatorData); response.signature = b64(r.signature); }
  if (r.userHandle) response.userHandle = b64(r.userHandle);
  return { id: cred.id, rawId: b64(cred.rawId), type: cred.type, response, clientExtensionResults: cred.getClientExtensionResults?.() || {} };
}

async function post(path, body) {
  const res = await authed(path, JSON_POST(body));
  const doc = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(doc.detail || `HTTP ${res.status}`);
  if (doc.proof) keep(doc.proof);
  return doc;
}

/** Create a passkey: the first one with the setup code, more from a signed-in page. */
export async function enrol({ setup = null, label = "passkey" } = {}) {
  const begun = await post("/auth/enrol/begin", { setup, label });
  const cred = await navigator.credentials.create({ publicKey: creationOptions(begun.options) });
  return post("/auth/enrol/finish", { challenge_id: begun.challenge_id, credential: credentialJSON(cred) });
}

export async function login() {
  const begun = await post("/auth/login/begin", {});
  const cred = await navigator.credentials.get({ publicKey: requestOptions(begun.options) });
  return post("/auth/login/finish", { challenge_id: begun.challenge_id, credential: credentialJSON(cred) });
}

export async function logout() {
  await fetch("/auth/logout", JSON_POST({})).catch(() => null);
  keep(null);
  location.reload();
}

export async function status() {
  const res = await authed("/auth/status").catch(() => null);
  return res && res.ok ? res.json().catch(() => null) : null;
}

const CSS = `
#signin { position: fixed; inset: 0; z-index: 50; display: grid; place-items: center; background: var(--sr-bg, #fff); color: var(--sr-fg, #000); font-family: var(--sr-text, sans-serif); }
#signin .box { width: min(26rem, 92vw); border: 1px solid var(--sr-line, #ccc); border-radius: var(--sr-radius, 8px); padding: 1.4rem; display: grid; gap: .8rem; }
#signin h1 { font-family: var(--sr-mono, monospace); font-size: 1rem; font-weight: 500; margin: 0; }
#signin p { margin: 0; color: var(--sr-fg-light, #444); font-size: .86rem; }
#signin input { border: 1px solid var(--sr-pill-border, #999); border-radius: 999px; padding: .45rem .8rem; background: var(--sr-bg, #fff); font-family: var(--sr-mono, monospace); font-size: .8rem; }
#signin button { font-family: var(--sr-mono, monospace); border-radius: 999px; padding: .5rem 1rem; border: 0; background: var(--sr-accent, #0a5); color: var(--sr-on-accent, #fff); cursor: pointer; }
#signin .err { color: var(--sr-bad, #c22); font-family: var(--sr-mono, monospace); font-size: .74rem; white-space: pre-wrap; }
`;

function el(tag, props = {}, ...kids) {
  const n = Object.assign(document.createElement(tag), props);
  n.append(...kids);
  return n;
}

/** The sign-in screen; ``state`` is GET /auth/status. */
export function showSignIn(state) {
  if (document.getElementById("signin")) return;
  document.head.append(el("style", { textContent: CSS }));
  const params = new URLSearchParams(location.search);
  const err = el("div", { className: "err", role: "alert" });
  const box = el("div", { className: "box" });
  const screen = el("div", { id: "signin" }, box);
  const done = () => { params.delete("setup"); const q = params.toString(); history.replaceState(null, "", location.pathname + (q ? `?${q}` : "")); location.reload(); };
  const run = (fn) => async () => { err.textContent = ""; try { await fn(); done(); } catch (e) { err.textContent = e.message || String(e); } };
  if (!window.isSecureContext || !window.PublicKeyCredential) {
    box.append(el("h1", { textContent: "Passkeys are not available here" }),
      el("p", { textContent: `Open http://localhost:${location.port}/ in a browser that supports passkeys.` }));
  } else if (state.setup_required) {
    const code = el("input", { value: params.get("setup") || "", placeholder: "setup code", autocomplete: "off", ariaLabel: "setup code" });
    const label = el("input", { value: "this computer", placeholder: "passkey name", ariaLabel: "passkey name" });
    box.append(el("h1", { textContent: "Create the first passkey" }),
      el("p", { textContent: "This dashboard has no passkey yet. Use the setup code it printed when it started (the link carries it)." }),
      code, label, el("button", { type: "button", textContent: "Create passkey", onclick: run(() => enrol({ setup: code.value.trim(), label: label.value.trim() })) }), err);
  } else {
    box.append(el("h1", { textContent: "Sign in" }),
      el("p", { textContent: "Use a passkey enrolled on this dashboard." }),
      el("button", { type: "button", textContent: "Sign in with a passkey", onclick: run(login) }), err);
  }
  document.body.append(screen);
}

/** true: go on (signed in, sign-in off, or no dashboard server answering). false: the sign-in screen is up. */
export async function signIn() {
  const state = await status();
  if (!state || state.enabled === false || state.signed_in) return true;
  showSignIn(state);
  return false;
}

/** A 401 (or a refused socket) mid-session: the session ended; back to the sign-in screen. */
export async function signedOut() {
  const state = await status();
  if (state && state.enabled !== false && !state.signed_in) showSignIn(state);
}
