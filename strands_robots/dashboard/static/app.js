var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import { r as reactExports, j as jsxRuntimeExports, c as client, R as React } from "./vendor/react.js";
(function polyfill() {
  const relList = document.createElement("link").relList;
  if (relList && relList.supports && relList.supports("modulepreload")) {
    return;
  }
  for (const link of document.querySelectorAll('link[rel="modulepreload"]')) {
    processPreload(link);
  }
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type !== "childList") {
        continue;
      }
      for (const node of mutation.addedNodes) {
        if (node.tagName === "LINK" && node.rel === "modulepreload")
          processPreload(node);
      }
    }
  }).observe(document, { childList: true, subtree: true });
  function getFetchOpts(link) {
    const fetchOpts = {};
    if (link.integrity) fetchOpts.integrity = link.integrity;
    if (link.referrerPolicy) fetchOpts.referrerPolicy = link.referrerPolicy;
    if (link.crossOrigin === "use-credentials")
      fetchOpts.credentials = "include";
    else if (link.crossOrigin === "anonymous") fetchOpts.credentials = "omit";
    else fetchOpts.credentials = "same-origin";
    return fetchOpts;
  }
  function processPreload(link) {
    if (link.ep)
      return;
    link.ep = true;
    const fetchOpts = getFetchOpts(link);
    fetch(link.href, fetchOpts);
  }
})();
const MIN_USEFUL_OPEN_MS = 2e4;
const MAX_RETRY_MS = 3e4;
const CHURN_OPENS_PER_MIN = 6;
const CHURN_FLOOR_MS = 5e3;
function backoffMs(attempt) {
  return Math.min(MAX_RETRY_MS, 1e3 * Math.pow(2, Math.max(0, attempt - 1)));
}
function planRetry({ attempt, frames, openMs, code, recentOpens, sessionExpired, pageRefused }) {
  if (sessionExpired) {
    return { attempt, delayMs: null, reason: "this sign-in has expired — sign in again" };
  }
  if (pageRefused && openMs === void 0 && frames === 0) {
    return { attempt, delayMs: null, reason: "this page is being refused — sign in again (the camera was never asked)" };
  }
  if (code === 1008 || code === 4401) {
    return { attempt, delayMs: null, reason: "the server refused this socket (unauthorized)" };
  }
  const shortLived = openMs === void 0 || openMs < MIN_USEFUL_OPEN_MS;
  const churning = shortLived && (recentOpens ?? 0) >= CHURN_OPENS_PER_MIN;
  if (churning) {
    return {
      attempt: Math.max(attempt, 1),
      delayMs: Math.max(CHURN_FLOOR_MS, Math.min(backoffMs(attempt), MAX_RETRY_MS)),
      reason: frames > 0 ? "this stream keeps dying after a few frames — the link may not sustain it, so retrying more slowly" : "this socket keeps reopening without proving anything — retrying more slowly"
    };
  }
  const proved = frames > 0 || openMs !== void 0 && openMs >= MIN_USEFUL_OPEN_MS;
  if (proved) {
    return {
      attempt: 1,
      delayMs: backoffMs(1),
      reason: frames > 0 ? "this socket delivered frames, so the drop is treated as a blip" : "this socket stayed open long enough to count as working"
    };
  }
  const next = attempt + 1;
  return {
    attempt: next,
    delayMs: backoffMs(next),
    reason: openMs !== void 0 && shortLived ? "accepted, then closed with nothing sent — counted as a failure, not a success" : "the socket never opened"
  };
}
const FRAME_LIVENESS_MAX_AGE_S = 15;
function frameProvesLiveness(input) {
  const { frameT, nowS } = input;
  const maxAgeS = input.maxAgeS ?? FRAME_LIVENESS_MAX_AGE_S;
  if (frameT == null || !Number.isFinite(frameT) || frameT <= 0) {
    return false;
  }
  const age = nowS - frameT;
  if (age < 0) return false;
  return age <= maxAgeS;
}
const PEER_STALE_S = 15;
function sweepStale(peers, nowS) {
  let changed = false;
  const next = { ...peers };
  for (const [id, peer] of Object.entries(next)) {
    const stale = nowS - (peer.last_seen ?? 0) > PEER_STALE_S;
    if (stale !== peer.stale) {
      next[id] = { ...peer, stale };
      changed = true;
    }
  }
  return changed ? next : peers;
}
function rebaseSnapshotPeers(peers, serverNowS, nowS) {
  if (!serverNowS) return peers;
  const out = {};
  for (const [id, peer] of Object.entries(peers)) {
    if (peer.last_seen === void 0) {
      out[id] = peer;
      continue;
    }
    const ageS2 = Math.max(0, serverNowS - peer.last_seen);
    out[id] = { ...peer, last_seen: nowS - ageS2 };
  }
  return out;
}
function mergeMeshEvent(peers, ev, nowS) {
  var _a, _b;
  const id = ev == null ? void 0 : ev.peer_id;
  switch (ev == null ? void 0 : ev.type) {
    case "snapshot":
      return rebaseSnapshotPeers(ev.peers ?? {}, ev.t, nowS);
    case "mesh_reconfigured":
      return {};
    case "presence":
    case "state":
    case "stream":
    case "pose":
    case "health":
    case "imu":
    case "odom": {
      if (!id) return peers;
      return { ...peers, [id]: { ...peers[id], peer_id: id, [ev.type]: ev.data, last_seen: nowS, stale: false } };
    }
    case "lidar": {
      if (!id) return peers;
      const kind = ev.kind === "state" ? "state" : "summary";
      const lidar = { ...(_a = peers[id]) == null ? void 0 : _a.lidar, [kind]: ev.data };
      return { ...peers, [id]: { ...peers[id], peer_id: id, lidar, last_seen: nowS, stale: false } };
    }
    case "camera_meta": {
      if (!id) return peers;
      const peer = peers[id];
      if (!peer) return peers;
      const cameras2 = { ...peer.cameras, [ev.cam]: ev.data };
      const fresh = frameProvesLiveness({ frameT: (_b = ev.data) == null ? void 0 : _b.t, nowS });
      return {
        ...peers,
        [id]: fresh ? { ...peer, cameras: cameras2, last_seen: nowS, stale: false } : { ...peer, cameras: cameras2 }
      };
    }
    default:
      return peers;
  }
}
function templateMatches(template, path) {
  if (!template.includes("{")) return template === path;
  const parts = template.split(/\{[^}]*\}/);
  const rx = new RegExp("^" + parts.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("[^/]+") + "$");
  return rx.test(path);
}
function normalisePath(path) {
  const bare = path.split("#")[0].split("?")[0];
  return bare.length > 1 && bare.endsWith("/") ? bare.slice(0, -1) : bare;
}
function routeKnown(livePaths, path) {
  if (!Array.isArray(livePaths) || livePaths.length === 0) return null;
  const wanted = normalisePath(path);
  return livePaths.some((t) => templateMatches(t, wanted));
}
function staleRouteMessage(path) {
  return `this dashboard's server does not have ${normalisePath(path)} — it is running code from before this feature existed. Restart the dashboard from a terminal to pick it up (the page itself is already new; a restart from an agent or daemon would come back with no camera access).`;
}
function unroutedByDetail(detail) {
  if (typeof detail !== "string") return false;
  const d = detail.trim().toLowerCase();
  return d.startsWith("no endpoint at") || d === "not found";
}
const MAX_LISTED = 6;
function isPlainObject(v) {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function nameOf(item) {
  if (typeof item === "string") return item.trim() || null;
  if (typeof item === "number" || typeof item === "boolean") return String(item);
  if (isPlainObject(item)) {
    for (const k of ["name", "peer_id", "id", "device_name", "dataset", "path"]) {
      const v = item[k];
      if (typeof v === "string" && v.trim()) return v.trim();
    }
  }
  return null;
}
function listPhrase(key, items) {
  const names = items.map(nameOf).filter((n) => !!n);
  const label2 = key.replace(/_/g, " ");
  if (names.length === 0) {
    return items.length ? `${label2}: ${items.length}` : null;
  }
  const shown = names.slice(0, MAX_LISTED).join(", ");
  return names.length > MAX_LISTED ? `${label2}: ${shown} and ${names.length - MAX_LISTED} more` : `${label2}: ${shown}`;
}
function validationSentence(items) {
  const parts = [];
  for (const item of items) {
    if (!isPlainObject(item)) return null;
    const msg = typeof item.msg === "string" ? item.msg : null;
    if (!msg) return null;
    const loc = Array.isArray(item.loc) ? item.loc.filter((x) => typeof x === "string" && x !== "body") : [];
    const field = loc.length ? String(loc[loc.length - 1]) : null;
    parts.push(field ? `${field}: ${msg}` : msg);
  }
  return parts.length ? parts.join("; ") : null;
}
function detailSentence(detail) {
  if (typeof detail === "string") return detail.trim();
  if (typeof detail === "number" || typeof detail === "boolean") return String(detail);
  if (detail === null || detail === void 0) return "";
  if (Array.isArray(detail)) {
    const strings = detail.map(nameOf).filter((n) => !!n);
    if (strings.length === detail.length && strings.length) return strings.join("; ");
    return validationSentence(detail) ?? JSON.stringify(detail);
  }
  if (!isPlainObject(detail)) return JSON.stringify(detail);
  const head = ["error", "message", "reason"].map((k) => detail[k]).find((v) => typeof v === "string" && v.trim());
  const because = typeof detail.detail === "string" && detail.detail.trim() ? detail.detail.trim() : null;
  const hint = typeof detail.hint === "string" && detail.hint.trim() ? detail.hint.trim() : null;
  if (!head && !because && !hint) return JSON.stringify(detail);
  const lists = [];
  for (const [k, v] of Object.entries(detail)) {
    if (k === "error" || k === "message" || k === "reason" || k === "detail" || k === "hint") continue;
    if (Array.isArray(v)) {
      const phrase = listPhrase(k, v);
      if (phrase) lists.push(phrase);
    }
  }
  let text = [head == null ? void 0 : head.trim(), because].filter(Boolean).join(" — ");
  if (hint) text = text ? `${text} — ${hint}` : hint;
  if (lists.length) text += ` (${lists.join("; ")})`;
  return text;
}
const BASE_KEY = "strands.backend";
const TOKEN_KEY = "strands.token";
function normalize(raw) {
  const value = (raw ?? "").trim();
  if (!value) return "";
  const withScheme = /^[a-z]+:\/\//i.test(value) ? value : `http://${value}`;
  try {
    const url = new URL(withScheme);
    if (url.protocol === "ws:") url.protocol = "http:";
    if (url.protocol === "wss:") url.protocol = "https:";
    if (url.protocol !== "http:" && url.protocol !== "https:") return "";
    return url.origin;
  } catch {
    return "";
  }
}
let cachedBase = null;
let absorbedUrl = false;
let urlBase = null;
function absorbUrl() {
  if (absorbedUrl) return;
  absorbedUrl = true;
  try {
    const params = new URLSearchParams(location.search);
    const fromToken = params.get("token");
    if (fromToken) localStorage.setItem(TOKEN_KEY, fromToken);
    urlBase = params.get("backend");
    if (fromToken !== null || urlBase !== null) {
      try {
        params.delete("token");
        params.delete("backend");
        const rest = params.toString();
        history.replaceState(null, "", `${location.pathname}${rest ? `?${rest}` : ""}${location.hash || ""}`);
      } catch {
      }
    }
  } catch {
    urlBase = null;
  }
}
function backendBase() {
  absorbUrl();
  if (cachedBase !== null) return cachedBase;
  if (urlBase !== null) {
    cachedBase = normalize(urlBase);
    localStorage.setItem(BASE_KEY, cachedBase);
    return cachedBase;
  }
  cachedBase = normalize(localStorage.getItem(BASE_KEY) ?? "");
  return cachedBase;
}
function authToken() {
  absorbUrl();
  return (localStorage.getItem(TOKEN_KEY) ?? "").trim();
}
const authListeners = /* @__PURE__ */ new Set();
function subscribeAuth(fn) {
  authListeners.add(fn);
  return () => {
    authListeners.delete(fn);
  };
}
function notifyAuth() {
  for (const fn of authListeners) fn();
}
function setAuthToken(token) {
  const value = token.trim();
  if (value) localStorage.setItem(TOKEN_KEY, value);
  else localStorage.removeItem(TOKEN_KEY);
  notifyAuth();
}
function backendLabel() {
  const base = backendBase();
  return base ? base.replace(/^https?:\/\//, "") : `${location.host} (this origin)`;
}
function backendKey() {
  return `${backendBase()}|${authToken() ? "auth" : "open"}`;
}
function setBackendBase(raw) {
  cachedBase = normalize(raw);
  if (cachedBase) localStorage.setItem(BASE_KEY, cachedBase);
  else localStorage.removeItem(BASE_KEY);
  forgetLiveRoutes();
  notifyAuth();
}
function apiUrl(path) {
  const base = backendBase();
  return base ? `${base}${path}` : path;
}
function wsUrl(path) {
  const base = backendBase();
  const origin = base || location.origin;
  const url = new URL(path, origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
class HttpError extends Error {
  constructor(status, message, body) {
    super(message);
    __publicField(this, "status");
    __publicField(this, "body");
    this.name = "HttpError";
    this.status = status;
    this.body = body;
  }
}
let _liveRoutes = null;
let _liveRoutesTried = false;
let _liveRoutesAt = 0;
const LIVE_ROUTES_TTL_MS = 6e4;
async function serverRoutePaths() {
  return liveRoutes();
}
async function liveRoutes() {
  if (_liveRoutesTried && Date.now() - _liveRoutesAt < LIVE_ROUTES_TTL_MS) return _liveRoutes;
  _liveRoutesTried = true;
  _liveRoutesAt = Date.now();
  try {
    const token = authToken();
    const res = await fetch(apiUrl("/openapi.json"), {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    if (!res.ok) {
      noteAuthRefusal(res.status);
      return null;
    }
    noteAuthAccepted("/openapi.json");
    const doc = await res.json();
    const paths = doc && doc.paths && typeof doc.paths === "object" ? Object.keys(doc.paths) : [];
    _liveRoutes = paths.length ? paths : null;
  } catch {
    _liveRoutes = null;
  }
  return _liveRoutes;
}
function forgetLiveRoutes() {
  _liveRoutes = null;
  _liveRoutesTried = false;
  _liveRoutesAt = 0;
}
let _refusedAt = null;
function noteAuthRefusal(status, at = Date.now()) {
  if (status === 401 || status === 403) _refusedAt = at;
}
const PROVES_NOTHING = [
  "/api/health",
  "/api/auth/status",
  "/api/auth/register/",
  "/api/auth/login/"
];
function noteAuthAccepted(path) {
  if (path !== void 0 && PROVES_NOTHING.some((p) => path.startsWith(p))) return;
  _refusedAt = null;
}
function authRefusedRecently(withinMs = 6e4, now = Date.now()) {
  return _refusedAt !== null && now - _refusedAt <= withinMs;
}
let lastRenewalAtS = 0;
function lastRenewalAt() {
  return lastRenewalAtS;
}
function absorbRenewedSession(res) {
  var _a;
  let offered = null;
  try {
    offered = ((_a = res == null ? void 0 : res.headers) == null ? void 0 : _a.get("X-Session-Token")) ?? null;
  } catch {
    return false;
  }
  const fresh = (offered ?? "").trim();
  if (!fresh) return false;
  const current = (localStorage.getItem(TOKEN_KEY) ?? "").trim();
  if (!current || fresh === current) return false;
  if (fresh.split(".").length !== 3) return false;
  setAuthToken(fresh);
  lastRenewalAtS = Date.now() / 1e3;
  return true;
}
async function api(path, init = {}) {
  const token = authToken();
  const headers = { ...init.headers };
  if (init.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(apiUrl(path), { ...init, headers });
  } catch (e) {
    throw new HttpError(0, `cannot reach ${backendLabel()}: ${e instanceof Error ? e.message : e}`);
  }
  absorbRenewedSession(res);
  const text = await res.text();
  let body = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
  }
  if (!res.ok) {
    noteAuthRefusal(res.status);
    const detail = body && (body.detail ?? body.error) || text || res.statusText;
    let message = detailSentence(detail) || text || res.statusText;
    if (res.status === 404 && (routeKnown(await liveRoutes(), path) === false || unroutedByDetail(body && (body.detail ?? null)))) {
      message = staleRouteMessage(path);
    }
    throw new HttpError(res.status, message, body);
  }
  noteAuthAccepted(path);
  return body;
}
const post = (path, body) => api(path, { method: "POST", body: body === void 0 ? "{}" : JSON.stringify(body) });
const del = (path) => api(path, { method: "DELETE" });
async function apiBlob(path) {
  const token = authToken();
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(apiUrl(path), { headers });
  } catch (e) {
    throw new HttpError(0, `cannot reach ${backendLabel()}: ${e instanceof Error ? e.message : e}`);
  }
  absorbRenewedSession(res);
  if (!res.ok) {
    noteAuthRefusal(res.status);
    const text = await res.text();
    let detail = text || res.statusText;
    try {
      detail = JSON.parse(text).detail ?? detail;
    } catch {
    }
    throw new HttpError(res.status, detailSentence(detail) || text || res.statusText);
  }
  noteAuthAccepted(path);
  return URL.createObjectURL(await res.blob());
}
const EXPIRING_SOON_S = 300;
function decodeSegment(seg) {
  try {
    const norm2 = seg.replace(/-/g, "+").replace(/_/g, "/");
    const pad = "=".repeat((4 - norm2.length % 4) % 4);
    const bin = atob(norm2 + pad);
    return decodeURIComponent(Array.from(bin, (c) => "%" + c.charCodeAt(0).toString(16).padStart(2, "0")).join(""));
  } catch {
    return null;
  }
}
function tokenExpiry(token) {
  const raw = (token ?? "").trim();
  if (!raw) return null;
  const parts = raw.split(".");
  if (parts.length !== 3) return null;
  const json = decodeSegment(parts[1]);
  if (!json) return null;
  try {
    const claims = JSON.parse(json);
    const exp = claims == null ? void 0 : claims.exp;
    return typeof exp === "number" && Number.isFinite(exp) ? exp : null;
  } catch {
    return null;
  }
}
function humaniseSeconds(s) {
  const abs = Math.abs(s);
  if (abs < 90) return `${Math.round(abs)} seconds`;
  if (abs < 5400) return `${Math.round(abs / 60)} minutes`;
  const hours = abs / 3600;
  const shown = hours < 10 ? Number(hours.toFixed(1)) : Math.round(hours);
  return `${shown} hour${shown === 1 ? "" : "s"}`;
}
function sessionVerdict(token, nowS, renewedAtS = 0) {
  const raw = (token ?? "").trim();
  if (!raw) {
    return { state: "none", expiresInS: null, text: null, refusesUntilSignIn: false };
  }
  const exp = tokenExpiry(raw);
  if (exp === null) {
    return { state: "opaque", expiresInS: null, text: null, refusesUntilSignIn: false };
  }
  const left = exp - nowS;
  if (left <= 0) {
    return {
      state: "expired",
      expiresInS: left,
      // The two facts the operator needs: it is not the robot's fault, and one tap fixes it.
      text: `this sign-in expired ${humaniseSeconds(left)} ago — sign in again to see cameras and control the fleet. Nothing is wrong with the robots; the page is being refused.`,
      refusesUntilSignIn: true
    };
  }
  if (left <= EXPIRING_SOON_S) {
    return {
      state: "expiring",
      expiresInS: left,
      text: renewedAtS > 0 ? `this sign-in lapses in ${humaniseSeconds(left)} and is no longer being renewed — this page renewed it automatically before, so the connection is now being refused or the session hit its 30-day maximum. Sign in again before starting a recording.` : `this sign-in lapses in ${humaniseSeconds(left)} — sign in again before starting a recording, or it will be refused part-way through.`,
      refusesUntilSignIn: false
    };
  }
  return { state: "valid", expiresInS: left, text: null, refusesUntilSignIn: false };
}
const ACTIVITY_CAP = 200;
function useMesh() {
  const [conn, setConn] = reactExports.useState("connecting");
  const [dashboardId, setDashboardId] = reactExports.useState("");
  const [peers, setPeers] = reactExports.useState({});
  const [safetyFlash, setSafetyFlash] = reactExports.useState(null);
  const [mesh, setMesh] = reactExports.useState({});
  const [absentChildren, setAbsentChildren] = reactExports.useState([]);
  const [quietChildren, setQuietChildren] = reactExports.useState([]);
  const [activity, setActivity] = reactExports.useState([]);
  const [loaded, setLoaded] = reactExports.useState(false);
  const [lastEventAt, setLastEventAt] = reactExports.useState(void 0);
  const [everOpen, setEverOpen] = reactExports.useState(false);
  const retryRef = reactExports.useRef(0);
  reactExports.useEffect(() => {
    let ws = null;
    let closed = false;
    let flashTimer;
    let retryTimer;
    let framesThisSocket = 0;
    let openedAt;
    const connect = () => {
      if (closed) return;
      framesThisSocket = 0;
      openedAt = void 0;
      ws = new WebSocket(wsUrl("/ws/mesh"));
      setConn("connecting");
      ws.onopen = () => {
        openedAt = Date.now();
        setConn("open");
        setEverOpen(true);
      };
      ws.onclose = (ev) => {
        const plan = planRetry({
          attempt: retryRef.current,
          frames: framesThisSocket,
          openMs: openedAt !== void 0 ? Date.now() - openedAt : void 0,
          code: ev.code,
          sessionExpired: sessionVerdict(authToken(), Date.now() / 1e3).refusesUntilSignIn,
          pageRefused: authRefusedRecently()
        });
        retryRef.current = plan.attempt;
        if (plan.delayMs === null) {
          setConn("unauthorized");
          return;
        }
        setConn("closed");
        if (!closed) retryTimer = setTimeout(connect, plan.delayMs);
      };
      ws.onmessage = (msg) => {
        let ev;
        try {
          ev = JSON.parse(msg.data);
        } catch {
          return;
        }
        framesThisSocket += 1;
        setLastEventAt(Date.now());
        switch (ev.type) {
          case "snapshot":
            setDashboardId(ev.dashboard_peer_id);
            setPeers((p) => mergeMeshEvent(p, ev, Date.now() / 1e3));
            if (ev.mesh) setMesh(ev.mesh);
            setAbsentChildren(Array.isArray(ev.absent_children) ? ev.absent_children : []);
            setQuietChildren(Array.isArray(ev.managed_no_presence) ? ev.managed_no_presence : []);
            setLoaded(true);
            break;
          case "presence":
          case "state":
          case "stream":
          case "camera_meta":
          case "pose":
          case "health":
          case "imu":
          case "odom":
          case "lidar":
            setPeers((p) => mergeMeshEvent(p, ev, Date.now() / 1e3));
            break;
          case "safety":
            setSafetyFlash(ev.kind);
            clearTimeout(flashTimer);
            flashTimer = setTimeout(() => setSafetyFlash(null), 5e3);
            break;
          case "activity":
            setActivity((a) => [ev.data, ...a].slice(0, ACTIVITY_CAP));
            break;
          case "mesh_reconfigured":
            setMesh(ev.mesh);
            setPeers((p) => mergeMeshEvent(p, ev, Date.now() / 1e3));
            break;
        }
      };
    };
    connect();
    const sweep = setInterval(() => {
      const now = Date.now() / 1e3;
      setPeers((p) => sweepStale(p, now));
    }, 5e3);
    return () => {
      closed = true;
      clearInterval(sweep);
      if (retryTimer) clearTimeout(retryTimer);
      if (flashTimer) clearTimeout(flashTimer);
      ws == null ? void 0 : ws.close();
    };
  }, []);
  return { conn, dashboardId, peers, safetyFlash, mesh, activity, absentChildren, quietChildren, loaded, lastEventAt, everOpen };
}
function bundleAgeText(loadedAtMs, nowMs) {
  if (loadedAtMs === null || !Number.isFinite(loadedAtMs)) return null;
  const s = (nowMs - loadedAtMs) / 1e3;
  if (s < 0) return null;
  if (s < 90) return "just now";
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${(s / 3600).toFixed(1)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
function reloadImpact(runningPeerIds) {
  const names = [...new Set(runningPeerIds.map((n) => (n ?? "").trim()).filter(Boolean))];
  if (names.length === 0) {
    return {
      busy: false,
      text: "Nothing is running right now — a good moment to reload. Camera streams reconnect by themselves."
    };
  }
  const who = names.length === 1 ? names[0] : names.length === 2 ? `${names[0]} and ${names[1]}` : `${names[0]}, ${names[1]} and ${names.length - 2} more`;
  return {
    busy: true,
    text: `${who} ${names.length === 1 ? "is" : "are"} running — the task itself keeps running on the robot, but reloading drops the camera streams and anything typed into a form. Between runs is safer.`
  };
}
function wakeLockAction(s) {
  if (!s.supported) return "none";
  if (s.want) {
    if (s.held) return "none";
    return s.visible ? "request" : "none";
  }
  return s.held ? "release" : "none";
}
function wakeLockNote(s) {
  if (!s.want) return null;
  if (!s.supported) return "this browser cannot keep the screen awake — sleep may drop the camera view";
  return s.held ? null : "screen sleep is not being prevented yet";
}
function usePwa() {
  const needRefresh = false;
  const [online, setOnline] = reactExports.useState(navigator.onLine);
  const loadedAtRef = reactExports.useRef(Date.now());
  const [installable, setInstallable] = reactExports.useState(false);
  const promptRef = reactExports.useRef(null);
  const wakeRef = reactExports.useRef(null);
  const wantAwakeRef = reactExports.useRef(false);
  reactExports.useEffect(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    const onPrompt = (e) => {
      e.preventDefault();
      promptRef.current = e;
      setInstallable(true);
    };
    window.addEventListener("beforeinstallprompt", onPrompt);
    const onInstalled = () => {
      setInstallable(false);
      promptRef.current = null;
    };
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);
  const install = reactExports.useCallback(async () => {
    const prompt = promptRef.current;
    if (!prompt) return;
    promptRef.current = null;
    setInstallable(false);
    try {
      await prompt.prompt();
    } catch {
    }
  }, []);
  const update = reactExports.useCallback(() => {
    location.reload();
  }, []);
  const applyWakeLock = reactExports.useCallback(async () => {
    var _a, _b;
    const anyNav = navigator;
    const action = wakeLockAction({
      want: wantAwakeRef.current,
      held: !!wakeRef.current,
      visible: document.visibilityState === "visible",
      supported: !!anyNav.wakeLock
    });
    if (action === "request") {
      try {
        wakeRef.current = await anyNav.wakeLock.request("screen");
        (_b = (_a = wakeRef.current).addEventListener) == null ? void 0 : _b.call(_a, "release", () => {
          wakeRef.current = null;
        });
      } catch {
      }
    } else if (action === "release") {
      try {
        await wakeRef.current.release();
      } catch {
      }
      wakeRef.current = null;
    }
  }, []);
  const keepAwake = reactExports.useCallback(async (want) => {
    wantAwakeRef.current = want;
    await applyWakeLock();
  }, [applyWakeLock]);
  reactExports.useEffect(() => {
    const onVisible = () => {
      void applyWakeLock();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [applyWakeLock]);
  const standalone = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  return {
    online,
    needRefresh,
    update,
    installable,
    install,
    keepAwake,
    standalone,
    /** honest word about the screen: null when there is nothing to say (see lib/wakeLock) */
    wakeNote: () => wakeLockNote({
      want: wantAwakeRef.current,
      held: !!wakeRef.current,
      supported: !!navigator.wakeLock
    }),
    /** how long this tab has been running the bundle it loaded, for the update prompt */
    bundleAge: () => bundleAgeText(loadedAtRef.current, Date.now())
  };
}
const LIVE = {
  kind: "live",
  commandsWork: true,
  misleading: false,
  headline: "",
  detail: ""
};
const PHYSICAL = "Use the arms’ power switch — that is the only brake that does not go through this page.";
function linkHealth(i) {
  const stallMs = i.stallMs ?? 2e4;
  const showing = i.peerCount > 0;
  if (!i.browserOnline) {
    return {
      kind: "device-offline",
      commandsWork: false,
      misleading: showing,
      headline: "This device is offline",
      detail: showing ? `The fleet below is a cached snapshot. Commands and 🛑 STOP ALL cannot leave this device. ${PHYSICAL}` : "Commands cannot leave this device until its network is back.",
      estopReason: "this device has no network — STOP ALL cannot be sent"
    };
  }
  if (i.conn === "unauthorized") {
    if (i.sessionExpired) {
      return {
        kind: "unauthorized",
        commandsWork: false,
        misleading: showing,
        headline: "Your sign-in has expired",
        // The fix is one tap and it is on this page, so say that before anything else.
        detail: `Sign in again to command the fleet — nothing is wrong with the robots, this page is being refused. Until then 🛑 STOP ALL cannot be sent. ${PHYSICAL}`,
        estopReason: "this sign-in has expired — STOP ALL will be refused until you sign in again"
      };
    }
    return {
      kind: "unauthorized",
      commandsWork: false,
      misleading: showing,
      headline: "The server rejected this session",
      detail: "Every command, including 🛑 STOP ALL, will be refused until the token is accepted again.",
      estopReason: "the server is rejecting this session — STOP ALL will be refused"
    };
  }
  if (i.conn === "open" && i.meshOnline === false) {
    return {
      kind: "mesh-down",
      commandsWork: false,
      misleading: showing,
      headline: "This dashboard is not on the robot mesh",
      detail: (showing ? "The API is up and this page is connected to it, but its mesh session is down — the fleet below is the last thing the mesh reported and the robots keep doing whatever they were last told. No command, including 🛑 STOP ALL, can reach them. " : "The API is up and this page is connected to it, but its mesh session is down, so no robot can be seen or commanded. ") + PHYSICAL,
      estopReason: "the dashboard's mesh session is down — STOP ALL cannot reach any robot"
    };
  }
  if (i.conn === "open") {
    if (showing && i.lastEventAt !== void 0 && i.now - i.lastEventAt > stallMs) {
      return {
        kind: "stalled",
        commandsWork: true,
        misleading: true,
        headline: "No fleet updates for a while",
        detail: `The socket is open but has sent nothing for ${Math.round((i.now - i.lastEventAt) / 1e3)}s, so what you see below may no longer be true. Commands should still get through.`
      };
    }
    return LIVE;
  }
  if (i.conn === "connecting" && !i.everOpen) {
    return { ...LIVE, kind: "connecting", commandsWork: false };
  }
  const frozenFor = i.lastEventAt !== void 0 ? Math.round((i.now - i.lastEventAt) / 1e3) : void 0;
  const reconnecting = i.conn === "connecting";
  return {
    kind: "lost",
    commandsWork: false,
    misleading: showing,
    headline: reconnecting ? "Reconnecting to the dashboard API…" : "Disconnected from the dashboard API",
    detail: showing ? `The fleet below is frozen${frozenFor !== void 0 ? ` (${frozenFor}s old)` : ""} and the robots keep doing whatever they were last told. 🛑 STOP ALL cannot reach them from here. ${PHYSICAL}` : "The dashboard API is unreachable from this browser.",
    estopReason: "the dashboard API is unreachable — STOP ALL cannot be delivered"
  };
}
function estopPosture(v) {
  return v.commandsWork ? { degraded: false, title: "Stop every robot on the mesh - keyboard shortcut: ." } : { degraded: true, title: `⚠ ${v.estopReason ?? "the link is down"} — still worth pressing, it is sent the moment the link returns. ${PHYSICAL}` };
}
const DISMISS_KEY = "lanHintDismissed";
function lanHintVerdict(args) {
  const { body, origin, dismissed } = args;
  if (!body) return { show: false, reason: "no answer from the server (old build, or offline)" };
  if (body.same_network !== true) {
    return { show: false, reason: body.same_network === false ? "viewer is on another network" : "server could not tell" };
  }
  const urls = (body.lan_urls || []).filter((u) => typeof u === "string" && u.startsWith("http://"));
  if (!urls.length) return { show: false, reason: "local, but the server named no address to offer" };
  const here = urls.find((u) => sameOrigin(u, origin));
  if (here) return { show: false, reason: "this page is already served from the local address" };
  const url = urls[0];
  if (dismissed.includes(url)) return { show: false, reason: "dismissed for this address" };
  return {
    show: true,
    url,
    text: `You are on the same network as this dashboard. ${url} skips the trip out to the internet and back — camera streams stall on that round trip.`,
    reason: "local viewer coming in over the tunnel"
  };
}
function sameOrigin(a, b) {
  try {
    const ua = new URL(a);
    const ub = new URL(b);
    return ua.hostname === ub.hostname && (ua.port || "80") === (ub.port || "80");
  } catch {
    return false;
  }
}
function handoffHref(url, res) {
  const token = res && typeof res.token === "string" && res.token.trim() ? res.token.trim() : null;
  if (!token) return url;
  try {
    const u = new URL(url);
    u.searchParams.set("token", token);
    return u.toString();
  } catch {
    return url;
  }
}
function readDismissed(store) {
  try {
    const raw = store == null ? void 0 : store.getItem(DISMISS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}
function LanHint() {
  const [body, setBody] = reactExports.useState(null);
  const [dismissed, setDismissed] = reactExports.useState(
    () => readDismissed(typeof localStorage === "undefined" ? null : localStorage)
  );
  const [leaving, setLeaving] = reactExports.useState(false);
  reactExports.useEffect(() => {
    let alive = true;
    api("/api/network/hint").then((b) => {
      if (alive) setBody(b);
    }).catch(() => {
    });
    return () => {
      alive = false;
    };
  }, []);
  const verdict2 = lanHintVerdict({
    body,
    origin: typeof location === "undefined" ? "" : location.origin,
    dismissed
  });
  if (!verdict2.show) return null;
  const dismiss = () => {
    const next = [...dismissed, verdict2.url];
    setDismissed(next);
    try {
      localStorage.setItem(DISMISS_KEY, JSON.stringify(next));
    } catch {
    }
  };
  const go = async (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
    e.preventDefault();
    if (leaving) return;
    setLeaving(true);
    let href = verdict2.url;
    try {
      href = handoffHref(verdict2.url, await post("/api/auth/handoff"));
    } catch {
    }
    location.href = href;
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "lan-hint", role: "status", style: { gridColumn: "1 / -1" }, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("span", { "aria-hidden": "true", children: "🏠" }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "lan-hint-text", children: verdict2.text }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("a", { className: "lan-hint-go", href: verdict2.url, onClick: go, children: leaving ? "carrying your sign-in over…" : "open the local address" }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "lan-hint-dismiss", onClick: dismiss, "aria-label": "dismiss this hint", children: "×" })
  ] });
}
const BUNDLE_ROUTES = [
  "/api/activity",
  "/api/agent",
  "/api/auth/credentials",
  "/api/auth/credentials/{p}",
  "/api/auth/handoff",
  "/api/auth/login/",
  "/api/auth/login/begin",
  "/api/auth/login/finish",
  "/api/auth/register/",
  "/api/auth/register/begin",
  "/api/auth/register/finish",
  "/api/auth/status",
  "/api/calibration",
  "/api/calibration/run",
  "/api/calibration/run/{p}",
  "/api/calibration/run/{p}/cancel",
  "/api/calibration/run/{p}/key",
  "/api/calibration/{p}",
  "/api/checkpoints/search",
  "/api/collect",
  "/api/config",
  "/api/consent",
  "/api/consent/revoke",
  "/api/datasets/labels",
  "/api/deploy/snippet",
  "/api/devices",
  "/api/devices/arm-role",
  "/api/devices/camera/{p}/modes",
  "/api/devices/camera/{p}/preview",
  "/api/devices/despawn",
  "/api/devices/logs/{p}",
  "/api/devices/profiles",
  "/api/devices/spawn",
  "/api/devices/spawn-remembered",
  "/api/devices/{p}/cameras",
  "/api/fleet",
  "/api/health",
  "/api/mesh/restart",
  "/api/mesh/safety/estop",
  "/api/mesh/safety/resume",
  "/api/network/hint",
  "/api/policies/validate",
  "/api/record/close",
  "/api/record/episode/discard",
  "/api/record/episode/redo",
  "/api/record/episode/start",
  "/api/record/episode/stop",
  "/api/record/open",
  "/api/record/session",
  "/api/record/upload-preflight",
  "/api/replay",
  "/api/robots/registry",
  "/api/robots/{p}/policy-fit",
  "/api/robots/{p}/stop",
  "/api/robots/{p}/task",
  "/api/robots/{p}/teleop",
  "/api/robots/{p}/teleop/publish",
  "/api/robots/{p}/teleop/receive",
  "/api/robots/{p}/teleop/stop",
  "/api/robots/{p}/twin",
  "/api/safety",
  "/api/safety/estop",
  "/api/safety/resume",
  "/api/safety/{p}",
  "/api/settings",
  "/api/sim",
  "/api/sim/ports",
  "/api/sim/{p}",
  "/api/sim/{p}/joints",
  "/api/sim/{p}/reset",
  "/api/sim/{p}/stream.mjpg",
  "/api/training/datasets",
  "/api/training/export",
  "/api/training/jobs",
  "/api/training/output-dir",
  "/api/training/status",
  "/api/training/submit",
  "/api/training/trainers",
  "/api/training/validate"
];
function darkRoutes(livePaths, needed = BUNDLE_ROUTES) {
  if (!Array.isArray(livePaths) || livePaths.length === 0) return [];
  return needed.filter((p) => p.endsWith("/") ? !livePaths.some((t) => t.startsWith(p)) : routeKnown(livePaths, p) === false).sort();
}
function darkFeatureMessage(dark) {
  if (dark.length === 0) return null;
  const n = dark.length;
  return `${n} ${n === 1 ? "feature" : "features"} on this page ${n === 1 ? "is" : "are"} dark: the server answering here is running older code and has no route for ${n === 1 ? "it" : "them"}. Restarting the dashboard from a terminal lights ${n === 1 ? "it" : "them"} up — not from a background daemon, which macOS can never grant camera access to.`;
}
const AGO = (since, now) => {
  if (!since) return "";
  const s = Math.max(0, Math.round(now / 1e3 - since));
  if (s < 90) return ` ${s}s ago`;
  if (s < 5400) return ` ${Math.round(s / 60)}m ago`;
  return ` ${Math.round(s / 3600)}h ago`;
};
function lockoutBadge(v, now = Date.now()) {
  const state = (v == null ? void 0 : v.state) ?? null;
  if (state === "locked") {
    const who = (v == null ? void 0 : v.by) ? ` by ${v.by}` : "";
    return {
      label: "e-stop locked",
      tone: "locked",
      title: `This robot is refusing every command except status${who}${AGO(v == null ? void 0 : v.since, now)}.
` + ((v == null ? void 0 : v.reason) ? `${v.reason}.
` : "") + "Clearing it needs the operator override code (Safety > resume). Resuming moves nothing:\nit only stops commands being refused."
    };
  }
  if (state === "unknown" && (v == null ? void 0 : v.since)) {
    return {
      label: "e-stop?",
      tone: "doubt",
      title: `The e-stop state of this robot is unknown${AGO(v == null ? void 0 : v.since, now)}.
` + ((v == null ? void 0 : v.reason) ? `${v.reason}.
` : "") + "It will read as clear again as soon as it accepts a command a lockout would refuse."
    };
  }
  return { label: null, tone: null, title: "" };
}
function lockoutBanner(peers) {
  const locked = peers.filter((p) => {
    var _a;
    return ((_a = p.lockout) == null ? void 0 : _a.state) === "locked";
  });
  if (locked.length) {
    const by = locked.map((p) => {
      var _a;
      return (_a = p.lockout) == null ? void 0 : _a.by;
    }).find(Boolean);
    const who = by ? ` by ${by}` : "";
    const names = locked.map((p) => p.peer_id).join(", ");
    return {
      severity: "bad",
      text: locked.length === 1 ? `${names} is e-stop locked${who} — it refuses every command except status. Safety > resume needs the override code.` : `${locked.length} robots are e-stop locked${who} (${names}) — they refuse every command except status. Safety > resume needs the override code.`
    };
  }
  const doubt = peers.filter((p) => {
    var _a, _b;
    return ((_a = p.lockout) == null ? void 0 : _a.state) === "unknown" && ((_b = p.lockout) == null ? void 0 : _b.since);
  });
  if (doubt.length) {
    return {
      severity: "warn",
      text: `E-stop state unknown for ${doubt.length === 1 ? doubt[0].peer_id : `${doubt.length} robots`}: a resume was broadcast, but each robot verifies the override code itself, so this is not proof any of them cleared.`
    };
  }
  return null;
}
let _composite = null;
function composeConfigDoc(s, agent) {
  const t = s.settings ?? {};
  const a = t.agent ?? {};
  const v = t.voice ?? {};
  const m = t.mesh ?? {};
  const sec = t.security ?? {};
  return {
    agent: {
      model_id: a.model_id ?? null,
      known_models: (agent == null ? void 0 : agent.model) ? [agent.model] : [],
      system_prompt: a.system_prompt ?? "",
      is_default_prompt: !a.system_prompt,
      temperature: a.temperature ?? null,
      max_tokens: a.max_tokens ?? null,
      built: !!agent
    },
    voice: { provider: v.provider ?? "openai", voice_name: v.voice_name ?? null, providers: ["openai", "nova_sonic"] },
    mesh: {
      connect: m.connect ?? [],
      listen: m.listen ?? [],
      port: m.port ?? void 0,
      backend: m.backend ?? void 0,
      camera_hz: m.camera_hz ?? void 0,
      policy_allow: m.policy_type_allow ?? [],
      settings: m
    },
    runtime: { trust_remote_code: !!(t.runtime ?? {}).trust_remote_code },
    security: { auth_enabled: !!sec.auth_token, cors_origins: sec.cors_origins ?? [] },
    policies: [],
    env: [],
    env_file: "",
    settings_file: s.file ?? ""
  };
}
async function fetchConfigDoc() {
  if (_composite !== false) {
    try {
      const doc = await api("/api/config");
      _composite = true;
      return doc;
    } catch (e) {
      if (!(e instanceof HttpError) || e.status !== 404) throw e;
      _composite = false;
    }
  }
  const [settings, agent] = await Promise.all([
    api("/api/settings"),
    api("/api/agent").catch(() => null)
  ]);
  return composeConfigDoc(settings, agent);
}
const SETTINGS_SECTIONS = /* @__PURE__ */ new Set(["agent", "voice", "mesh", "runtime", "security"]);
function toSettingsPatch(body) {
  const patch = {};
  const ignored = [];
  for (const [k, v] of Object.entries(body)) {
    if (SETTINGS_SECTIONS.has(k) && v && typeof v === "object") patch[k] = v;
    else ignored.push(k);
  }
  return { patch, ignored };
}
async function saveConfigDoc(body) {
  if (_composite !== false) {
    try {
      return await post("/api/config", body);
    } catch (e) {
      if (!(e instanceof HttpError) || e.status !== 404) throw e;
      _composite = false;
    }
  }
  const { patch, ignored } = toSettingsPatch(body);
  if (!Object.keys(patch).length) {
    return { applied: [], restart_required: [], env_written: [], skipped_masked: [], agent_reset: false, errors: [], ignored };
  }
  const r = await post("/api/settings", patch);
  const changed = r.changed ?? [];
  return {
    applied: changed,
    // Settings main stores but only a new process reads: main has no hot re-point yet.
    startup_required: changed.filter((k) => k.startsWith("mesh.") || k.startsWith("security.")),
    restart_required: [],
    env_written: [],
    skipped_masked: [],
    agent_reset: changed.some((k) => k.startsWith("agent.")),
    errors: r.errors ?? [],
    ignored
  };
}
const CTX = reactExports.createContext({
  config: null,
  policies: [],
  provider: () => void 0,
  loading: true,
  error: null,
  reload: async () => {
  },
  save: async () => ({ applied: [], restart_required: [], env_written: [], skipped_masked: [], agent_reset: false, errors: [] })
});
function ConfigProvider({ children }) {
  const [config, setConfig] = reactExports.useState(null);
  const [loading, setLoading] = reactExports.useState(true);
  const [error, setError] = reactExports.useState(null);
  const reload = reactExports.useCallback(async () => {
    setLoading(true);
    try {
      setConfig(await fetchConfigDoc());
      setError(null);
    } catch (e) {
      setError((e == null ? void 0 : e.message) ?? String(e));
    } finally {
      setLoading(false);
    }
  }, []);
  reactExports.useEffect(() => {
    void reload();
  }, [reload]);
  const save = reactExports.useCallback(async (body) => {
    const result = await saveConfigDoc(body);
    await reload();
    return result;
  }, [reload]);
  const policies = (config == null ? void 0 : config.policies) ?? [];
  const byName = reactExports.useMemo(() => new Map(policies.map((p) => [p.name, p])), [policies]);
  const provider = reactExports.useCallback((name) => byName.get(name), [byName]);
  return /* @__PURE__ */ jsxRuntimeExports.jsx(CTX.Provider, { value: { config, policies, provider, loading, error, reload, save }, children });
}
const useConfig = () => reactExports.useContext(CTX);
const NOTHING = { text: "", vague: false };
function serverNotice(block) {
  if (!block || block.storm !== true) return NOTHING;
  const recent = block.recent ?? 0;
  if (recent <= 0) return NOTHING;
  const worst = block.worst;
  if (!(worst == null ? void 0 : worst.client)) {
    return {
      text: `Something is being refused by this server — ${recent} handshake(s) in the last ${Math.round((block.window_s ?? 300) / 60)} minutes. Your own session is fine.`,
      vague: true
    };
  }
  const suffix = "Your own session is fine — this is another client.";
  return { text: `${block.text ?? `${worst.client} is being refused repeatedly.`} ${suffix}`, vague: false };
}
function serverPredatesBuildStamp(health) {
  if (!health || typeof health !== "object" || Array.isArray(health)) return false;
  const h = health;
  if (typeof h.status !== "string" && typeof h.t !== "number") return false;
  return !("build" in h);
}
const FIELD_GAPS = [
  { field: "origin", says: "which robots this dashboard started itself" }
];
function fleetFieldGaps(peers) {
  const list = Array.isArray(peers) ? peers : peers && typeof peers === "object" ? Object.values(peers) : [];
  const rows = list.filter((p) => !!p && typeof p === "object");
  if (rows.length === 0) return [];
  return FIELD_GAPS.filter((g) => rows.every((r) => !(g.field in r))).map((g) => g.says);
}
function staleServerNotice(health, gaps) {
  if (!serverPredatesBuildStamp(health)) return NOTHING;
  if (gaps.length === 0) return NOTHING;
  const list = gaps.length === 1 ? gaps[0] : `${gaps.slice(0, -1).join(", ")} and ${gaps[gaps.length - 1]}`;
  return {
    text: `This page can show ${list}, but the server answering it is older than the code you are looking at, so it never sends them. Restart the dashboard from a terminal to pick them up — a restart is also the only way macOS will grant it camera access.`,
    vague: false
  };
}
const SCOPE_NOTE = "This is the link between this page and the dashboard. It says nothing about the robots or the cameras — each tile reports its own state.";
function connBadge(conn, opts = {}) {
  const meshDown = !!opts.meshDown;
  switch (conn) {
    case "open":
      return meshDown ? {
        // The socket IS open, so OFFLINE would be a lie. Narrow the claim.
        label: "LIVE · page only",
        tone: "warn",
        title: `This page is connected, but the dashboard's own mesh session is closed, so robot telemetry and commands are not flowing. ${SCOPE_NOTE}`,
        aria: "Dashboard link live, but the robot mesh session is down"
      } : {
        label: "LIVE",
        tone: "",
        title: `Connected to the dashboard. ${SCOPE_NOTE}`,
        aria: "Dashboard link: live"
      };
    case "connecting":
      return {
        label: "CONNECTING",
        tone: "warn",
        title: `Opening the connection to the dashboard. ${SCOPE_NOTE}`,
        aria: "Dashboard link: connecting"
      };
    case "closed":
      return {
        label: "OFFLINE",
        tone: "bad",
        title: `No connection to the dashboard: nothing on this page is updating, and buttons will not reach the robots. ${SCOPE_NOTE}`,
        aria: "Dashboard link: offline — nothing on this page is updating"
      };
    case "unauthorized":
      return {
        label: "NO ACCESS",
        tone: "bad",
        title: "The server rejected this token — set it in Settings. Nothing on this page is updating.",
        aria: "Dashboard link: not authorised — the server rejected this token"
      };
    default: {
      const unknown = conn;
      return {
        label: String(unknown).toUpperCase(),
        tone: "warn",
        title: `Unrecognised connection state "${unknown}". ${SCOPE_NOTE}`,
        aria: `Dashboard link: ${unknown}`
      };
    }
  }
}
function recordNavFlag(mock, base = "Record teleop episodes into a dataset") {
  if (mock !== true) {
    return { flagged: false, suffix: "", cls: "", title: base, aria: "record" };
  }
  return {
    flagged: true,
    suffix: " · rehearsal",
    cls: "rehearsal",
    title: `${base}. REHEARSAL: this backend has no /api/record, so the buttons work but nothing is written to disk and no dataset is produced.`,
    aria: "record — rehearsal only, nothing is written to disk"
  };
}
const SIGNALS = {
  9: {
    phrase: "killed (SIGKILL) — nothing here sends that: despawn asks with SIGTERM first, so the OS under memory pressure, a script, or a person ended it",
    unexplained: true
  },
  15: { phrase: "asked to stop (SIGTERM) — a despawn, a shutdown, or a script", unexplained: false },
  6: { phrase: "crashed (abort) — its own output names where", unexplained: false },
  11: { phrase: "crashed (segfault) — its own output names where", unexplained: false },
  2: { phrase: "interrupted (SIGINT) — a Ctrl-C in whatever started it", unexplained: false },
  1: { phrase: "hung up (SIGHUP) — the terminal that owned it went away", unexplained: false }
};
function deathVerdict(returncode) {
  if (returncode === null || returncode === void 0) {
    return { phrase: "gone, with no exit status recorded — its log is the only witness", unexplained: true };
  }
  if (returncode === 0) return { phrase: "exited cleanly (code 0) — it finished, or something asked it to", unexplained: false };
  if (returncode > 0) return { phrase: `exited with code ${returncode} — a failure inside the robot; its own output names it`, unexplained: false };
  const sig = -returncode;
  return SIGNALS[sig] ?? { phrase: `killed by signal ${sig}`, unexplained: true };
}
function retainedOutputIsStartup(input) {
  const lines = input.lines ?? [];
  const startedAt = input.startedAt;
  if (!lines.length || startedAt === null || startedAt === void 0 || !Number.isFinite(startedAt)) return null;
  const windowS = input.windowS ?? 120;
  const d = new Date(startedAt * 1e3);
  const startPastHour = d.getMinutes() * 60 + d.getSeconds();
  let clocked = 0;
  for (const line of lines) {
    const m = /^(\d{2}):(\d{2}):(\d{2})\b/.exec(line ?? "");
    if (!m) continue;
    clocked++;
    const pastHour = Number(m[2]) * 60 + Number(m[3]);
    let diff = Math.abs(pastHour - startPastHour);
    if (diff > 1800) diff = 3600 - diff;
    if (diff > windowS) return false;
  }
  return clocked === 0 ? null : true;
}
function shortCause(returncode) {
  return deathVerdict(returncode).phrase.split(" — ")[0];
}
function absentNotice(children) {
  if (!Array.isArray(children) || children.length === 0) return null;
  const named = children.filter((c) => c && typeof c.peer_id === "string" && c.peer_id.length > 0);
  const surprises = named.filter((c) => c.returncode !== 0);
  if (surprises.length === 0) return null;
  const detail = surprises.map((c) => `${c.peer_id} — ${deathVerdict(c.returncode).phrase}`).join("\n");
  const headline = surprises.length === 1 ? `${surprises[0].peer_id} is gone — ${shortCause(surprises[0].returncode)}` : `${surprises.length} robots you started are gone`;
  return { headline, detail, count: surprises.length };
}
function quietNotice(ids, dead = []) {
  if (!Array.isArray(ids) || ids.length === 0) return null;
  const buried = new Set(
    (Array.isArray(dead) ? dead : []).map((c) => c && c.peer_id).filter(Boolean)
  );
  const quiet = ids.filter((id) => typeof id === "string" && id.length > 0 && !buried.has(id));
  if (quiet.length === 0) return null;
  const detail = quiet.map((id) => `${id} — the process is running, but it has never joined the fleet`).join("\n");
  const headline = quiet.length === 1 ? `${quiet[0]} started but never joined the fleet` : `${quiet.length} robots you started never joined the fleet`;
  return { headline, detail, count: quiet.length };
}
function StrandsMark({ size = 22, title }) {
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "svg",
    {
      className: "mark",
      width: size * 290 / 463,
      height: size,
      viewBox: "0 0 290 463",
      fill: "none",
      xmlns: "http://www.w3.org/2000/svg",
      role: title ? "img" : void 0,
      "aria-label": title,
      "aria-hidden": title ? void 0 : true,
      focusable: "false",
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "path",
          {
            className: "mark-back",
            d: "M97.2902 52.7884C85.0674 49.1667 72.2234 56.1389 68.6017 68.3616C64.9801 80.5843 71.9524 93.4283 84.1749 97.0501L235.117 139.775C245.223 142.769 246.357 156.628 236.874 161.226L32.546 260.291C-14.9439 283.316 -9.16107 352.74 41.4835 367.591L189.551 411.009L190.125 411.169C202.183 414.376 214.665 407.396 218.196 395.355C221.784 383.122 214.774 370.296 202.541 366.709L54.4738 323.291C44.3447 320.321 43.1879 306.436 52.6857 301.831L257.014 202.766C304.432 179.776 298.758 110.483 248.233 95.512L97.2902 52.7884Z"
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "path",
          {
            className: "mark-front",
            d: "M259.147 0.981812C271.389 -2.57498 284.197 4.46571 287.754 16.7074C291.311 28.9492 284.27 41.757 272.028 45.3138L71.1727 103.671C40.7142 112.521 37.1976 154.262 65.7459 168.083L241.343 253.093C307.872 285.302 299.794 382.546 228.862 403.336L30.4041 461.502C18.1707 465.088 5.34708 458.078 1.76153 445.844C-1.8239 433.611 5.18637 420.787 17.4197 417.202L215.878 359.035C246.277 350.125 249.739 308.449 221.226 294.645L45.6297 209.635C-20.9834 177.386 -12.7772 79.9893 58.2928 59.3402L259.147 0.981812Z"
          }
        )
      ]
    }
  );
}
function FleetBar({
  conn,
  peerCount,
  dashboardId,
  safetyFlash,
  mesh,
  online,
  installable,
  activityCount,
  recordMock,
  absentChildren,
  quietChildren,
  onInstall,
  onSettings,
  onWireSecurity,
  onActivity,
  onDevices,
  onTraining,
  onRecord,
  onSim,
  onHelp
}) {
  const absentDeath = absentNotice(absentChildren);
  const quiet = quietNotice(quietChildren, absentChildren);
  const meshDown = mesh.online === false;
  const badge = connBadge(conn, { meshDown });
  const rec = recordNavFlag(recordMock);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("header", { className: "fleetbar", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "brand", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "logo", children: /* @__PURE__ */ jsxRuntimeExports.jsx(StrandsMark, { size: 26, title: "Strands Agents" }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h1", { children: "strands robots" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sub", title: `API: ${backendLabel()}`, children: [
          dashboardId || "fleet cockpit",
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "backend", children: [
            " · ",
            backendLabel()
          ] })
        ] })
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "fleet-right", children: [
      safetyFlash && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `safety ${safetyFlash}`, children: safetyFlash === "estop" ? "🛑 E-STOP" : "✅ RESUMED" }),
      !online && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge warn", title: "this device has no network", children: "offline" }),
      mesh.local_dev && /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "badge warnchip",
          onClick: onWireSecurity,
          title: "Robot mesh traffic is not encrypted. Fine on a trusted LAN - click for details and how to enable wire security.",
          children: "mesh unencrypted · local only"
        }
      ),
      meshDown && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge danger", title: "the dashboard's own mesh session is closed", children: "mesh down" }),
      installable && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "chip", onClick: onInstall, title: "Install as an app", children: "⤓ install" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "chip", onClick: onDevices, title: "Local hardware and managed robots", children: "⚙ devices" }),
      quiet && /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: "chip warn",
          onClick: onDevices,
          title: `${quiet.detail}

Open devices for its log — the refusal that kept it out of the fleet is in there (a missing calibration, a busy servo bus), and despawn is there too.`,
          children: [
            "🫥 ",
            quiet.headline
          ]
        }
      ),
      absentDeath && /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: "chip warn",
          onClick: onDevices,
          title: `${absentDeath.detail}

Open devices for the exit status and the last output.`,
          children: [
            "⚰ ",
            absentDeath.headline
          ]
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: `chip${rec.cls ? ` ${rec.cls}` : ""}`,
          onClick: onRecord,
          title: rec.title,
          "aria-label": rec.aria,
          children: [
            "⏺ record",
            rec.suffix
          ]
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "chip", onClick: onTraining, title: "Train policies on recorded datasets", children: "🎓 train" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "chip", onClick: onSim, title: "Simulated robots in this process: twin, camera, joint targets", children: "🧊 sim" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "chip", onClick: onActivity, title: "Command history", children: [
        "☰ activity",
        activityCount > 0 ? ` (${activityCount})` : ""
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "chip", onClick: onSettings, title: "Settings", children: "⚒ settings" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "chip",
          onClick: onHelp,
          title: "What this page is, how to stop a robot, and where the docs are",
          "aria-keyshortcuts": "?",
          children: "? help"
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "peers", children: [
        peerCount,
        " peer",
        peerCount === 1 ? "" : "s"
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "span",
        {
          className: `conn ${conn}${badge.tone ? ` ${badge.tone}` : ""}`,
          title: badge.title,
          "aria-label": badge.aria,
          children: badge.label
        }
      )
    ] })
  ] });
}
function emptySession() {
  return {
    dataset: null,
    task: "",
    leader: null,
    follower: null,
    target_episodes: 10,
    episodes: [],
    phase: "idle",
    fps: 30
  };
}
function makeMock() {
  let s = emptySession();
  let startedAt = 0;
  const clone = () => JSON.parse(JSON.stringify(s));
  return {
    mock: true,
    async session() {
      if (s.phase === "recording" && s.episodes.length > 0) {
        const ep = s.episodes[s.episodes.length - 1];
        ep.frames = Math.max(0, Math.round((Date.now() - startedAt) / 1e3 * s.fps));
        ep.duration_s = Math.round((Date.now() - startedAt) / 1e3 * 10) / 10;
      }
      return clone();
    },
    async open(opts) {
      s = { ...emptySession(), ...opts, fps: opts.fps ?? emptySession().fps, episodes: [], phase: "idle" };
      return clone();
    },
    async startEpisode() {
      if (!s.dataset) throw new Error("no open session");
      if (s.phase === "recording") return clone();
      s.phase = "recording";
      startedAt = Date.now();
      s.episodes.push({ index: s.episodes.length, frames: 0, duration_s: 0, thumbnails: {} });
      return clone();
    },
    async stopEpisode() {
      if (s.phase !== "recording") return clone();
      const duration = (Date.now() - startedAt) / 1e3;
      const ep = s.episodes[s.episodes.length - 1];
      ep.frames = Math.max(1, Math.round(duration * s.fps));
      ep.duration_s = Math.round(duration * 10) / 10;
      s.phase = "idle";
      return clone();
    },
    async redoEpisode() {
      if (s.phase === "recording") s.episodes.pop();
      s.phase = "idle";
      return clone();
    },
    async discard(index) {
      if (!s.dataset) throw new Error("no open session");
      const ep = s.episodes.find((e) => e.index === index);
      if (!ep) throw new Error(`no saved episode with index ${index}`);
      ep.discarded = true;
      return clone();
    },
    async close() {
      s = emptySession();
      return { ok: true, detail: "mock session closed (nothing was written)" };
    },
    async uploadPreflight() {
      return {
        ok: false,
        state: "no_credential",
        needs_force: false,
        user: null,
        destination: s.dataset,
        detail: "this is the in-browser rehearsal - nothing is written and nothing can be published"
      };
    }
  };
}
function makeReal() {
  return {
    mock: false,
    session: () => api("/api/record/session"),
    open: (opts) => post("/api/record/open", opts),
    startEpisode: () => post("/api/record/episode/start"),
    stopEpisode: () => post("/api/record/episode/stop"),
    redoEpisode: () => post("/api/record/episode/redo"),
    discard: (index) => post("/api/record/episode/discard", { index }),
    close: (opts) => post("/api/record/close", opts ?? {}),
    uploadPreflight: () => api("/api/record/upload-preflight")
  };
}
let cached = null;
function getRecordApi() {
  if (!cached) {
    cached = api("/api/record/session").then(() => makeReal()).catch((e) => e && e.status === 404 ? makeMock() : makeReal());
  }
  return cached;
}
const RUNNING = /* @__PURE__ */ new Set(["running", "executing"]);
function reportedTaskStatus(peer) {
  var _a, _b, _c;
  const s = ((_b = (_a = peer.state) == null ? void 0 : _a.task) == null ? void 0 : _b.status) ?? ((_c = peer.presence) == null ? void 0 : _c.task_status);
  return typeof s === "string" ? s : void 0;
}
function isRunningStatus(status) {
  return status !== void 0 && RUNNING.has(status);
}
function nextPhase(phase, reported) {
  if (isRunningStatus(reported)) return phase === "running" ? null : "running";
  if (reported !== void 0 && phase === "running") return "done";
  return null;
}
function deriveTaskFlags(input) {
  const { phase, reported, twinBusy } = input;
  return {
    running: isRunningStatus(reported) || phase === "starting" || phase === "running",
    busy: phase === "starting" || phase === "stopping" || twinBusy
  };
}
function describe(value, cap = 300) {
  if (value === void 0) return "no result payload";
  try {
    return String(JSON.stringify(value) ?? String(value)).slice(0, cap);
  } catch {
    return String(value).slice(0, cap);
  }
}
function errorInResult(result) {
  var _a;
  const err = (result == null ? void 0 : result.error) ?? ((_a = result == null ? void 0 : result.result) == null ? void 0 : _a.error);
  if (err === void 0 || err === null || err === "") return void 0;
  return String(err);
}
function interpretRun(res) {
  const err = errorInResult(res == null ? void 0 : res.result);
  if ((res == null ? void 0 : res.ok) && !err) {
    const via = res.routed_to ? ` via ${res.routed_to}` : "";
    const twin = res.mirrored_to_twin ? " + twin" : "";
    return { outcome: { ok: true, text: `running${via}${twin}` }, phase: "running" };
  }
  return {
    outcome: {
      ok: false,
      // The peer's own words beat "refused" — and when the envelope said ok while the payload
      // carried an error, that error IS the news.
      text: err ?? "refused",
      detail: describe(res == null ? void 0 : res.result)
    },
    phase: "failed"
  };
}
function interpretStop(res) {
  const state = res == null ? void 0 : res.state;
  if (state === "stopped") return { outcome: { ok: true, text: "stopped" }, phase: "idle" };
  if (state === "no_answer") {
    return {
      outcome: { ok: false, text: "no answer — robot may still be moving", ambiguous: true },
      phase: "failed"
    };
  }
  const detail = typeof (res == null ? void 0 : res.detail) === "string" ? res.detail : describe((res == null ? void 0 : res.detail) ?? {});
  return { outcome: { ok: false, text: `not stopped: ${detail}` }, phase: "failed" };
}
function looksLikeNeed(value) {
  return !!value && typeof value === "object" && typeof value.kind === "string" && typeof value.scope === "string" && typeof value.title === "string" && typeof value.risk === "string";
}
function findConsent(payload, depth = 3) {
  if (!payload || typeof payload !== "object" || depth <= 0) return null;
  if (looksLikeNeed(payload.needs_consent)) return payload.needs_consent;
  for (const key of ["result", "detail", "body", "error"]) {
    const nested = payload[key];
    if (nested && typeof nested === "object") {
      const found = findConsent(nested, depth - 1);
      if (found) return found;
    }
  }
  return null;
}
function canApprove(need) {
  var _a;
  if (typeof need.grantable === "boolean") return need.grantable;
  if (need.kind === "hf_repo_allow" || need.kind === "policy_type_allow" || need.kind === "policy_host_allow") return !!need.subject;
  return !!need.env_var || !!((_a = need.grants) == null ? void 0 : _a.length);
}
function blockedReason(need) {
  if (need.kind === "hf_repo_allow") {
    return "The repository name in this refusal could not be read safely, so there is nothing to allow. Check the model path and try again.";
  }
  if (need.kind === "policy_type_allow") {
    return "The policy name in this refusal could not be read safely, so there is nothing to allow. Check the policy type or provider you asked for and try again.";
  }
  if (need.kind === "policy_host_allow") {
    return "The address in this refusal could not be read as a host, so there is nothing to allow. Use a plain hostname or IP (optionally with a port) and try again.";
  }
  return "This refusal did not say what approving would change, so there is nothing to grant from here. It may come from a newer guard than this page — reload, and if it persists, grant it in the environment instead.";
}
const OPEN_ENDED = /* @__PURE__ */ new Set(["trust_remote_code", "agent_physical_motion"]);
const BOUNDED = /* @__PURE__ */ new Set(["hf_repo_allow", "teleop_degree_units", "policy_type_allow"]);
function severity(need) {
  if (OPEN_ENDED.has(need.kind)) return "danger";
  return BOUNDED.has(need.kind) ? "warn" : "danger";
}
function approveConsent(need) {
  return post("/api/consent", { kind: need.kind, subject: need.subject ?? null });
}
function afterApproval(result, target) {
  const granted = result.granted || result.already_granted;
  if (!granted) {
    return { retryNow: false, note: result.note || "nothing was granted — retrying would fail the same way" };
  }
  if (target === "spawn") {
    return { retryNow: true, note: result.note || "granted — starting again" };
  }
  return {
    retryNow: false,
    note: result.note || "granted — but this robot is already running with the old permissions. Respawn it, then run again."
  };
}
const NOT_A_GRANT = /* @__PURE__ */ new Set(["locks", "kinds", "env_file"]);
function nothingGranted(state) {
  if (!state) return false;
  for (const [key, value] of Object.entries(state)) {
    if (NOT_A_GRANT.has(key)) continue;
    if (Array.isArray(value)) {
      if (value.length > 0) return false;
    } else if (value && typeof value === "object") {
      if (value.granted) return false;
    } else if (value === true) {
      return false;
    }
  }
  return true;
}
const HARDWARE$1 = "The arms’ power switch is the only brake that does not go through this page — use it now.";
function reason(f) {
  const m = String(f.message ?? "").trim();
  return m || "no detail";
}
function refusedBeforeActing(status) {
  const s = Number(status ?? 0);
  if (!Number.isFinite(s) || s <= 0) return false;
  if (s >= 500) return false;
  return s === 400 || s === 401 || s === 403 || s === 404 || s === 405 || s === 422 || s === 429;
}
function estopFailureVerdict(f) {
  if (refusedBeforeActing(f.status)) {
    return {
      delivered: "no",
      headline: `✗ the dashboard refused to send the stop (${f.status}: ${reason(f)}) — nothing was sent, no robot was told to stop.`,
      advice: `Fix that first (a token in Settings usually), and do not wait for it: ${HARDWARE$1}`,
      retryRepeats: false
    };
  }
  const transport = !Number(f.status ?? 0);
  return {
    delivered: "unknown",
    headline: transport ? `⚠ no answer came back (${reason(f)}) — the stop MAY already have reached the fleet, and it may not. This page cannot tell.` : `⚠ the server failed mid-stop (${f.status}: ${reason(f)}) — some peers and the fleet lockout MAY already have been signalled.`,
    advice: `Assume the robots are still moving: ${HARDWARE$1} If the lockout did engage, every peer will refuse commands until you resume with the override code — that is the stop working, not a new fault.`,
    retryRepeats: true
  };
}
function resumeFailureVerdict(f) {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `✗ rejected (${f.status}: ${reason(f)}) — the lockout is still in place (wrong code? brute-force cooldown?).`,
      delivered: "no"
    };
  }
  return {
    text: `⚠ no answer (${reason(f)}) — the lockout may or may not have cleared. Check whether a robot accepts a command before resuming again.`,
    delivered: "unknown"
  };
}
const HARDWARE = "Use STOP ALL (press .) or the arms’ power switch.";
function why(f) {
  return String(f.message ?? "").trim() || "no detail";
}
function detailOf$1(f) {
  const s = Number(f.status ?? 0);
  return s ? `HTTP ${s}` : "no answer — delivery unknown";
}
function runFailure(f) {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `refused (${f.status}): ${why(f)} — nothing was sent to the arm, the policy is NOT running.`,
      detail: detailOf$1(f),
      ambiguous: false
    };
  }
  return {
    text: `${why(f)} — the policy MAY have started: this arm can be moving. Keep hands clear and watch this card: if it starts saying “running”, it did start. Do not press ▶ again until you know — that would dispatch a second task.`,
    detail: detailOf$1(f),
    ambiguous: true
  };
}
function stopFailure(f) {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `stop refused (${f.status}): ${why(f)} — it never reached the robot, which is still doing whatever it was doing. ${HARDWARE}`,
      detail: detailOf$1(f),
      ambiguous: false
    };
  }
  return {
    text: `${why(f)} — the stop may NOT have been delivered. Assume the arm is still moving. ${HARDWARE}`,
    detail: detailOf$1(f),
    ambiguous: true
  };
}
function useTask(peer) {
  const [phase, setPhase] = reactExports.useState("idle");
  const [outcome, setOutcome] = reactExports.useState(null);
  const [twinBusy, setTwinBusy] = reactExports.useState(false);
  const [consent, setConsent] = reactExports.useState(null);
  const lastBody = reactExports.useRef(null);
  const mounted = reactExports.useRef(true);
  reactExports.useEffect(() => () => {
    mounted.current = false;
  }, []);
  const reported = reportedTaskStatus(peer);
  const { running, busy } = deriveTaskFlags({ phase, reported, twinBusy });
  reactExports.useEffect(() => {
    const next = nextPhase(phase, reported);
    if (next) setPhase(next);
  }, [reported]);
  const fail = (e) => {
    if (e instanceof HttpError) {
      return { ok: false, text: e.message, detail: e.status ? `HTTP ${e.status}` : "unreachable" };
    }
    return { ok: false, text: e instanceof Error ? e.message : String(e) };
  };
  const physicalFail = (e, kind) => {
    const f = { status: e instanceof HttpError ? e.status : 0, message: e instanceof Error ? e.message : String(e) };
    const v = kind === "run" ? runFailure(f) : stopFailure(f);
    return { ok: false, text: v.text, detail: v.detail, ambiguous: v.ambiguous };
  };
  const run = async (body) => {
    setPhase("starting");
    setOutcome(null);
    setConsent(null);
    lastBody.current = body;
    try {
      const res = await post(
        `/api/robots/${encodeURIComponent(peer.peer_id)}/task`,
        /**
         * The confirmation marker: this call is only reachable through the run form, whose ▶ passes
         * the RunConfirm dialog first, so the browser can honestly say a human confirmed.
         */
        { ...body, confirmed: true }
      );
      if (!mounted.current) return;
      const v = interpretRun(res);
      setOutcome(v.outcome);
      setPhase(v.phase);
      if (!v.outcome.ok) setConsent(findConsent(res));
    } catch (e) {
      if (!mounted.current) return;
      setOutcome(physicalFail(e, "run"));
      setPhase("failed");
      if (e instanceof HttpError) setConsent(findConsent(e.body));
    }
  };
  const retryLast = async () => {
    setConsent(null);
    if (lastBody.current) await run(lastBody.current);
  };
  const stop = async () => {
    setPhase("stopping");
    setOutcome(null);
    try {
      const res = await post(`/api/robots/${encodeURIComponent(peer.peer_id)}/stop`);
      if (!mounted.current) return;
      const v = interpretStop(res);
      setOutcome(v.outcome);
      setPhase(v.phase);
    } catch (e) {
      if (!mounted.current) return;
      setOutcome(physicalFail(e, "stop"));
      setPhase("failed");
    }
  };
  const toggleTwin = async () => {
    setTwinBusy(true);
    try {
      await post(`/api/robots/${encodeURIComponent(peer.peer_id)}/twin`, {});
    } catch (e) {
      setOutcome(fail(e));
    } finally {
      if (mounted.current) setTwinBusy(false);
    }
  };
  return {
    phase,
    outcome,
    running,
    busy,
    twinBusy,
    run,
    stop,
    toggleTwin,
    setOutcome,
    consent,
    clearConsent: () => setConsent(null),
    retryLast
  };
}
const BUS_RECOVERY_WARN_AT = 5;
function busRecoveryBadge(count) {
  const n = typeof count === "number" && Number.isFinite(count) ? Math.floor(count) : 0;
  if (n <= 0) return null;
  const tone = n >= BUS_RECOVERY_WARN_AT ? "warn" : "";
  const plural2 = n === 1 ? "once" : `${n} times`;
  const why2 = `This arm's serial bus was left marked in-use by an exchange that never finished, and the dashboard cleared it and read again - ${plural2} since this robot started.

Nothing is wrong with the reading you are looking at: the joints below are real. But a bus strands for physical reasons - a marginal USB cable, a hub browning out under load, a connector working loose as the arm moves.

`;
  const verdict2 = n >= BUS_RECOVERY_WARN_AT ? "This has now happened often enough to be a pattern rather than bad luck: swap the cable, try a powered hub or a different port, and prefer a direct connection over a chain of hubs. Recording a dataset through a bus this flaky risks episodes with gaps in them." : "Once or twice is a hiccup and needs nothing from you. Worth remembering if it keeps climbing.";
  return { label: n === 1 ? "bus healed once" : `bus healed ×${n}`, tone, title: why2 + verdict2 };
}
const TELEMETRY_CAP = 120;
const emptyRing = () => ({ samples: [], prev: [], jointsSeen: null, lastT: void 0 });
function jointValues(peer) {
  var _a;
  const joints = (_a = peer.state) == null ? void 0 : _a.joints;
  if (!joints) return [];
  return Object.values(joints).map((v) => {
    if (typeof v === "number") return v;
    if (Array.isArray(v)) return v[0] ?? 0;
    return v.position ?? 0;
  });
}
function motionBetween(prev, values) {
  if (prev.length !== values.length || values.length === 0) return 0;
  let motion = 0;
  for (let i = 0; i < values.length; i++) motion += Math.abs(values[i] - prev[i]);
  return motion / values.length;
}
function advance(acc, peer, nowS) {
  var _a;
  const stateT = (_a = peer.state) == null ? void 0 : _a.t;
  if (stateT === void 0 || stateT === acc.lastT) return acc;
  const values = jointValues(peer);
  return {
    samples: [...acc.samples, { t: nowS, motion: motionBetween(acc.prev, values) }].slice(-TELEMETRY_CAP),
    prev: values,
    // Once joints have been seen they stay seen: an arm that drops a frame has not stopped being
    // an arm, and `jointsSeen: false` is authoritative enough downstream to suppress the whole
    // motion sentence.
    jointsSeen: (acc.jointsSeen ?? false) || values.length > 0,
    lastT: stateT
  };
}
const TELEMETRY_GAP_S = 5;
function recentRun(samples, maxGapS = TELEMETRY_GAP_S) {
  for (let i = samples.length - 1; i > 0; i--) {
    if (samples[i].t - samples[i - 1].t > maxGapS) return samples.slice(i);
  }
  return samples;
}
function summarize(acc, nowS) {
  const { jointsSeen } = acc;
  const samples = recentRun(acc.samples);
  const newest = acc.samples[acc.samples.length - 1];
  if (samples.length < 2) {
    return { samples, hz: 0, moving: null, stateAgeS: newest ? nowS - newest.t : null, jointsSeen };
  }
  const span = samples[samples.length - 1].t - samples[0].t;
  const hz = span > 0 ? (samples.length - 1) / span : 0;
  const peak = Math.max(...samples.map((s) => s.motion), 1e-6);
  const moving = jointsSeen && samples.length >= 10 ? samples.slice(-10).some((s) => s.motion > peak * 0.05) : null;
  return { samples, hz, moving, stateAgeS: nowS - samples[samples.length - 1].t, jointsSeen };
}
function useTelemetry(peer) {
  var _a;
  const acc = reactExports.useRef(emptyRing());
  const [, tick] = reactExports.useState(0);
  const stateT = (_a = peer.state) == null ? void 0 : _a.t;
  reactExports.useEffect(() => {
    const next = advance(acc.current, peer, Date.now() / 1e3);
    if (next === acc.current) return;
    acc.current = next;
    tick((n) => n + 1);
  }, [stateT]);
  return summarize(acc.current, Date.now() / 1e3);
}
const quote = (s, max = 44) => `“${s.length > max ? s.slice(0, max - 1) + "…" : s}”`;
function statusSentence(f) {
  if (f.stale) {
    const ago2 = f.lastSeenAgoS != null ? ` for ${Math.round(f.lastSeenAgoS)}s` : "";
    return {
      severity: "danger",
      word: "offline",
      text: `no heartbeat${ago2} — state unknown, treat the arm as unpredictable`
    };
  }
  if (f.stateAgeS != null && f.stateAgeS > 5) {
    return {
      severity: "warn",
      word: "frozen",
      text: `peer is alive but its state stream stopped ${Math.round(f.stateAgeS)}s ago — joints shown are stale`
    };
  }
  if (f.hwConnected === false) {
    return {
      severity: "warn",
      word: "no hw",
      text: "hardware not connected — the arm is unplugged or unpowered, nothing can move"
    };
  }
  if ((f.lockout ?? "").trim().toLowerCase() === "locked") {
    if (f.moving === true) {
      return {
        severity: "danger",
        word: "locked?!",
        text: "an e-stop lockout is in place but the joints are MOVING — the lockout is not holding, or something outside the mesh is driving the arm; keep hands clear"
      };
    }
    return {
      severity: "warn",
      word: "locked",
      text: "e-stop lockout — commands are refused and the arm is holding where the stop caught it; clearing the lockout is what makes it live again"
    };
  }
  const status = (f.taskStatus ?? "").trim().toLowerCase();
  const running = status === "running";
  if (running) {
    const what = f.instruction ? ` ${quote(f.instruction)}` : "";
    const since = f.taskDurationS != null && f.taskDurationS >= 1 ? `, ${Math.round(f.taskDurationS)}s in` : "";
    if (f.moving === false) {
      return {
        severity: "warn",
        word: "wedged?",
        text: `policy${what} says running but the arm is not moving${since} — wedged, or producing no-ops`
      };
    }
    return {
      severity: "active",
      word: "running",
      text: `running${what}${since} — arm is under policy control, keep hands clear`
    };
  }
  if (status === "connecting") {
    return f.moving === true ? {
      severity: "warn",
      word: "starting",
      text: "bringing the hardware up and the arm is ALREADY MOVING — homing or a queued command, keep hands clear"
    } : {
      severity: "active",
      word: "starting",
      text: "bringing the hardware up — torque can engage and the arm move without warning, keep hands clear"
    };
  }
  if (f.moving === true) {
    return {
      severity: "warn",
      word: "moving",
      text: "arm is MOVING with no task — teleop or another client is commanding it, keep hands clear"
    };
  }
  if (status === "error") {
    return {
      severity: "warn",
      word: "failed",
      text: "the task ended in ERROR — nothing is commanding the arm now, but it stopped wherever it got to; read the robot log before approaching"
    };
  }
  if (status === "stopped") {
    return {
      severity: "ok",
      word: "stopped",
      text: "the task was stopped before finishing — the arm is holding where it stopped, and a resume would move it from there"
    };
  }
  if (status !== "" && status !== "idle" && status !== "completed") {
    return {
      severity: "warn",
      word: "unknown",
      text: `the robot reports task status ${quote(status, 24)} — this dashboard does not know that state, so stillness is not confirmed here`
    };
  }
  if (f.moving == null || f.jointsSeen === false) {
    if (f.jointsSeen === false && f.hostsChildren && f.hostsChildren.length) {
      const kids = f.hostsChildren;
      return {
        severity: "ok",
        word: "process",
        text: kids.length === 1 ? `hosts ${kids[0]} — this is the process, not an arm; the joints are on that card` : `hosts ${kids.length} robots (${kids.join(", ")}) — this is the process, not an arm`
      };
    }
    if (f.jointsSeen === false) {
      return {
        severity: "warn",
        word: "idle?",
        text: "the robot reports idle, but it publishes no joint positions — stillness cannot be confirmed here, so treat the arm as able to move"
      };
    }
    return {
      severity: "ok",
      word: "idle",
      text: "idle per the robot — motion not measured yet, a second of telemetry decides"
    };
  }
  return {
    severity: "ok",
    word: "idle",
    text: "idle and still — safe to approach"
  };
}
function ribbonDetail(line) {
  const word = line.word.trim().toLowerCase();
  const text = line.text;
  if (!word || !text.toLowerCase().startsWith(word)) return text;
  const after = text.charAt(word.length);
  if (after && /[A-Za-z0-9]/.test(after)) return text;
  const rest = text.slice(word.length).replace(/^[\s\u2014:,-]+/, "");
  return rest.length > 0 ? rest : text;
}
function peerStatusFields(peer, telemetry, hostsChildren) {
  var _a, _b, _c, _d, _e, _f, _g;
  const p = peer.presence;
  return {
    stale: !!peer.stale,
    lastSeenAgoS: peer.last_seen ? Date.now() / 1e3 - peer.last_seen : null,
    hwConnected: (p == null ? void 0 : p.connected) ?? null,
    taskStatus: ((_b = (_a = peer.state) == null ? void 0 : _a.task) == null ? void 0 : _b.status) ?? (p == null ? void 0 : p.task_status) ?? null,
    instruction: ((_d = (_c = peer.state) == null ? void 0 : _c.task) == null ? void 0 : _d.instruction) || (p == null ? void 0 : p.instruction) || null,
    taskDurationS: ((_f = (_e = peer.state) == null ? void 0 : _e.task) == null ? void 0 : _f.duration) ?? null,
    moving: telemetry.moving ?? null,
    jointsSeen: telemetry.jointsSeen ?? null,
    stateAgeS: telemetry.stateAgeS ?? null,
    lockout: ((_g = peer.lockout) == null ? void 0 : _g.state) ?? null,
    hostsChildren: hostsChildren ?? null
  };
}
function twinButtonCopy(o) {
  const twinId = `${o.peerId}-twin`;
  if (o.busy) {
    return {
      label: "…",
      cls: o.twinLive ? "on" : "",
      pressed: !!o.twinLive,
      title: `waiting for ${twinId} — a sim peer takes a moment to start or stop`,
      aria: `sim twin of ${o.peerId}: working`
    };
  }
  if (o.twinLive) {
    return {
      label: "twin on",
      cls: "on",
      pressed: true,
      title: `${twinId} is running: tasks sent to this robot are mirrored to it. Click to stop the twin — the real arm is not affected either way.`,
      aria: `stop the sim twin of ${o.peerId}`
    };
  }
  return {
    label: "+ twin",
    cls: "",
    pressed: false,
    title: `Start ${twinId}, a simulated copy of this arm as its own peer. Tasks you send this robot are mirrored to it, so you can watch a policy in sim before trusting it on metal. The real arm is not touched.`,
    aria: `start a sim twin of ${o.peerId}`
  };
}
const CAMERA_STOPPED_AGE_S = 120;
function captureAge(meta, nowS) {
  const t = meta == null ? void 0 : meta.t;
  if (t == null || typeof t !== "number" || !Number.isFinite(t) || t <= 0) return null;
  const age = nowS - t;
  if (age < 0) return null;
  return age;
}
function stoppedCameras(cameras2, nowS, maxAgeS = CAMERA_STOPPED_AGE_S) {
  const out = [];
  for (const [camera, meta] of Object.entries(cameras2 ?? {})) {
    const ageS2 = captureAge(meta, nowS);
    if (ageS2 !== null && ageS2 > maxAgeS) out.push({ camera, ageS: ageS2 });
  }
  return out.sort((a, b) => b.ageS - a.ageS);
}
function agoText(seconds) {
  if (seconds < 90) return `${Math.round(seconds)}s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
  return `${(seconds / 3600).toFixed(1)}h ago`;
}
function cameraWarning(stopped, opts = {}) {
  if (stopped.length === 0) return null;
  const which = stopped.map((c) => `${c.camera} (last frame ${agoText(c.ageS)})`).join(", ");
  const who = opts.peerId ? `${opts.peerId}: ` : "";
  const plural2 = stopped.length > 1 ? "cameras have" : "camera has";
  return `${who}${stopped.length} ${plural2} stopped publishing — ${which}. Recording now writes episodes with a frozen or missing image stream, which you would only notice at training time.`;
}
function deadCameraNote(stopped, totalCameras) {
  if (stopped.length === 0 || totalCameras === 0) return null;
  const which = stopped.map((c) => `${c.camera} ${agoText(c.ageS)}`).join(", ");
  const noun = totalCameras > 1 ? "cameras" : "camera";
  return `${stopped.length} of ${totalCameras} ${noun} stopped — ${which}`;
}
const STALL_MS = 2500;
const BUSY = /(in use|busy|already open|cannot open|could not open|-11852|EBUSY|Resource temporarily unavailable)/i;
const DENIED = /(unauthorized|not permitted|permission|denied|forbidden|TCC)/i;
const PUBLISH_FRESH_MS = 15e3;
const CAPTURE_STALE_MS = 1e4;
function ageText(ms) {
  if (!isFinite(ms) || ms < 0) return "unknown";
  if (ms < 1e3) return "<1s";
  if (ms < 9e4) return `${Math.round(ms / 1e3)}s`;
  if (ms < 54e5) return `${Math.round(ms / 6e4)}m`;
  if (ms < 1728e5) return `${(ms / 36e5).toFixed(1).replace(/\.0$/, "")}h`;
  return `${Math.round(ms / 864e5)}d`;
}
function publishedAtMs(publishedAt) {
  if (publishedAt === void 0 || publishedAt === null || !isFinite(publishedAt)) return void 0;
  if (publishedAt <= 0) return void 0;
  return publishedAt < 1e12 ? publishedAt * 1e3 : publishedAt;
}
function classifyCamera(input) {
  const { now, conn, frames, lastFrameAt, error, publishedAt } = input;
  const stallMs = input.stallMs ?? STALL_MS;
  const age = lastFrameAt === void 0 ? void 0 : now - lastFrameAt;
  const hadFrames = frames > 0 && age !== void 0;
  const secs = (ms) => ms < 1e3 ? "<1s" : `${Math.round(ms / 1e3)}s`;
  const pubMs = publishedAtMs(publishedAt);
  const pubAge = pubMs === void 0 ? void 0 : now - pubMs;
  if (error && DENIED.test(error)) {
    return { kind: "unauthorized", title: "not permitted", detail: "this session may not read camera frames", live: false, frozen: hadFrames };
  }
  if (error && BUSY.test(error)) {
    return { kind: "busy", title: "camera busy", detail: "another app is holding this device", live: false, frozen: hadFrames };
  }
  if (error) {
    return { kind: "error", title: "no image", detail: error, live: false, frozen: hadFrames };
  }
  const captureAge2 = pubAge !== void 0 && pubAge >= 0 ? pubAge : void 0;
  const captureStale = captureAge2 !== void 0 && captureAge2 > (input.captureStaleMs ?? CAPTURE_STALE_MS);
  if (hadFrames && (age > stallMs || captureStale)) {
    const detail = captureStale && age <= stallMs ? `the peer says it captured this ${ageText(captureAge2)} ago - it arrived here ${ageText(age)} ago as a replay of its last frame, not a new one` : captureStale ? `last frame ${ageText(age)} ago, and the peer says it captured it ${ageText(captureAge2)} ago` : `last frame ${ageText(age)} ago`;
    return {
      kind: "stalled",
      title: captureStale && age <= stallMs ? "stale frame" : "stalled",
      detail,
      live: false,
      frozen: true
    };
  }
  if (hadFrames) return { kind: "live", title: "live", detail: "", live: true, frozen: false };
  if (conn === "closed") {
    if (input.retryInMs !== void 0) {
      return {
        kind: "retrying",
        title: "reconnecting",
        detail: `in ${secs(input.retryInMs)}${input.attempt ? ` (attempt ${input.attempt})` : ""}`,
        live: false,
        frozen: false
      };
    }
    return { kind: "closed", title: "disconnected", detail: "stream closed", live: false, frozen: false };
  }
  if (conn === "open") {
    if (pubAge !== void 0 && pubAge > (input.publishFreshMs ?? PUBLISH_FRESH_MS)) {
      return {
        kind: "silent",
        title: "no frames",
        // Says WHERE it stopped: the stream is fine, the camera at the other end
        // is not, so the next step is that robot's log rather than this page.
        detail: `the peer's last frame is ${ageText(pubAge)} old - the camera stopped there, not in transit`,
        live: false,
        frozen: false
      };
    }
    return pubAge !== void 0 ? { kind: "waiting", title: "waiting", detail: `peer published ${ageText(pubAge)} ago, none arrived here yet`, live: false, frozen: false } : { kind: "silent", title: "no frames", detail: "stream open, camera sending nothing", live: false, frozen: false };
  }
  return { kind: "connecting", title: "connecting", detail: "opening the stream", live: false, frozen: false };
}
function pacedFps(message) {
  const m = /at\s+([0-9]+(?:\.[0-9]+)?)\s*fps/i.exec(message);
  if (!m) return null;
  const v = Number(m[1]);
  return Number.isFinite(v) && v > 0 ? v : null;
}
function pacingFromNotice(ev) {
  if (!ev || typeof ev !== "object") return null;
  const e = ev;
  if (e.type !== "camera_error" || e.throttled !== true) return null;
  const message = typeof e.error === "string" ? e.error : "";
  const fps = pacedFps(message);
  return {
    fps,
    note: fps === null ? "the server is pacing this stream while it settles" : `paced by the server at ${fps} fps — the link was being saturated`
  };
}
function nextRequestedFps(own, paced) {
  const caps = [own, paced].filter((v) => typeof v === "number" && v > 0);
  return caps.length ? Math.min(...caps) : null;
}
const DEGRADED_FPS = 1;
function CameraTile({ peerId, cam, big = false, meta, onConfigure }) {
  var _a;
  const imgRef = reactExports.useRef(null);
  const [status, setStatus] = reactExports.useState(() => classifyCamera({ now: Date.now(), conn: "connecting", frames: 0 }));
  const [fps, setFps] = reactExports.useState(0);
  const frames = reactExports.useRef(0);
  const lastFrameAt = reactExports.useRef(void 0);
  const times = reactExports.useRef([]);
  const conn = reactExports.useRef("connecting");
  const error = reactExports.useRef(null);
  const retryAt = reactExports.useRef(void 0);
  const openLog = reactExports.useRef([]);
  const degraded = reactExports.useRef(null);
  const pacedFps2 = reactExports.useRef(null);
  const [pacedNote, setPacedNote] = reactExports.useState(null);
  const tries = reactExports.useRef(0);
  const metaRef = reactExports.useRef(meta);
  metaRef.current = meta;
  reactExports.useEffect(() => {
    let ws = null;
    let url = null;
    let stopped = false;
    let retryTimer;
    let openedAt;
    let framesThisSocket = 0;
    const tick = window.setInterval(() => {
      var _a2;
      const now = Date.now();
      setStatus(classifyCamera({
        now,
        conn: conn.current,
        frames: frames.current,
        lastFrameAt: lastFrameAt.current,
        error: error.current,
        publishedAt: (_a2 = metaRef.current) == null ? void 0 : _a2.t,
        retryInMs: retryAt.current !== void 0 ? Math.max(0, retryAt.current - now) : void 0,
        attempt: retryAt.current !== void 0 ? tries.current : void 0
      }));
      const t = performance.now();
      times.current = times.current.filter((x) => t - x < 2e3);
      setFps(times.current.length > 1 ? (times.current.length - 1) / ((t - times.current[0]) / 1e3) : 0);
    }, 400);
    const open = () => {
      if (stopped) return;
      conn.current = "connecting";
      framesThisSocket = 0;
      openedAt = void 0;
      const capped = openLog.current.length >= CHURN_OPENS_PER_MIN;
      degraded.current = nextRequestedFps(capped ? DEGRADED_FPS : null, pacedFps2.current);
      const path = `/ws/camera/${encodeURIComponent(peerId)}/${encodeURIComponent(cam)}`;
      const rate = degraded.current;
      ws = new WebSocket(wsUrl(rate === null ? path : `${path}?max_fps=${rate}`));
      ws.binaryType = "blob";
      ws.onopen = () => {
        conn.current = "open";
        openedAt = Date.now();
        retryAt.current = void 0;
        openLog.current.push(openedAt);
        while (openLog.current.length && openedAt - openLog.current[0] > 6e4) openLog.current.shift();
      };
      ws.onmessage = (msg) => {
        if (typeof msg.data === "string") {
          try {
            const ev = JSON.parse(msg.data);
            const pacing = pacingFromNotice(ev);
            if (pacing) {
              pacedFps2.current = pacing.fps;
              setPacedNote(pacing.note);
            } else if (ev.type === "camera_error") error.current = ev.error;
          } catch {
          }
          return;
        }
        if (!(msg.data instanceof Blob)) return;
        const next = URL.createObjectURL(msg.data);
        if (imgRef.current) imgRef.current.src = next;
        if (url) URL.revokeObjectURL(url);
        url = next;
        error.current = null;
        frames.current += 1;
        framesThisSocket += 1;
        lastFrameAt.current = Date.now();
        times.current.push(performance.now());
      };
      ws.onclose = (ev) => {
        conn.current = "closed";
        const plan = planRetry({
          attempt: tries.current,
          frames: framesThisSocket,
          openMs: openedAt !== void 0 ? Date.now() - openedAt : void 0,
          code: ev.code,
          recentOpens: openLog.current.length,
          sessionExpired: sessionVerdict(authToken(), Date.now() / 1e3).refusesUntilSignIn,
          pageRefused: authRefusedRecently()
        });
        tries.current = plan.attempt;
        if (plan.delayMs === null) {
          error.current = plan.reason.includes("sign in") ? plan.reason : "unauthorized";
          retryAt.current = void 0;
          return;
        }
        if (stopped) return;
        retryAt.current = Date.now() + plan.delayMs;
        retryTimer = window.setTimeout(open, plan.delayMs);
      };
    };
    open();
    return () => {
      stopped = true;
      window.clearInterval(tick);
      if (retryTimer) window.clearTimeout(retryTimer);
      ws == null ? void 0 : ws.close();
      if (url) URL.revokeObjectURL(url);
    };
  }, [peerId, cam]);
  const shape = ((_a = meta == null ? void 0 : meta.shape) == null ? void 0 : _a.length) ? `${meta.shape[1]}×${meta.shape[0]}` : null;
  const cls = ["camtile", big ? "big" : "", `cam-${status.kind}`, status.frozen ? "frozen" : ""].filter(Boolean).join(" ");
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: cls, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("img", { ref: imgRef, alt: `${peerId} ${cam} camera` }),
    !status.live && // aria-live: a stream dying is news. A sighted user gets it from the
    // overlay appearing; a screen reader needs it announced.
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camstate", role: "status", "aria-live": "polite", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: status.title }),
      status.detail && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: status.detail }),
      status.frozen && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "camstale", children: "showing the last frame received" })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: status.live ? "camlabel live" : "camlabel", children: [
      cam,
      status.live && fps > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { children: [
        " ",
        fps.toFixed(0),
        "fps"
      ] }),
      shape && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { children: [
        " ",
        shape
      ] })
    ] }),
    onConfigure && /* @__PURE__ */ jsxRuntimeExports.jsx(
      "button",
      {
        className: "camcfg",
        onClick: onConfigure,
        "aria-label": `adjust ${cam} camera settings`,
        title: `adjust ${cam} — fps, size, device (applying restarts the robot)`,
        children: "⚙"
      }
    ),
    pacedNote && // Subdued, and NOT in the error overlay: the tile is alive. Announced politely
    // because a rate change explains a jerky picture a sighted user can see.
    /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "campaced", role: "status", "aria-live": "polite", children: pacedNote })
  ] });
}
function rowsFromConfig(config) {
  return Object.entries(config ?? {}).map(([name, c]) => ({
    name,
    indexOrPath: String(c.index_or_path ?? ""),
    fps: c.fps != null ? String(c.fps) : "",
    width: c.width != null ? String(c.width) : "",
    height: c.height != null ? String(c.height) : ""
  }));
}
function parseIndexOrPath(raw) {
  const t = raw.trim();
  if (!t) return null;
  if (/^\d+$/.test(t)) return Number(t);
  return t;
}
const BOUNDS = {
  fps: [1, 240],
  width: [16, 7680],
  height: [16, 4320]
};
function intField(row, field) {
  const raw = row[field].trim();
  if (!raw) return {};
  if (!/^\d+$/.test(raw)) return { error: `${row.name || "(unnamed)"}: ${field} must be a whole number` };
  const v = Number(raw);
  const [lo, hi] = BOUNDS[field];
  if (v < lo || v > hi) return { error: `${row.name || "(unnamed)"}: ${field}=${v} is outside ${lo}..${hi}` };
  return { value: v };
}
function configFromRows(rows) {
  const live = rows.filter((r) => r.name.trim() || r.indexOrPath.trim() || r.fps.trim() || r.width.trim() || r.height.trim());
  if (live.length === 0) return { cameras: null };
  const out = {};
  for (const row of live) {
    const name = row.name.trim();
    if (!name) return { cameras: null, error: "every camera needs a name (top / wrist / main…)" };
    if (out[name]) return { cameras: null, error: `two cameras are both named "${name}"` };
    const iop = parseIndexOrPath(row.indexOrPath);
    if (iop === null) return { cameras: null, error: `${name}: needs an index (0, 1…) or a device path` };
    const entry = { index_or_path: iop };
    for (const field of ["fps", "width", "height"]) {
      const { value, error } = intField(row, field);
      if (error) return { cameras: null, error };
      if (value != null) entry[field] = value;
    }
    if (entry.width == null !== (entry.height == null)) {
      return { cameras: null, error: `${name}: give both width and height, or neither (driver default)` };
    }
    out[name] = entry;
  }
  return { cameras: out };
}
function focusTarget(rows, focusCam, adding) {
  if (adding) return rows.length ? { index: rows.length - 1, field: "name" } : null;
  if (!focusCam) return null;
  const index = rows.findIndex((r) => r.name === focusCam);
  return index >= 0 ? { index, field: "fps" } : null;
}
function applySummary(rows, peerId) {
  const { cameras: cameras2 } = configFromRows(rows);
  const n = cameras2 ? Object.keys(cameras2).length : 0;
  const what = n === 0 ? "detach every camera from" : `apply ${n} camera${n === 1 ? "" : "s"} to`;
  return `This will ${what} ${peerId} by restarting it — its streams (and any running task) stop during the restart.`;
}
const DEFAULT_MESH_CAMERA_HZ = 5;
function previewRateNote(fps, cameraHz) {
  const publish = cameraHz == null || !Number.isFinite(cameraHz) || cameraHz <= 0 ? DEFAULT_MESH_CAMERA_HZ : cameraHz;
  const capture = fps == null || !Number.isFinite(fps) || fps <= 0 ? null : fps;
  if (capture == null || publish >= capture) return null;
  return `${capture} fps is the camera's capture rate — this dashboard receives ${publish}/s, the mesh publish rate (Settings › mesh › camera_hz). The recording on disk is not capped by it.`;
}
const PROBE = /state probe 'hw_joints' failed[^:]*:\s*(.*)$/;
const EXC = /^([A-Za-z_][A-Za-z0-9_]*)\(/;
const PORT = /['"](\/dev\/[^'"]+)['"]/;
function jointFailure(lines) {
  var _a, _b, _c;
  const list = (lines ?? []).filter((l) => typeof l === "string");
  let idx = -1;
  for (let i = list.length - 1; i >= 0; i--) if (PROBE.test(list[i])) {
    idx = i;
    break;
  }
  if (idx < 0) return null;
  const rest = (((_a = list[idx].match(PROBE)) == null ? void 0 : _a[1]) ?? "").trim();
  if (!rest) return null;
  const kind = ((_b = rest.match(EXC)) == null ? void 0 : _b[1]) ?? null;
  const port = ((_c = rest.match(PORT)) == null ? void 0 : _c[1]) ?? null;
  const tailMisleads = list.slice(idx + 1).some((l) => /hardware connected|\bonline\b/.test(l));
  const out = { kind, quote: rest, headline: "", tailMisleads: tailMisleads || void 0 };
  if (/Port is in use/i.test(rest) || /sync read 'Present_Position'/.test(rest)) {
    out.headline = `something else on this machine holds ${port ?? "the serial port"}, so this arm cannot read its own position`;
    out.remedy = "the bus has more than one owner — stop the other holder (a leftover robot process or a script), then respawn this arm; nothing needs unplugging";
    return out;
  }
  if (/has no calibration registered/i.test(rest)) {
    out.headline = "this arm was spawned with a robot id that has no calibration file, so every joint read is refused";
    out.remedy = "lerobot looks for calibration/robots/<type>/<robot_id>.json under ~/.cache/huggingface/lerobot — respawn it with an id that HAS a file, or calibrate it under this one. It is a name mismatch, not a hardware fault";
    return out;
  }
  out.headline = kind ? `this arm could not read its joints — ${kind}, which the dashboard has no advice for` : "this arm could not read its joints";
  return out;
}
function jointFailureBadge(f) {
  if (!f) return null;
  if (/holds .*serial port|holds \/dev/.test(f.headline)) return "the serial port is held by something else";
  if (/no calibration file/.test(f.headline)) return "its robot id has no calibration file";
  return f.kind ? `${f.kind} — open it for the arm's own words` : "open it for the arm's own words";
}
function jointFailureLine(f) {
  if (!f) return null;
  const tail = f.tailMisleads ? ' · its log then says "hardware connected" — that is the PROCESS, not the joints' : "";
  return `no joints: ${f.headline}${f.remedy ? ` — ${f.remedy}` : ""}${tail}`;
}
const VERDICT_TTL_MS = 6e4;
function verdictIsStale(atMs, nowMs, ttlMs = VERDICT_TTL_MS) {
  if (!atMs) return true;
  if (atMs > nowMs) return true;
  return nowMs - atMs >= ttlMs;
}
const CACHE = /* @__PURE__ */ new Map();
const INFLIGHT = /* @__PURE__ */ new Map();
const LISTENERS = /* @__PURE__ */ new Set();
const NO_LOG = {
  line: "no joints, and no log to read — this arm was started outside the dashboard, so its reason is in the console that launched it",
  badge: "no log to read — started outside the dashboard"
};
const TICK_MS = 3e4;
async function load(peerId) {
  const cached2 = CACHE.get(peerId);
  if (cached2 && !verdictIsStale(cached2.at, Date.now() + TICK_MS / 6)) return cached2;
  const running = INFLIGHT.get(peerId);
  if (running) return running;
  const p = (async () => {
    let entry;
    try {
      const r = await api(`/api/devices/logs/${encodeURIComponent(peerId)}`);
      const f = jointFailure(r == null ? void 0 : r.lines);
      entry = { line: jointFailureLine(f), badge: jointFailureBadge(f), at: Date.now() };
    } catch {
      entry = { ...NO_LOG, at: Date.now() };
    }
    CACHE.set(peerId, entry);
    INFLIGHT.delete(peerId);
    for (const l of LISTENERS) l();
    return entry;
  })();
  INFLIGHT.set(peerId, p);
  return p;
}
function forgetJointFailure(peerId) {
  if (arguments.length === 0) {
    CACHE.clear();
    return;
  }
  if (peerId) CACHE.delete(peerId);
}
function useJointFailure(peerId, enabled) {
  const [entry, setEntry] = reactExports.useState(() => CACHE.get(peerId) ?? { line: null, badge: null });
  reactExports.useEffect(() => {
    if (!enabled) {
      setEntry({ line: null, badge: null });
      return;
    }
    let live = true;
    const seen = CACHE.get(peerId);
    if (seen) setEntry(seen);
    const tick = setInterval(() => {
      void load(peerId).then((e) => {
        if (live) setEntry(e);
      });
    }, TICK_MS);
    void load(peerId).then((e) => {
      if (live) setEntry(e);
    });
    const onChange = () => {
      const e = CACHE.get(peerId);
      if (live && e) setEntry(e);
    };
    LISTENERS.add(onChange);
    return () => {
      live = false;
      clearInterval(tick);
      LISTENERS.delete(onChange);
    };
  }, [peerId, enabled]);
  return entry;
}
function CameraConfigSheet({ peerId, onClose, focusCam = null, startAdding = false }) {
  var _a;
  const { config } = useConfig();
  const [rows, setRows] = reactExports.useState(null);
  const askedFps = (rows ?? []).map((r) => Number(r.fps)).filter((n) => Number.isFinite(n) && n > 0);
  const rateNote = previewRateNote(askedFps.length ? Math.max(...askedFps) : null, (_a = config == null ? void 0 : config.mesh) == null ? void 0 : _a.camera_hz);
  const [detected, setDetected] = reactExports.useState([]);
  const [notManaged, setNotManaged] = reactExports.useState(false);
  const [confirming, setConfirming] = reactExports.useState(false);
  const [busy, setBusy] = reactExports.useState(false);
  const [error, setError] = reactExports.useState(null);
  const [done, setDone] = reactExports.useState(null);
  const opened = reactExports.useRef({ focusCam, startAdding });
  const focus = rows !== null ? focusTarget(rows, opened.current.focusCam, opened.current.startAdding) : null;
  reactExports.useEffect(() => {
    let alive = true;
    api("/api/devices").then((doc) => {
      var _a2;
      if (!alive) return;
      const m = (_a2 = doc == null ? void 0 : doc.managed) == null ? void 0 : _a2[peerId];
      if (!m) {
        setNotManaged(true);
        setRows([]);
      } else {
        const base = rowsFromConfig(m.cameras ?? {});
        setRows(opened.current.startAdding ? [...base, { name: "", indexOrPath: "", fps: "", width: "", height: "" }] : base);
      }
      setDetected(((doc == null ? void 0 : doc.cameras) ?? []).map((c) => ({
        index: c.index,
        label: c.label ?? c.name ?? null,
        in_use_by: c.claimed_by ?? null
      })));
    }).catch((e) => {
      if (alive) {
        setError((e == null ? void 0 : e.message) ?? String(e));
        setRows([]);
      }
    });
    return () => {
      alive = false;
    };
  }, [peerId]);
  const edit = (i, patch) => setRows((rs) => rs.map((r, j) => j === i ? { ...r, ...patch } : r));
  const remove = (i) => setRows((rs) => rs.filter((_, j) => j !== i));
  const add = (indexOrPath = "") => setRows((rs) => [...rs ?? [], { name: "", indexOrPath, fps: "", width: "", height: "" }]);
  const [probes, setProbes] = reactExports.useState({});
  const probe = async (idx) => {
    setProbes((p) => ({ ...p, [idx]: { ...p[idx], busy: true, error: void 0 } }));
    try {
      const r = await api(`/api/devices/camera/${idx}/modes`);
      setProbes((p) => ({ ...p, [idx]: { modes: (r == null ? void 0 : r.modes) ?? [] } }));
    } catch (e) {
      setProbes((p) => ({
        ...p,
        [idx]: {
          error: (e == null ? void 0 : e.status) === 404 ? "this dashboard process predates mode probing — type the values by hand" : (e == null ? void 0 : e.message) ?? String(e)
        }
      }));
    }
  };
  const check = rows ? configFromRows(rows) : { cameras: null };
  const apply = async () => {
    var _a2;
    if (!rows) return;
    setBusy(true);
    setError(null);
    try {
      forgetJointFailure(peerId);
      const r = await post(`/api/devices/${encodeURIComponent(peerId)}/cameras`, { cameras: check.cameras });
      if (r == null ? void 0 : r.error) setError(r.error);
      else if ((r == null ? void 0 : r.status) === "gone") setError(`${peerId} despawned during the respawn window — check devices › logs`);
      else setDone((r == null ? void 0 : r.status) === "running" ? `✓ ${peerId} is back on the mesh with the new cameras` : `respawned, not yet announced on the mesh (status: ${(r == null ? void 0 : r.status) ?? "starting"}) — watch its card, or check devices › logs if it stays away`);
    } catch (e) {
      setError((e == null ? void 0 : e.status) === 404 && /Not Found|no such|unknown managed/i.test((e == null ? void 0 : e.message) ?? "") ? ((_a2 = e == null ? void 0 : e.message) == null ? void 0 : _a2.includes("unknown managed")) ? e.message : "this dashboard process predates the camera-reconfigure rail — it needs a (terminal-started) restart to pick it up" : (e == null ? void 0 : e.message) ?? String(e));
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  };
  const free = detected.filter((d) => !d.in_use_by || d.in_use_by === peerId);
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-backdrop", onClick: busy ? void 0 : onClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet", onClick: (e) => e.stopPropagation(), children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
      "cameras — ",
      peerId
    ] }),
    rows === null && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "reading the current config…" }),
    notManaged && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "This robot was not spawned by this dashboard, so its process (and its cameras) belong to whoever started it — change the config where it runs." }),
    rows !== null && !notManaged && !done && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      rows.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "No cameras attached. Add one below." }),
      rows.map((r, i) => {
        var _a2;
        const iop = parseIndexOrPath(r.indexOrPath);
        const idx = typeof iop === "number" ? iop : null;
        const pr = idx !== null ? probes[idx] : void 0;
        return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cam-config-row", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                placeholder: "name (top / wrist)",
                "aria-label": `camera ${i + 1} name`,
                value: r.name,
                autoFocus: (focus == null ? void 0 : focus.index) === i && focus.field === "name",
                onChange: (e) => edit(i, { name: e.target.value })
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx("input", { placeholder: "index or path", "aria-label": `camera ${i + 1} index or path`, value: r.indexOrPath, onChange: (e) => edit(i, { indexOrPath: e.target.value }) }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                placeholder: "fps",
                "aria-label": `camera ${i + 1} fps`,
                inputMode: "numeric",
                value: r.fps,
                autoFocus: (focus == null ? void 0 : focus.index) === i && focus.field === "fps",
                onChange: (e) => edit(i, { fps: e.target.value })
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx("input", { placeholder: "width", "aria-label": `camera ${i + 1} width`, inputMode: "numeric", value: r.width, onChange: (e) => edit(i, { width: e.target.value }) }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("input", { placeholder: "height", "aria-label": `camera ${i + 1} height`, inputMode: "numeric", value: r.height, onChange: (e) => edit(i, { height: e.target.value }) }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", title: "detach this camera", onClick: () => remove(i), children: "detach" })
          ] }),
          idx !== null && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cam-config-modes", children: [
            !(pr == null ? void 0 : pr.modes) && /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn ghost",
                disabled: pr == null ? void 0 : pr.busy,
                onClick: () => probe(idx),
                title: "set + read back each candidate on the device — offers only what the camera agreed to",
                children: (pr == null ? void 0 : pr.busy) ? "asking the camera…" : "real modes"
              }
            ),
            (pr == null ? void 0 : pr.error) && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint", children: [
              "⚠ ",
              pr.error
            ] }),
            (pr == null ? void 0 : pr.modes) && pr.modes.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", children: "the camera verified no modes — the driver's defaults still work" }),
            (_a2 = pr == null ? void 0 : pr.modes) == null ? void 0 : _a2.map((m) => /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                className: "btn ghost",
                title: "fill fps/size with a mode this camera verified",
                onClick: () => edit(i, { fps: String(m.fps), width: String(m.width), height: String(m.height) }),
                children: [
                  m.width,
                  "×",
                  m.height,
                  " @ ",
                  m.fps
                ]
              },
              `${m.width}x${m.height}@${m.fps}`
            ))
          ] })
        ] }, i);
      }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cam-config-add", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => add(), children: "+ add camera" }),
        free.map((d) => /* @__PURE__ */ jsxRuntimeExports.jsxs(
          "button",
          {
            className: "btn ghost",
            title: d.label ?? void 0,
            onClick: () => add(String(d.index)),
            children: [
              "+ #",
              d.index,
              d.label ? ` ${d.label}` : ""
            ]
          },
          d.index
        ))
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Blank fps/size = the driver's defaults (640×480 @ 30). Detaching all cameras is allowed — the robot streams joints only." }),
      rateNote && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        "⏱ ",
        rateNote
      ] }),
      check.error && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", role: "alert", children: [
        "✗ ",
        check.error
      ] }),
      error && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", role: "alert", children: [
        "⚠ ",
        error
      ] }),
      !confirming ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", disabled: !!check.error || busy, onClick: () => setConfirming(true), children: "apply…" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, disabled: busy, children: "cancel" })
      ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: applySummary(rows, peerId) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn danger", onClick: apply, disabled: busy, children: busy ? "restarting…" : "restart with these cameras" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setConfirming(false), disabled: busy, children: "back" })
        ] })
      ] })
    ] }),
    done && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "result ok", role: "status", children: done }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, children: "close" }) })
    ] }),
    notManaged && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, children: "close" }) })
  ] }) });
}
const RADIAN_CEILING = 4;
const RADIAN_FLOOR = 3.2;
const SWITCH_FRAMES = 8;
const SERVO_SPAN = { lo: -100, hi: 100 };
const SERVO_GRIPPER_SPAN = { lo: 0, hi: 100 };
const RADIAN_SPAN = { lo: -Math.PI, hi: Math.PI };
const ONE_SIDED_NAME = /(gripper|grip|jaw|finger|claw)/i;
function isOneSidedJoint(name) {
  return ONE_SIDED_NAME.test(name);
}
function defaultSpan(name, unit) {
  if (unit === "servo") return isOneSidedJoint(name) ? SERVO_GRIPPER_SPAN : SERVO_SPAN;
  return RADIAN_SPAN;
}
function frameEvidence(samples) {
  let peak = 0;
  let sawFinite = false;
  for (const [, pos] of samples) {
    if (!Number.isFinite(pos)) continue;
    sawFinite = true;
    peak = Math.max(peak, Math.abs(pos));
  }
  if (!sawFinite) return void 0;
  if (peak === 0) return void 0;
  if (peak > RADIAN_CEILING) return "servo";
  if (peak <= RADIAN_FLOOR) return "radian";
  return void 0;
}
function decideStripScale(samples, prev, switchFrames = SWITCH_FRAMES) {
  const evidence = frameEvidence(samples);
  let unit = (prev == null ? void 0 : prev.unit) ?? evidence ?? "radian";
  let pending = null;
  let pendingFrames = 0;
  if (prev && !evidence) {
    pending = prev.pending;
    pendingFrames = prev.pendingFrames;
  } else if (prev && evidence && evidence !== prev.unit) {
    pendingFrames = prev.pending === evidence ? prev.pendingFrames + 1 : 1;
    if (pendingFrames >= switchFrames) {
      unit = evidence;
      pending = null;
      pendingFrames = 0;
    } else {
      pending = evidence;
    }
  }
  const carried = prev && unit === prev.unit ? prev.ranges : {};
  const ranges = {};
  for (const [name, raw] of samples) {
    const pos = Number.isFinite(raw) ? raw : 0;
    const base = carried[name] ?? defaultSpan(name, unit);
    ranges[name] = { lo: Math.min(base.lo, pos), hi: Math.max(base.hi, pos) };
  }
  return { unit, pending, pendingFrames, ranges };
}
function fillPercent(pos, range) {
  const width = range.hi - range.lo || 1;
  const pct = (pos - range.lo) / width * 100;
  if (!Number.isFinite(pct)) return 0;
  return Math.max(0, Math.min(100, pct));
}
const HISTORY_WINDOW_MS = 6e4;
const MAX_POINTS = 900;
const GAP_MS = 400;
function createHistory() {
  return /* @__PURE__ */ new Map();
}
function pushFrame(history2, samples, now, windowMs = HISTORY_WINDOW_MS) {
  for (const [name, v] of samples) {
    if (!Number.isFinite(v)) continue;
    let track = history2.get(name);
    if (!track) {
      track = [];
      history2.set(name, track);
    }
    const last = track[track.length - 1];
    if (last && last.t === now) {
      last.v = v;
      continue;
    }
    track.push({ t: now, v });
    const cutoff = now - windowMs;
    let drop = 0;
    while (drop < track.length && track[drop].t < cutoff) drop++;
    if (drop) track.splice(0, drop);
    if (track.length > MAX_POINTS) track.splice(0, track.length - MAX_POINTS);
  }
  return history2;
}
function traceFor(track, now, range, w, h, windowMs = HISTORY_WINDOW_MS, gapMs = GAP_MS) {
  if (!track || track.length === 0 || w <= 0 || h <= 0) return [];
  const span = range.hi - range.lo;
  const out = [];
  const oldest = now - windowMs;
  for (let i = 0; i < track.length; i++) {
    const s = track[i];
    if (s.t < oldest) continue;
    const x = w - (now - s.t) / windowMs * w;
    const frac = span > 0 ? (s.v - range.lo) / span : 0.5;
    const y = h - Math.min(1, Math.max(0, frac)) * h;
    const next = track[i + 1];
    out.push({ x, y, gapAfter: !!next && next.t - s.t > gapMs });
  }
  return out;
}
function historyClaim(subject, track, now, windowMs = HISTORY_WINDOW_MS) {
  const windowS = Math.round(windowMs / 1e3);
  const held = heldSeconds(track, now);
  if (held < 1) return `no movement history for ${subject} yet`;
  if (held >= windowS * 0.97) return `last ${windowS}s of ${subject}`;
  return `${Math.round(held)}s of ${subject} so far — the ${windowS}s window is not full yet`;
}
function heldSeconds(track, now) {
  if (!track || track.length < 2) return 0;
  return Math.max(0, (now - track[0].t) / 1e3);
}
function stalled(track, now, gapMs = GAP_MS * 2) {
  if (!track || track.length === 0) return false;
  return now - track[track.length - 1].t > gapMs;
}
function JointSpark({
  track,
  range,
  frame,
  height = 22,
  live = true
}) {
  const ref = reactExports.useRef(null);
  reactExports.useEffect(() => {
    var _a;
    const canvas = ref.current;
    if (!canvas) return;
    let raf = 0;
    let timer;
    const draw = () => {
      const parent = canvas.parentElement;
      const cssW = Math.max(1, (parent == null ? void 0 : parent.clientWidth) ?? canvas.clientWidth);
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(height * dpr);
        canvas.style.width = `${cssW}px`;
        canvas.style.height = `${height}px`;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, height);
      const now = Date.now();
      const pts = traceFor(track, now, range, cssW, height);
      const styles = getComputedStyle(canvas);
      ctx.strokeStyle = styles.getPropertyValue("--spark-axis").trim() || "rgba(255,255,255,.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, Math.round(height / 2) + 0.5);
      ctx.lineTo(cssW, Math.round(height / 2) + 0.5);
      ctx.stroke();
      if (pts.length === 0) return;
      const line = styles.getPropertyValue("--spark-line").trim() || "#4a9eff";
      const fill = ctx.createLinearGradient(0, 0, 0, height);
      fill.addColorStop(0, styles.getPropertyValue("--spark-fill").trim() || "rgba(74,158,255,.22)");
      fill.addColorStop(1, "rgba(0,0,0,0)");
      let i = 0;
      while (i < pts.length) {
        let j = i;
        while (j < pts.length - 1 && !pts[j].gapAfter) j++;
        const seg = pts.slice(i, j + 1);
        if (seg.length > 1) {
          ctx.beginPath();
          ctx.moveTo(seg[0].x, seg[0].y);
          for (const p of seg.slice(1)) ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = line;
          ctx.lineWidth = 1.4;
          ctx.lineJoin = "round";
          ctx.stroke();
          ctx.lineTo(seg[seg.length - 1].x, height);
          ctx.lineTo(seg[0].x, height);
          ctx.closePath();
          ctx.fillStyle = fill;
          ctx.fill();
        }
        i = j + 1;
      }
      const head = pts[pts.length - 1];
      ctx.beginPath();
      ctx.arc(head.x, head.y, 1.9, 0, Math.PI * 2);
      ctx.fillStyle = line;
      ctx.fill();
    };
    raf = requestAnimationFrame(draw);
    const calm = !!((_a = window.matchMedia) == null ? void 0 : _a.call(window, "(prefers-reduced-motion: reduce)").matches);
    if (live) {
      timer = window.setInterval(() => {
        if (calm && !stalled(track, Date.now())) return;
        raf = requestAnimationFrame(draw);
      }, calm ? 1e3 : 250);
    }
    return () => {
      cancelAnimationFrame(raf);
      if (timer) window.clearInterval(timer);
    };
  }, [track, range.lo, range.hi, frame, height, live]);
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    "canvas",
    {
      ref,
      className: "jspark",
      role: "img",
      "aria-label": historyClaim("movement", track, Date.now())
    }
  );
}
function humanJointName(key) {
  const bare = key.replace(/(_pos|\.pos)$/, "");
  const words2 = bare.replace(/[_.]+/g, " ").replace(/\s+/g, " ").trim();
  if (!words2) return key;
  return words2.charAt(0).toUpperCase() + words2.slice(1);
}
function humanJointNames(keys) {
  const human2 = keys.map(humanJointName);
  const collides = new Set(human2).size !== human2.length;
  return collides ? keys.slice() : human2;
}
function stripLegend(unit, windowMs, jointNames = []) {
  const secs = Math.round(windowMs / 1e3);
  const hasGripper = jointNames.some(isOneSidedJoint);
  const units = unit === "radian" ? "values in radians" : hasGripper ? "values in degrees, gripper on its own 0…100 scale" : "values in degrees";
  return `${units} · bar = position within travel seen so far · line = last ${secs}s`;
}
const LAGGING_MS = 2e3;
const FROZEN_MS = 1e4;
function human(ms) {
  const s = ms / 1e3;
  return s >= 10 ? `${Math.round(s)}s` : `${s.toFixed(1)}s`;
}
function jointAgeNote(ageMs) {
  if (ageMs === null || ageMs === void 0 || !Number.isFinite(ageMs)) {
    return { level: "unknown", text: null, dim: false };
  }
  const age = Math.max(0, ageMs);
  if (age < LAGGING_MS) return { level: "live", text: null, dim: false };
  if (age < FROZEN_MS) {
    return {
      level: "lagging",
      text: `⚠ these numbers are ${human(age)} old — the state stream is lagging`,
      dim: false
    };
  }
  return {
    level: "frozen",
    // Name the failure AND the wrong conclusion, because the wrong conclusion is
    // the one a hand acts on.
    text: `⚠ frozen ${human(age)} ago — this is the last frame received, not where the arm is now`,
    dim: true
  };
}
const STATE_QUIET_S = 10;
function failingForText(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 10) return null;
  if (seconds < 90) return `for ${Math.round(seconds)}s`;
  if (seconds < 5400) return `for ${Math.round(seconds / 60)}m`;
  return `for ${(seconds / 3600).toFixed(1)}h`;
}
function expectsJoints(presence) {
  var _a;
  const n = (_a = presence == null ? void 0 : presence.action_keys) == null ? void 0 : _a.length;
  if (typeof n === "number" && n > 0) return n;
  if (presence == null ? void 0 : presence.hw) return "yes";
  return "unknown";
}
function jointAbsence(input) {
  const { state, presence, problem, nowS } = input;
  const verdict2 = (problem == null ? void 0 : problem.headline) ? problem : null;
  const expects = expectsJoints(presence);
  const ageS2 = typeof (state == null ? void 0 : state.t) === "number" && state.t > 0 ? nowS - state.t : null;
  const stateArriving = ageS2 !== null && ageS2 <= STATE_QUIET_S;
  if (state == null || ageS2 === null) {
    if (verdict2) {
      return {
        text: `no state frames yet — ${verdict2.headline}`,
        tone: "attention",
        hint: verdict2.remedy ?? null,
        detail: verdict2.detail ?? null
      };
    }
    if (expects === "unknown") return { text: "no joint data on this peer", tone: "none", hint: null };
    const count2 = expects === "yes" ? "" : ` (${expects} joints expected)`;
    return { text: `waiting for the first state frame${count2}`, tone: "waiting", hint: null };
  }
  if (!stateArriving) {
    const ago2 = ageS2 < 90 ? `${Math.round(ageS2)}s` : ageS2 < 5400 ? `${Math.round(ageS2 / 60)}m` : `${(ageS2 / 3600).toFixed(1)}h`;
    if (input.peerStale === false) {
      return {
        text: `state is ${ago2} behind — no joints in it`,
        tone: "attention",
        hint: (verdict2 == null ? void 0 : verdict2.remedy) ?? "the process is alive (presence is current), so the servo-bus read is what fails: devices > logs for this peer",
        detail: (verdict2 == null ? void 0 : verdict2.detail) ?? null
      };
    }
    return {
      text: `state went quiet ${ago2} ago — no joints since`,
      tone: "attention",
      hint: (verdict2 == null ? void 0 : verdict2.remedy) ?? "the peer or the mesh, not the servo bus: its process may have exited",
      detail: (verdict2 == null ? void 0 : verdict2.detail) ?? null
    };
  }
  if (verdict2) {
    const lasting = failingForText(verdict2.for_seconds);
    const extras = [];
    if (verdict2.detail) extras.push(verdict2.detail);
    if (typeof verdict2.failures === "number" && verdict2.failures > 1) {
      extras.push(`${verdict2.failures} consecutive failed reads`);
    }
    if (extras.length > 0) {
      extras.push(verdict2.source === "peer" ? "reported by the robot itself, and it clears when the read works again" : "read from this robot's log, and it clears only when the log records a recovery");
    }
    return {
      text: lasting ? `no joint positions ${lasting} — ${verdict2.headline}` : `no joint positions — ${verdict2.headline}`,
      tone: "attention",
      hint: verdict2.remedy ?? "check its log (devices → logs)",
      detail: extras.length > 0 ? extras.join(" · ") : null
    };
  }
  if (expects === "unknown") {
    return { text: "this peer publishes state without joint positions", tone: "none", hint: null };
  }
  const count = expects === "yes" ? "" : ` — ${expects} expected`;
  return {
    text: `state is arriving, but carries no joint positions${count}`,
    tone: "attention",
    hint: "the arm is alive and talking; a safety lockout and a failed bus read both look like this — check its log (devices → logs)"
  };
}
function readValue(v) {
  if (typeof v === "number") return v;
  if (Array.isArray(v)) return v[0] ?? 0;
  if (v && typeof v === "object") return v.position ?? 0;
  return 0;
}
function JointStrip({
  state,
  presence,
  problem,
  peerStale,
  history: showHistory = true
}) {
  const memo = reactExports.useRef(void 0);
  const hist = reactExports.useRef(createHistory());
  const [frame, setFrame] = reactExports.useState(0);
  const pending = reactExports.useRef(null);
  const lastAt = reactExports.useRef(null);
  const [nowMs, setNowMs] = reactExports.useState(() => Date.now());
  reactExports.useEffect(() => {
    if (!pending.current) return;
    lastAt.current = Date.now();
    setNowMs(lastAt.current);
    if (!showHistory) return;
    pushFrame(hist.current, pending.current, lastAt.current);
    setFrame((f) => f + 1);
  }, [state, showHistory]);
  reactExports.useEffect(() => {
    const t = setInterval(() => setNowMs(Date.now()), 1e3);
    return () => clearInterval(t);
  }, []);
  const joints = state == null ? void 0 : state.joints;
  if (!joints || Object.keys(joints).length === 0) {
    const note = jointAbsence({ state, presence, problem, peerStale, nowS: nowMs / 1e3 });
    return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "joints empty", "data-tone": note.tone, title: note.detail ?? void 0, children: [
      note.tone === "attention" ? "⚠ " : "",
      note.text,
      note.hint && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", title: typeof note.hint === "string" ? note.hint : void 0, children: note.hint })
    ] });
  }
  const entries = Object.entries(joints).slice(0, 12);
  const samples = entries.map(([name, v]) => [name, readValue(v)]);
  memo.current = decideStripScale(samples, memo.current);
  const { unit, ranges } = memo.current;
  pending.current = samples;
  const fmt2 = (n) => Math.abs(n) >= 100 ? n.toFixed(0) : Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(2);
  const labels = humanJointNames(entries.map(([name]) => name));
  const fresh = jointAgeNote(lastAt.current === null ? null : nowMs - lastAt.current);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "joints", "data-unit": unit, "data-fresh": fresh.level, children: [
    entries.map(([name, v], i) => {
      const pos = samples[i][1];
      const range = ranges[name];
      const pct = fillPercent(pos, range);
      const vel = v && typeof v === "object" && !Array.isArray(v) ? v.velocity : void 0;
      return /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "div",
        {
          className: "joint",
          title: `${name}: ${pos.toFixed(3)}${vel !== void 0 ? ` (v=${vel.toFixed(2)})` : ""} · ${unit} scale ${range.lo.toFixed(2)}…${range.hi.toFixed(2)}`,
          children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "jname", title: name, children: labels[i] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "jval", children: fmt2(pos) }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "jbar", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "jfill", style: { width: `${pct}%` } }),
              vel !== void 0 && Math.abs(vel) > 1e-3 && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "jvel" })
            ] }),
            showHistory && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "jhist", title: historyClaim(name, hist.current.get(name), Date.now()), children: /* @__PURE__ */ jsxRuntimeExports.jsx(JointSpark, { track: hist.current.get(name), range, frame }) })
          ]
        },
        name
      );
    }),
    fresh.text && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: `jstale${fresh.dim ? " frozen" : ""}`, role: "status", "aria-live": "polite", children: fresh.text }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "jlegend", children: stripLegend(unit, HISTORY_WINDOW_MS, entries.map(([n]) => n)) })
  ] });
}
function motionChip(moving, opts = {}) {
  if (moving === true) {
    return {
      tone: "moving",
      label: "moving",
      title: "joints are changing (mean absolute delta over the last samples) - keep hands clear",
      aria: "joints are moving, keep hands clear"
    };
  }
  if (moving === false) {
    return {
      tone: "still",
      label: "still",
      title: "joints are not changing (mean absolute delta over the last samples)",
      aria: "joints measured still"
    };
  }
  if (opts.jointsSeen === false) {
    return {
      tone: "unknown",
      label: "motion unknown",
      title: "this robot publishes no joint positions, so movement cannot be measured here - treat the arm as able to move",
      aria: "motion unknown, this robot publishes no joint positions"
    };
  }
  return {
    tone: "unknown",
    label: "measuring",
    title: "not enough state samples yet to judge movement (about a second of telemetry)",
    aria: "motion not measured yet"
  };
}
function TelemetryStrip({ peer }) {
  var _a;
  const { samples, hz, moving, stateAgeS, jointsSeen } = useTelemetry(peer);
  if (samples.length < 2) return null;
  const peak = Math.max(...samples.map((s) => s.motion), 1e-6);
  const age = stateAgeS ?? 0;
  const points = samples.map((s, i) => `${i / (TELEMETRY_CAP - 1) * 100},${20 - s.motion / peak * 18}`).join(" ");
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "telemetry", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("svg", { className: "spark", viewBox: "0 0 100 20", preserveAspectRatio: "none", "aria-hidden": true, children: /* @__PURE__ */ jsxRuntimeExports.jsx("polyline", { points }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "metric", title: "measured rate of this robot's state topic (nominal 10 Hz)", children: [
      "state ",
      hz.toFixed(1),
      " Hz"
    ] }),
    (() => {
      const chip = motionChip(moving, { jointsSeen });
      return /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: `motionchip ${chip.tone}`, title: chip.title, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "motiondot", "aria-hidden": true }),
        chip.label,
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "sr-only", children: chip.aria })
      ] });
    })(),
    age > 3 && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "metric warn", title: "no state message recently", children: [
      "stale ",
      age.toFixed(0),
      "s"
    ] }),
    ((_a = peer.state) == null ? void 0 : _a.sim_time) !== void 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "metric", title: "simulation clock", children: [
      "t=",
      peer.state.sim_time.toFixed(1)
    ] })
  ] });
}
function norm$1(v) {
  if (v === null || v === void 0) return "";
  if (typeof v === "object") {
    try {
      return JSON.stringify(v, Object.keys(v).sort());
    } catch {
      return String(v);
    }
  }
  return String(v);
}
function changedKeys(a, b) {
  const keys = /* @__PURE__ */ new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})]);
  const out = [];
  for (const k of keys) {
    if (norm$1(a == null ? void 0 : a[k]) !== norm$1(b == null ? void 0 : b[k])) out.push(k);
  }
  return out.sort();
}
function validationScope(validated, current) {
  if (!validated) return { applies: true, changed: [], note: "" };
  const changed = changedKeys(validated.config ?? {}, current.config ?? {});
  if (validated.provider !== current.provider) changed.unshift("provider");
  if (changed.length === 0) return { applies: true, changed: [], note: "" };
  const shown = changed.slice(0, 3).join(", ");
  const more = changed.length > 3 ? ` +${changed.length - 3} more` : "";
  return {
    applies: false,
    changed,
    // Names WHAT moved: "something changed" would send them hunting.
    note: `this verdict was taken before ${shown}${more} changed — it does not describe the form as it is now. Validate again.`
  };
}
function numField(raw, rules) {
  const text = (raw ?? "").trim();
  const { what, min, max, integer = true, remedy } = rules;
  if (!text) return { value: 0, problem: `how many ${what}?`, note: null };
  const n = Number(text);
  if (!Number.isFinite(n)) return { value: 0, problem: `“${text}” is not a number`, note: null };
  if (n < min) {
    return {
      value: 0,
      note: null,
      problem: n < 0 ? `${what} cannot be negative` : `${n} is below the minimum of ${min} ${what}`
    };
  }
  if (n > max) {
    return {
      value: 0,
      note: null,
      problem: `${n} is more than this screen will start (max ${max} ${what})${remedy ? ` — ${remedy}` : ""}`
    };
  }
  if (integer && !Number.isInteger(n)) {
    const floored = Math.floor(n);
    if (floored < min) return { value: 0, problem: `${what} cannot be less than ${min}`, note: null };
    return { value: floored, problem: null, note: `using ${floored} ${what} — ${text} is not a whole number` };
  }
  return { value: n, problem: null, note: null };
}
function isLatestRequest(seq, latest) {
  return seq === latest;
}
function newerThanApplied(seq, applied) {
  return applied === void 0 || seq > applied;
}
function emptyNote({ query, hubProblem }) {
  const q = String(query ?? "").trim();
  const named = q ? `“${q}”` : "that";
  const problem = String(hubProblem ?? "").trim();
  if (problem) {
    return q ? `no checkpoint already on this machine matches ${named} — and the Hub was not searched (${problem}), so this is not "it does not exist".` : `nothing in this machine's local cache — and the Hub was not searched (${problem}), so the catalogue is unknown from here.`;
  }
  return q ? `no checkpoints match ${named} (local cache + Hub).` : "type part of a checkpoint name — local cache and the Hub are both searched.";
}
function CheckpointPicker({ value, onPick, disabled }) {
  const [query, setQuery] = reactExports.useState(value);
  const [rows, setRows] = reactExports.useState([]);
  const [open, setOpen] = reactExports.useState(false);
  const [loading, setLoading] = reactExports.useState(false);
  const [hubProblem, setHubProblem] = reactExports.useState(null);
  const [hfAuth, setHfAuth] = reactExports.useState(null);
  const [failed, setFailed] = reactExports.useState(null);
  const debounce = reactExports.useRef();
  const seq = reactExports.useRef(0);
  const [shownQuery, setShownQuery] = reactExports.useState("");
  const rootRef = reactExports.useRef(null);
  reactExports.useEffect(() => {
    setQuery(value);
  }, [value]);
  reactExports.useEffect(() => {
    const close = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const searchNow = (q) => {
    clearTimeout(debounce.current);
    const mine = ++seq.current;
    debounce.current = setTimeout(async () => {
      setLoading(true);
      try {
        const j = await api(`/api/checkpoints/search?q=${encodeURIComponent(q)}&limit=12`);
        if (!isLatestRequest(mine, seq.current)) return;
        setRows(j.results ?? []);
        setHubProblem(j.hub_problem ?? null);
        setHfAuth(j.hf_auth ?? null);
        setFailed(null);
        setShownQuery(q);
        setOpen(true);
      } catch (e) {
        if (!isLatestRequest(mine, seq.current)) return;
        setRows([]);
        setFailed((e == null ? void 0 : e.message) ?? String(e));
        setShownQuery(q);
        setOpen(true);
      } finally {
        if (isLatestRequest(mine, seq.current)) setLoading(false);
      }
    }, 300);
  };
  const fmt2 = (n) => n == null ? "" : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k` : String(n);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "ckpt", ref: rootRef, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx(
      "input",
      {
        placeholder: "search checkpoints… (e.g. smolvla, act so101)",
        "aria-label": "search checkpoints",
        value: query,
        onChange: (e) => {
          setQuery(e.target.value);
          onPick(e.target.value, null);
          searchNow(e.target.value);
        },
        onFocus: () => {
          if (rows.length) setOpen(true);
          else searchNow(query);
        },
        disabled
      }
    ),
    loading && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "ckpt-spin", children: "…" }),
    open && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "ckpt-menu", children: [
      failed && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "ckpt-note bad", children: [
        "✗ search failed: ",
        failed
      ] }),
      !failed && hubProblem && rows.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "ckpt-note warn", children: [
        "⚠ ",
        hubProblem
      ] }),
      !failed && hfAuth && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: `ckpt-note ${hfAuth.authenticated ? "ok" : ""}`, children: hfAuth.authenticated ? `HF: signed in as ${hfAuth.user} — private + gated repos reachable` : `HF: anonymous — ${hfAuth.detail ?? "public repos only"}` }),
      rows.length === 0 && !failed && // Scoped to what was actually consulted: with the Hub down, only the
      // local cache answered, and "no checkpoints match" would be a claim
      // about a catalogue nobody asked.
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: hubProblem ? "ckpt-note warn" : "ckpt-note", children: emptyNote({ query: shownQuery, hubProblem }) }),
      rows.map((r) => /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: "ckpt-row",
          onMouseDown: (e) => e.preventDefault(),
          onClick: () => {
            onPick(r.repo_id, r.policy_type);
            setQuery(r.repo_id);
            setOpen(false);
          },
          children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "ckpt-id", children: r.repo_id }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "ckpt-meta", children: [
              r.local && /* @__PURE__ */ jsxRuntimeExports.jsx("b", { className: "ckpt-local", children: "local" }),
              r.policy_type && /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: r.policy_type }),
              r.downloads != null && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
                "↓",
                fmt2(r.downloads)
              ] })
            ] })
          ]
        },
        r.repo_id
      ))
    ] })
  ] });
}
const KEY = "strands.deployIntent";
const TTL_MS = 10 * 60 * 1e3;
const CLOCK_GRACE_MS = 60 * 1e3;
function setDeployIntent(i) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ ...i, at: Date.now() }));
  } catch {
  }
}
function peekDeployIntent(now = Date.now()) {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const i = JSON.parse(raw);
    if (!i || typeof i.checkpoint !== "string" || !i.checkpoint) return null;
    const age = now - i.at;
    if (typeof i.at !== "number" || !Number.isFinite(age) || age > TTL_MS || age < -CLOCK_GRACE_MS) {
      sessionStorage.removeItem(KEY);
      return null;
    }
    return i;
  } catch {
    return null;
  }
}
function clearDeployIntent() {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
  }
}
function runRisk(presence) {
  const hw = typeof (presence == null ? void 0 : presence.hw) === "string" ? presence.hw.trim() : "";
  const type = String((presence == null ? void 0 : presence.robot_type) ?? "").toLowerCase();
  if (type === "sim") {
    return { physical: false, reason: "simulated robot — nothing physical moves", device: null };
  }
  if (hw && !/^(sim|mock|fake|mujoco)/i.test(hw)) {
    return { physical: true, reason: `real hardware attached (${hw})`, device: hw };
  }
  if (hw) {
    return { physical: false, reason: `simulated backend (${hw})`, device: hw };
  }
  if ((presence == null ? void 0 : presence.connected) === false) {
    return { physical: true, reason: "hardware is not connected right now", device: null };
  }
  return { physical: true, reason: "this peer did not say whether it is real", device: null };
}
const COPY = {
  // --- checkpoint / model identity -----------------------------------------
  pretrained_name_or_path: {
    label: "checkpoint",
    hint: "a Hugging Face repo id or a local training output directory"
  },
  model_path: { label: "checkpoint path", hint: "a directory on the robot itself" },
  checkpoint: { label: "checkpoint", hint: "a trained policy to load" },
  repo_id: { label: "checkpoint", hint: "a Hugging Face repo id" },
  policy_type: {
    label: "policy family",
    hint: "which architecture the checkpoint is (act, smolvla, diffusion…)"
  },
  model: { label: "model" },
  model_id: { label: "model id" },
  onnx_path: { label: "ONNX file", hint: "an exported model file on the robot" },
  // --- inference server ----------------------------------------------------
  port: { label: "server port", hint: "the inference server must already be running" },
  policy_port: { label: "server port", hint: "the inference server must already be running" },
  host: { label: "server host", hint: "where that inference server listens" },
  policy_host: { label: "server host", hint: "where that inference server listens" },
  server_address: { label: "server address", hint: "host:port of a running inference server" },
  server_port: { label: "server port" },
  endpoint: { label: "endpoint URL" },
  api_key: { label: "API key", hint: "a credential — set it in Settings › Env, not here" },
  api_token: { label: "API token", hint: "a credential — set it in Settings › Env, not here" },
  connect_timeout: { label: "connect timeout (s)" },
  request_timeout: { label: "request timeout (s)" },
  timeout_ms: { label: "timeout (ms)" },
  auto_launch_server: { label: "launch the server for me" },
  server_mode: { label: "server mode" },
  // --- motion / control ----------------------------------------------------
  action_horizon: { label: "action horizon", hint: "how many future steps each inference returns" },
  actions_per_chunk: { label: "actions per chunk" },
  actions_per_step: { label: "actions per step" },
  control_frequency: { label: "control rate (Hz)" },
  n_steps: { label: "steps" },
  fast_mode: { label: "fast mode" },
  target_pose: { label: "target pose", hint: "a cartesian goal, as JSON" },
  target_joints: { label: "target joints", hint: "goal joint positions, as JSON" },
  robot_name: { label: "robot name", hint: "which robot on that peer" },
  robot: { label: "robot" },
  robot_config: { label: "robot config" },
  // --- data plumbing -------------------------------------------------------
  data_config: { label: "data config", hint: "the server's name for this robot's I/O layout" },
  image_keys: { label: "camera keys", hint: "which camera streams the policy expects" },
  observation_mapping: { label: "observation mapping" },
  action_mapping: { label: "action mapping" },
  action_space: { label: "action space" },
  strict_keys: { label: "strict key matching" },
  device: { label: "compute device", hint: "cpu, cuda or mps on the machine that runs the policy" },
  dtype: { label: "numeric precision" },
  cache_dir: { label: "cache directory" },
  trust_remote_code: {
    label: "allow the checkpoint to run its own code",
    hint: "a consent decision — the dashboard asks before it is set"
  },
  prompt: { label: "prompt" },
  text_prompt: { label: "prompt" },
  seed: { label: "random seed" },
  world_config: { label: "world config" },
  world_update: { label: "world update", hint: "obstacles/scene, as JSON" }
};
function fieldCopy(key) {
  const hit = COPY[key];
  if (!hit) return { label: key, known: false };
  return { label: hit.label, hint: hit.hint, known: true };
}
function requirementSummary(keys) {
  const labels = [];
  for (const k of keys) {
    const label2 = fieldCopy(k).label;
    if (!labels.includes(label2)) labels.push(label2);
  }
  return labels.join(" + ");
}
function missingSummary(keys) {
  return keys.map((k) => {
    const c = fieldCopy(k);
    return c.known ? `${c.label} (${k})` : k;
  }).join(", ");
}
function localOnlySummary(keys) {
  return keys.map((k) => {
    const c = fieldCopy(k);
    return c.known ? `${c.label} (${k})` : k;
  }).join(", ");
}
const POLICY_GROUPS = [
  "Safe test",
  "Learned policies (need a checkpoint)",
  "Remote inference (a server does the thinking)",
  "Motion planning (nothing learned)",
  "Humanoid whole-body motion",
  "Other"
];
const KNOWN = {
  mock: { label: "Mock — sine test (safe, no model, moves gently)", group: "Safe test" },
  lerobot_local: { label: "LeRobot — local checkpoint (runs here)", group: "Learned policies (need a checkpoint)" },
  groot: { label: "NVIDIA GR00T — N1.5 or N1.6", group: "Learned policies (need a checkpoint)" },
  cosmos3: { label: "NVIDIA Cosmos 3 — omnimodal VLA", group: "Learned policies (need a checkpoint)" },
  vera: { label: "MIT VERA — video-to-action", group: "Learned policies (need a checkpoint)" },
  molmoact2: { label: "MolmoAct 2 — SO-100 / SO-101", group: "Learned policies (need a checkpoint)" },
  lerobot_async: { label: "LeRobot — remote server (gRPC)", group: "Remote inference (a server does the thinking)" },
  remote: { label: "Remote inference — WebSocket", group: "Remote inference (a server does the thinking)" },
  curobo: { label: "NVIDIA cuRobo — collision-aware planner", group: "Motion planning (nothing learned)" },
  moveit2: { label: "MoveIt 2 — planner bridge", group: "Motion planning (nothing learned)" },
  wbc: { label: "GR00T whole-body control (SONIC)", group: "Humanoid whole-body motion" },
  wbc_gait: { label: "GR00T whole-body control — gait clock", group: "Humanoid whole-body motion" },
  motionbricks: { label: "NVIDIA MotionBricks — G1 motion", group: "Humanoid whole-body motion" },
  kimodo: { label: "NVIDIA Kimodo — G1 text-to-motion", group: "Humanoid whole-body motion" },
  protomotions: { label: "ProtoMotions — G1 motion tracker", group: "Humanoid whole-body motion" }
};
function policyLabel(name) {
  var _a;
  return ((_a = KNOWN[name]) == null ? void 0 : _a.label) ?? name;
}
function policyGroup(name) {
  var _a;
  return ((_a = KNOWN[name]) == null ? void 0 : _a.group) ?? "Other";
}
function groupPolicies(items, nameOf2) {
  const out = [];
  for (const group of POLICY_GROUPS) {
    const inGroup = items.filter((i) => policyGroup(nameOf2(i)) === group);
    if (inGroup.length) out.push({ group, items: inGroup });
  }
  return out;
}
function RunConfirm({
  peerId,
  risk,
  instruction,
  provider,
  model,
  durationS,
  onCancel,
  onConfirm
}) {
  const go = reactExports.useRef(null);
  reactExports.useEffect(() => {
    var _a;
    (_a = go.current) == null ? void 0 : _a.focus();
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-backdrop", onClick: onCancel, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "div",
    {
      className: "sheet run-confirm",
      onClick: (e) => e.stopPropagation(),
      role: "dialog",
      "aria-modal": "true",
      "aria-label": "Confirm running a policy",
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("h2", { children: [
          "⚠️ This will move ",
          peerId
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "rc-lede", children: "A physical arm is about to start moving on its own. Keep hands, cables and anything breakable out of its reach before you start." }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("dl", { className: "rc-facts", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "robot" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("dd", { children: [
              peerId,
              " ",
              /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint", children: [
                "(",
                risk.reason,
                ")"
              ] })
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "task" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("dd", { children: [
              "“",
              instruction,
              "”"
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "policy" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { children: provider })
          ] }),
          model ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "weights" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "rc-mono", children: model })
          ] }) : null,
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "runs for" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { children: durationS ? `${durationS}s, unless you stop it sooner` : "until you stop it" })
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "To stop it: ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "■" }),
          " on this card, ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "STOP ALL" }),
          " in the corner, or press ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "." }),
          " anywhere."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", ref: go, onClick: onCancel, children: "cancel" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn danger big", onClick: onConfirm, children: [
            "start moving ",
            peerId
          ] })
        ] })
      ]
    }
  ) });
}
function RunForm({ peerId, presence, running, busy, disabled, onRun, onStop }) {
  var _a, _b;
  const { policies } = useConfig();
  const [providerName, setProviderName] = reactExports.useState("mock");
  const [instruction, setInstruction] = reactExports.useState("");
  const [durationText, setDurationText] = reactExports.useState("15");
  const wantedDuration = numField(durationText, { what: "seconds", min: 1, max: 600, remedy: "run it again to go longer" });
  const duration = wantedDuration.value;
  const [advanced, setAdvanced] = reactExports.useState(false);
  const [fields, setFields] = reactExports.useState({});
  const [validating, setValidating] = reactExports.useState(false);
  const [validation, setValidation] = reactExports.useState(null);
  const [validatedFor, setValidatedFor] = reactExports.useState(null);
  const [staged, setStaged] = reactExports.useState(null);
  const [pending, setPending] = reactExports.useState(null);
  reactExports.useEffect(() => {
    const intent = peekDeployIntent();
    if (!intent) return;
    const target = policies.find((p) => {
      var _a2;
      return p.wire_safe && ((_a2 = p.wire_fields) == null ? void 0 : _a2.some((f) => f.key === "pretrained_name_or_path" || f.key === "model_path"));
    });
    if (!target) return;
    const pathKey = target.wire_fields.find((f) => f.key === "pretrained_name_or_path" || f.key === "model_path").key;
    setProviderName(target.name);
    setFields((prev) => ({
      ...prev,
      [pathKey]: intent.checkpoint,
      ...intent.policy_type && target.wire_fields.some((f) => f.key === "policy_type") ? { policy_type: intent.policy_type } : {}
    }));
    setStaged(intent);
    setAdvanced(true);
    clearDeployIntent();
  }, [policies]);
  const provider = reactExports.useMemo(
    () => policies.find((p) => p.name === providerName),
    [policies, providerName]
  );
  const wireFields = (provider == null ? void 0 : provider.wire_fields) ?? [];
  const value = (key, fallback) => {
    const raw = fields[key];
    if (raw !== void 0) return raw;
    return fallback === null || fallback === void 0 ? "" : String(fallback);
  };
  const [fit, setFit] = reactExports.useState(null);
  const checkpointField = (_a = wireFields.find(
    (f) => f.key === "pretrained_name_or_path" || f.key === "model_path"
  )) == null ? void 0 : _a.key;
  const checkpoint = checkpointField ? String(value(checkpointField, "")).trim() : "";
  reactExports.useEffect(() => {
    if (!checkpoint || !peerId) {
      setFit(null);
      return;
    }
    let alive = true;
    const t = setTimeout(() => {
      void api(
        `/api/robots/${encodeURIComponent(peerId)}/policy-fit?repo_id=${encodeURIComponent(checkpoint)}`
      ).then((v) => {
        if (alive) setFit(v);
      }).catch(() => {
        if (alive) setFit(null);
      });
    }, 400);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [checkpoint, peerId]);
  const fitBlocked = !!(fit == null ? void 0 : fit.blocking);
  const missing = wireFields.filter((f) => f.required && !String(value(f.key, f.default)).trim()).map((f) => f.key);
  const parseField = (type, raw) => {
    if (type === "int") return Number.parseInt(raw, 10);
    if (type === "float") return Number.parseFloat(raw);
    if (type === "bool") return raw === "true" || raw === "1";
    if (type === "json") return JSON.parse(raw);
    return raw;
  };
  const buildConfig = () => {
    const out = {};
    for (const f of wireFields) {
      const raw = String(value(f.key, f.default)).trim();
      if (!raw) continue;
      try {
        out[f.key] = parseField(f.type, raw);
      } catch {
        out[f.key] = raw;
      }
    }
    return out;
  };
  const submit = () => {
    if (!instruction.trim() || missing.length || wantedDuration.problem) return;
    const body = {
      instruction: instruction.trim(),
      policy_provider: providerName,
      duration
    };
    for (const f of wireFields) {
      const raw = String(value(f.key, f.default)).trim();
      if (!raw) continue;
      try {
        body[f.wire_key] = parseField(f.type, raw);
      } catch {
        setValidation({ ok: false, stage: "form", error: `${f.key}: not valid JSON` });
        return;
      }
    }
    setValidation(null);
    if (runRisk(presence).physical) {
      setPending(body);
      return;
    }
    onRun(body);
  };
  const validate = async () => {
    setValidating(true);
    try {
      const asked = { provider: providerName, config: buildConfig() };
      setValidation(await post("/api/policies/validate", {
        policy_provider: providerName,
        policy_config: asked.config,
        peer_id: peerId
      }));
      setValidatedFor(asked);
    } catch (e) {
      setValidation({ ok: false, stage: "request", error: (e == null ? void 0 : e.message) ?? String(e) });
      setValidatedFor({ provider: providerName, config: buildConfig() });
    } finally {
      setValidating(false);
    }
  };
  const locked = provider && !provider.wire_safe;
  const blocked = !!disabled || busy;
  const modelKey = (_b = wireFields.find(
    (f) => f.key === "pretrained_name_or_path" || f.key === "model_path"
  )) == null ? void 0 : _b.key;
  const modelValue = modelKey ? String(value(modelKey, "") || "").trim() : "";
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "runform", children: [
    pending && /* @__PURE__ */ jsxRuntimeExports.jsx(
      RunConfirm,
      {
        peerId,
        risk: runRisk(presence),
        instruction: pending.instruction,
        provider: providerName,
        model: modelValue || null,
        durationS: duration,
        onCancel: () => setPending(null),
        onConfirm: () => {
          const body = pending;
          setPending(null);
          onRun(body);
        }
      }
    ),
    staged && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "deploy-banner", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        "🚀 prefilled from ",
        staged.source,
        " — review below, then press Run. Nothing has started."
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => {
        setStaged(null);
        setFields({});
      }, children: "discard" })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "controls", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "select",
        {
          value: providerName,
          onChange: (e) => {
            setProviderName(e.target.value);
            setFields({});
            setValidation(null);
            setValidatedFor(null);
          },
          disabled: blocked,
          "aria-label": "Policy — what will drive this robot",
          title: provider ? `${policyLabel(provider.name)} (${provider.name}) — ${provider.description}` : "Policy — what will drive this robot",
          children: [
            policies.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "mock", children: policyLabel("mock") }),
            groupPolicies(policies, (p) => p.name).map((g) => /* @__PURE__ */ jsxRuntimeExports.jsx("optgroup", { label: g.group, children: g.items.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: p.name, disabled: !p.wire_safe, children: [
              p.wire_safe ? "" : "🔒 ",
              policyLabel(p.name),
              p.requires.length ? ` — needs ${requirementSummary(p.requires)}` : ""
            ] }, p.name)) }, g.group))
          ]
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "input",
        {
          "aria-label": "instruction for the policy",
          placeholder: "pick up the red cube",
          value: instruction,
          onChange: (e) => setInstruction(e.target.value),
          onKeyDown: (e) => e.key === "Enter" && submit(),
          disabled: blocked
        }
      ),
      running ? /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn stop", onClick: onStop, disabled: busy, title: "Stop this robot", children: "■" }) : /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn go",
          onClick: submit,
          disabled: blocked || !instruction.trim() || missing.length > 0 || !!locked || !!wantedDuration.problem || fitBlocked,
          ...fitBlocked ? { title: fit.problems.map((p) => p.detail).join(" — ") } : {},
          title: locked ? `${providerName} is not in the mesh policy allowlist` : wantedDuration.problem ? `duration: ${wantedDuration.problem}` : missing.length ? `missing: ${missingSummary(missing)}` : "Run",
          children: "▶"
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: advanced ? "btn ghost on" : "btn ghost",
          onClick: () => setAdvanced((a) => !a),
          title: "Provider options",
          children: "⚙"
        }
      )
    ] }),
    missing.length > 0 && !advanced && /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "needs", onClick: () => setAdvanced(true), children: [
      providerName,
      " needs ",
      missingSummary(missing),
      " → open options"
    ] }),
    wantedDuration.problem && !advanced && /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "needs", onClick: () => setAdvanced(true), children: [
      "duration: ",
      wantedDuration.problem,
      " → open options"
    ] }),
    advanced && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "advanced", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "duration (s)" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: "number",
            min: 1,
            max: 600,
            value: durationText,
            onChange: (e) => setDurationText(e.target.value),
            disabled: blocked,
            "aria-invalid": !!wantedDuration.problem,
            "aria-describedby": "run-duration-say"
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "run-duration-say", className: `fieldsay${wantedDuration.problem ? " bad" : ""}`, children: wantedDuration.problem ?? wantedDuration.note ?? "" })
      ] }),
      wireFields.map((f) => /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: f.required && !String(value(f.key, f.default)).trim() ? "field missing" : "field", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          fieldCopy(f.key).label,
          f.required && /* @__PURE__ */ jsxRuntimeExports.jsx("b", { title: "required", children: " *" }),
          fieldCopy(f.key).known && /* @__PURE__ */ jsxRuntimeExports.jsx("code", { className: "ident", title: "the API field name", children: f.key }),
          f.wire_key !== f.key && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { title: `sent as ${f.wire_key}`, children: [
            " →",
            f.wire_key
          ] })
        ] }),
        f.type === "bool" ? /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: value(f.key, f.default) || "false", onChange: (e) => setFields((s) => ({ ...s, [f.key]: e.target.value })), disabled: blocked, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "false", children: "false" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "true", children: "true" })
        ] }) : f.key === "pretrained_name_or_path" || f.key === "model_path" ? (
          /**
           * Checkpoint fields get a Hub+local-cache type-ahead: the registry names the field, this names
           * the VALUES (thousands of public LeRobot checkpoints).
           */
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            CheckpointPicker,
            {
              value: value(f.key, f.default),
              disabled: blocked,
              onPick: (repoId, policyType) => setFields((s) => {
                const next = { ...s, [f.key]: repoId };
                if (policyType && wireFields.some((w) => w.key === "policy_type") && !s.policy_type) {
                  next.policy_type = policyType;
                }
                return next;
              })
            }
          )
        ) : /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: f.type === "int" || f.type === "float" ? "number" : "text",
            value: value(f.key, f.default),
            placeholder: f.type === "json" ? '{"…": …}' : fieldCopy(f.key).hint ?? f.type,
            onChange: (e) => setFields((s) => ({ ...s, [f.key]: e.target.value })),
            disabled: blocked
          }
        )
      ] }, f.key)),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "advanced-actions", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: validate, disabled: validating, children: validating ? "checking…" : "✓ validate" }),
        (provider == null ? void 0 : provider.server_based) && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", children: "needs a running inference server at that host/port" })
      ] }),
      provider && provider.unsettable_over_mesh.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "hint local-only", children: [
        "these options only work when the policy is built on the robot itself — the mesh command cannot carry them:",
        " ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: localOnlySummary(provider.unsettable_over_mesh) })
      ] }),
      locked && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "hint warn", children: [
        "🔒 ",
        providerName,
        " is rejected by the mesh policy allowlist. Add it to",
        " ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "STRANDS_MESH_POLICY_TYPE_ALLOW" }),
        " on every peer."
      ] })
    ] }),
    fit && (fit.blocking || fit.evidence) && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: fit.blocking ? "validation bad" : "validation ok", role: fit.blocking ? "alert" : void 0, children: fit.blocking ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: "✗ this policy does not fit this robot" }),
      fit.problems.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: p.detail }, p.kind)),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: "Nothing on this form fixes that — pick a checkpoint trained on this robot, or run it in sim." })
    ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      "✓ checkpoint matches this robot (",
      fit.checked.join(", "),
      ")"
    ] }) }),
    validation && (() => {
      const scope = validationScope(validatedFor, { provider: providerName, config: buildConfig() });
      const cls = !scope.applies ? "validation stale" : validation.ok ? "validation ok" : "validation bad";
      return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: cls, children: [
        !scope.applies ? `${validation.ok ? "✓" : "✗"} (outdated) ${validation.ok ? `${(validatedFor == null ? void 0 : validatedFor.provider) ?? providerName} resolved` : `${validation.stage}: ${validation.error}`}` : validation.ok ? validation.resolved === false ? `— nothing to resolve for ${providerName}` : `✓ ${providerName} resolves` : `✗ ${validation.stage}: ${validation.error}`,
        !scope.applies && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint warn", children: scope.note }),
        validation.scope_note && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: validation.scope_note }),
        validation.note && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: validation.note })
      ] });
    })()
  ] });
}
function ConsentSheet({ need, target, onCancel, onRetry }) {
  var _a;
  const cancel = reactExports.useRef(null);
  const [busy, setBusy] = reactExports.useState(false);
  const [note, setNote] = reactExports.useState(null);
  const [error, setError] = reactExports.useState(null);
  const allowed = canApprove(need);
  const danger = severity(need) === "danger";
  reactExports.useEffect(() => {
    var _a2;
    (_a2 = cancel.current) == null ? void 0 : _a2.focus();
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);
  const approve = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await approveConsent(need);
      const verdict2 = afterApproval(result, target);
      setNote(verdict2.note);
      if (verdict2.retryNow) onRetry();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-backdrop", onClick: onCancel, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "div",
    {
      className: "sheet consent-sheet",
      onClick: (e) => e.stopPropagation(),
      role: "dialog",
      "aria-modal": "true",
      "aria-label": "Security approval",
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("h2", { children: [
          danger ? "⚠️" : "🔒",
          " ",
          need.title
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "cs-risk", children: need.risk }),
        ((_a = need.grants) == null ? void 0 : _a.length) ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "cs-label", children: "approving allows" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "cs-grants", children: need.grants.map((g) => /* @__PURE__ */ jsxRuntimeExports.jsx("li", { children: g }, g)) })
        ] }) : null,
        !allowed ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "cs-blocked", children: blockedReason(need) }) : null,
        note ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "cs-note", role: "status", children: note }) : null,
        error ? /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "cs-error", role: "alert", children: [
          "could not save the approval: ",
          error
        ] }) : null,
        need.message ? /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "cs-raw", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "what the refusal said" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: need.message })
        ] }) : null,
        need.env_var ? /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Stored on this machine as ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: need.env_var }),
          " — it survives a restart, and it is listed under Settings → Security → “Permissions you granted”, where you can revoke it."
        ] }) : null,
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", ref: cancel, onClick: onCancel, children: note ? "close" : "cancel" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: `btn ${danger ? "danger" : "primary"}`,
              disabled: !allowed || busy || !!note,
              onClick: approve,
              children: busy ? "saving…" : target === "spawn" ? "approve & start" : "approve"
            }
          )
        ] })
      ]
    }
  ) });
}
function RobotCard({ peer, twinLive = false, onOpen, onBusyChange, hostsChildren }) {
  var _a, _b;
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin, consent, clearConsent, retryLast } = useTask(peer);
  const [sheet, setSheet] = reactExports.useState(false);
  const [camSheet, setCamSheet] = reactExports.useState(null);
  const p = peer.presence;
  const type = (p == null ? void 0 : p.robot_type) ?? "?";
  const cams = Object.keys(peer.cameras ?? {});
  const offline = !!peer.stale;
  const telemetry = useTelemetry(peer);
  const twin = twinButtonCopy({ peerId: peer.peer_id, twinLive, busy: twinBusy });
  const status = type === "robot" ? statusSentence(peerStatusFields(peer, telemetry, hostsChildren)) : null;
  reactExports.useEffect(() => {
    onBusyChange == null ? void 0 : onBusyChange(peer.peer_id, running);
  }, [running, peer.peer_id]);
  const { badge: whyMute } = useJointFailure(peer.peer_id, Object.keys(((_a = peer.state) == null ? void 0 : _a.joints) ?? {}).length === 0);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `card${offline ? " stale" : ""}${phase === "failed" ? " failed" : ""}${running ? " running" : ""}`, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `typebadge ${type}`, children: type }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "peername", title: `open ${peer.peer_id}`, onClick: () => onOpen == null ? void 0 : onOpen(peer.peer_id), children: peer.peer_id }),
      peer.role && /* @__PURE__ */ jsxRuntimeExports.jsx(
        "span",
        {
          className: `rolebadge ${peer.role}`,
          title: peer.role_volts ? `measured ${peer.role_volts}V on its servo bus` : "measured off its servo bus",
          children: peer.role
        }
      ),
      (() => {
        const lb = lockoutBadge(peer.lockout);
        return lb.label ? /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `lockbadge ${lb.tone}`, title: lb.title, children: lb.label }) : null;
      })(),
      type === "robot" && peer.origin === "external" && /* @__PURE__ */ jsxRuntimeExports.jsx(
        "span",
        {
          className: "originbadge",
          title: "started outside this dashboard (your own script, or another machine).\nEverything on this card works normally. Only the three things that need a local\nchild process are unavailable: logs, camera reconfigure and despawn.",
          children: "external"
        }
      ),
      (() => {
        var _a2;
        const bus = busRecoveryBadge((_a2 = peer.state) == null ? void 0 : _a2.bus_recoveries);
        return bus ? /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge ${bus.tone}`, title: bus.title, children: bus.label }) : null;
      })(),
      (p == null ? void 0 : p.hostname) && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "host", children: p.hostname }),
      (p == null ? void 0 : p.connected) === false && type === "robot" && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge warn", title: "peer is online but its hardware is not connected", children: "hw off" }),
      type === "robot" && !peer.peer_id.includes("__") && !peer.peer_id.endsWith("-twin") && /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: `twinbtn${twin.cls ? ` ${twin.cls}` : ""}`,
          onClick: toggleTwin,
          disabled: twinBusy,
          title: twin.title,
          "aria-label": twin.aria,
          "aria-pressed": twin.pressed,
          children: twin.label
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "span",
        {
          className: offline ? "dot off" : running ? "dot busy" : "dot on",
          role: "img",
          "aria-label": offline ? "no heartbeat for 15s" : running ? "task running" : "idle",
          title: offline ? "no heartbeat for 15s" : running ? "task running" : "idle"
        }
      )
    ] }),
    whyMute && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small warn", role: "status", children: [
      "no joints — ",
      whyMute
    ] }),
    status && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `status-ribbon ${status.severity}`, role: "status", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { className: "status-word", children: status.word }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: ribbonDetail(status) })
    ] }),
    offline && !status && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "stale-note", children: [
      "no heartbeat — last seen",
      " ",
      peer.last_seen ? `${Math.round(Date.now() / 1e3 - peer.last_seen)}s ago` : "unknown"
    ] }),
    (() => {
      const note = deadCameraNote(stoppedCameras(peer.cameras ?? {}, Date.now() / 1e3), cams.length);
      return note ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "cam-dead-note", role: "status", children: note }) : null;
    })(),
    (() => {
      const canConfig = type === "robot" && peer.origin !== "external";
      return /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        cams.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: cams.length > 1 ? "cams multi" : "cams", children: cams.slice(0, 4).map((c) => {
          var _a2;
          return /* @__PURE__ */ jsxRuntimeExports.jsx(
            CameraTile,
            {
              peerId: peer.peer_id,
              cam: c,
              meta: (_a2 = peer.cameras) == null ? void 0 : _a2[c],
              onConfigure: canConfig ? () => setCamSheet({ cam: c, add: false }) : void 0
            },
            c
          );
        }) }),
        canConfig && /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "chip addcam",
            onClick: () => setCamSheet({ cam: null, add: true }),
            title: "attach another camera to this robot (applying restarts it)",
            children: "+ add camera"
          }
        ),
        camSheet && /* @__PURE__ */ jsxRuntimeExports.jsx(
          CameraConfigSheet,
          {
            peerId: peer.peer_id,
            focusCam: camSheet.cam,
            startAdding: camSheet.add,
            onClose: () => setCamSheet(null)
          }
        )
      ] });
    })(),
    /* @__PURE__ */ jsxRuntimeExports.jsx(JointStrip, { state: peer.state, presence: p, problem: peer.joint_problem, peerStale: peer.stale }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(TelemetryStrip, { peer }),
    peer.stream && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "streamline", children: [
      "step ",
      peer.stream.step,
      " · ",
      peer.stream.policy || "policy",
      " · ",
      (_b = peer.stream.instruction) == null ? void 0 : _b.slice(0, 40)
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(
      RunForm,
      {
        peerId: peer.peer_id,
        presence: p,
        running,
        busy,
        disabled: offline,
        onRun: run,
        onStop: stop
      }
    ),
    outcome && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: outcome.ok ? "result ok" : "result bad", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        outcome.ok ? "✓" : outcome.ambiguous ? "⚠ unknown —" : "✗",
        " ",
        outcome.text
      ] }),
      outcome.detail && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "details" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: outcome.detail })
      ] }),
      consent && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn small", onClick: () => setSheet(true), children: "review permission…" })
    ] }),
    consent && sheet && /* @__PURE__ */ jsxRuntimeExports.jsx(
      ConsentSheet,
      {
        need: consent,
        target: "peer",
        onCancel: () => {
          setSheet(false);
          clearConsent();
        },
        onRetry: () => {
          setSheet(false);
          void retryLast();
        }
      }
    )
  ] });
}
function isChildOf(child, parent) {
  return !!parent && !!child && child.startsWith(`${parent}__`) && child.length > parent.length + 2;
}
function armHosts(peers) {
  const list = (peers ?? []).filter((p) => p && p.peer_id);
  const out = {};
  for (const p of list) {
    const children = list.map((x) => x.peer_id).filter((id) => isChildOf(id, p.peer_id));
    if (!children.length) continue;
    if (typeof p.joints === "number" && p.joints > 0) continue;
    out[p.peer_id] = {
      children,
      why: children.length === 1 ? `hosts ${children[0]} — pick that, this is the process` : `hosts ${children.length} robots (${children.join(", ")}) — pick one of those, this is the process`
    };
  }
  return out;
}
function cameraEvidence(peerId, announced, arrived, requested) {
  const frames = (arrived ?? []).filter(Boolean);
  if (frames.length > 0) return { kind: "ok", cams: frames };
  const names = (announced ?? []).filter(Boolean);
  if (names.length > 0) {
    const list = names.join(", ");
    return {
      kind: "mute",
      announced: names,
      message: `${peerId} announces ${names.length} camera${names.length > 1 ? "s" : ""} (${list}) but no frames have arrived — recording now would capture joints only. A camera that is blocked by macOS, held by another process or unplugged looks identical from here: open devices › logs for ${peerId} to see which.`
    };
  }
  const asked = (requested ?? []).filter(Boolean);
  if (asked.length > 0) {
    const list = asked.join(", ");
    return {
      kind: "dropped",
      requested: asked,
      message: `${peerId} was started with ${list} but announces no cameras — they were dropped when it connected, which means the camera could not be opened: blocked by macOS privacy, held by another process, or unplugged. Recording now would capture joints only. ${peerId}'s log (devices › logs) names the one that failed.`
    };
  }
  return {
    kind: "unannounced",
    message: `${peerId} lists no cameras — recording now would capture joints only. That is either a deliberately joints-only robot or cameras that failed to open and were dropped when it connected; from here the two are indistinguishable, and ${peerId}'s log says which.`
  };
}
function cameraPlaceholder(ev) {
  if (ev.kind === "ok") return null;
  if (ev.kind === "dropped") {
    return {
      head: "cameras dropped",
      sub: `${ev.requested.join(", ")} requested, none opened`,
      title: ev.message
    };
  }
  if (ev.kind === "mute") {
    return {
      head: "no frames",
      sub: `${ev.announced.join(", ")} announced, nothing arriving`,
      title: ev.message
    };
  }
  return {
    head: "no camera",
    sub: "none listed — joints-only, or dropped at connect",
    title: ev.message
  };
}
const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
function looksLikeClose(c) {
  const words2 = `${c.label ?? ""} ${c.text ?? ""}`.trim().toLowerCase();
  if (/(^|\s)(close|dismiss)\b/.test(words2)) return true;
  return /^[\u00d7\u2715\u2716\u274c\u2573x]$/.test((c.text ?? "").trim());
}
function looksDangerous(c) {
  const words2 = `${c.label ?? ""} ${c.text ?? ""}`.toLowerCase();
  return /\b(run|start|record|recording|stop|e-?stop|resume|delete|remove|despawn|deploy|train|calibrate|home|move|teleop|replay)\b/.test(words2);
}
function focusPlan(candidates) {
  const autofocus = candidates.findIndex((c) => c.autofocus);
  if (autofocus >= 0) return autofocus;
  const close = candidates.findIndex(looksLikeClose);
  if (close >= 0) return close;
  const safe = candidates.findIndex((c) => !looksDangerous(c));
  if (safe >= 0) return safe;
  return "container";
}
function rememberOpener(active, body) {
  if (!active || active === body) return null;
  return { el: active };
}
function shouldRestoreFocus(s) {
  if (!s.openerConnected) return false;
  return s.activeInsideOverlay || s.activeIsBody;
}
function useDialogFocus(ref, open = true) {
  const opener = reactExports.useRef(null);
  reactExports.useEffect(() => {
    if (!open) return;
    const remembered = rememberOpener(document.activeElement, document.body);
    opener.current = (remembered == null ? void 0 : remembered.el) instanceof HTMLElement ? remembered.el : null;
    const id = requestAnimationFrame(() => {
      var _a;
      const node = ref.current;
      if (!node || node.contains(document.activeElement)) return;
      const els = [...node.querySelectorAll(FOCUSABLE)];
      const plan = focusPlan(els.map((el) => ({
        autofocus: el.hasAttribute("data-autofocus"),
        label: el.getAttribute("aria-label") ?? el.getAttribute("title") ?? "",
        text: el.textContent ?? ""
      })));
      if (plan === "container") {
        if (!node.hasAttribute("tabindex")) node.setAttribute("tabindex", "-1");
        node.focus();
        return;
      }
      (_a = els[plan]) == null ? void 0 : _a.focus();
    });
    return () => {
      var _a, _b, _c;
      cancelAnimationFrame(id);
      const active = document.activeElement;
      if (shouldRestoreFocus({
        activeInsideOverlay: ((_a = ref.current) == null ? void 0 : _a.contains(active)) ?? false,
        activeIsBody: active === document.body,
        openerConnected: ((_b = opener.current) == null ? void 0 : _b.isConnected) ?? false
      })) (_c = opener.current) == null ? void 0 : _c.focus();
    };
  }, [ref, open]);
}
const TONE = {
  refusing: "warn",
  unrouted: "warn",
  silent: "warn",
  stopped: "idle",
  following: "ok"
};
function teleopView(payload) {
  const health = payload == null ? void 0 : payload.health;
  if (!health || typeof health !== "object") return null;
  const h = health;
  const receivers = h.receivers ?? {};
  const publishers = h.publishers ?? {};
  const pubs = Object.entries(publishers);
  const live = pubs.filter(([, p]) => p.state === "publishing");
  const worst = h.worst ?? Object.values(receivers)[0] ?? null;
  if (!worst && pubs.length === 0) {
    return {
      tone: "idle",
      headline: "no teleop on this arm",
      streaming: false,
      detail: "it is neither following another arm nor publishing its own joints"
    };
  }
  if (!worst) {
    const [name, p] = live[0] ?? pubs[0];
    return {
      tone: live.length ? "ok" : "idle",
      headline: `publishing ${name}: ${p.headline ?? p.state ?? "unknown"}`,
      detail: p.detail ?? null,
      streaming: live.length > 0
    };
  }
  const view = {
    tone: TONE[worst.state ?? ""] ?? "warn",
    headline: worst.headline ?? worst.state ?? "teleop state unknown",
    detail: worst.detail ?? null,
    // A receiver that is following counts as traffic too: frames are being applied to a real arm.
    streaming: live.length > 0 || worst.state === "following" || worst.state === "refusing"
  };
  if (worst.refusal) view.consentKind = "teleop_degree_units";
  return view;
}
function stopVerdict(after) {
  if (!after) {
    return { ok: false, line: "stop was sent, but the arm did not answer when asked again — nothing confirms it landed" };
  }
  if (after.streaming) {
    return { ok: false, line: `stop was sent, but frames are STILL on the wire: ${after.headline}` };
  }
  return { ok: true, line: `teleop stopped — ${after.headline}` };
}
function startVerdict(after) {
  if (!after) {
    return { ok: false, line: "start was sent, but the arm did not answer when asked again — nothing confirms frames are flowing" };
  }
  if (after.consentKind) {
    return { ok: false, line: `started, but every frame is being REFUSED: ${after.headline} — the bound is widened at settings › consent › ${after.consentKind}, deliberately and by you` };
  }
  if (after.streaming) return { ok: true, line: `teleop live — ${after.headline}` };
  return { ok: false, line: "start was sent, but nothing is on the wire yet — a follower can take up to 45s to declare its subscriber, so ask again before assuming it failed" };
}
const isSim$1 = (p) => (p == null ? void 0 : p.robot_type) === "sim";
function teleopSubject(peerId, peers) {
  const list = (peers ?? []).filter((p) => p && p.peer_id);
  const host = armHosts(list)[peerId];
  if (!host) return null;
  return {
    children: host.children,
    why: host.children.length === 1 ? `this is the process, not the arm — the robot inside it is ${host.children[0]}` : `this is the process, not the arm — it hosts ${host.children.join(", ")}`
  };
}
const jointCount$2 = (p) => typeof (p == null ? void 0 : p.joints) === "number" && p.joints > 0 ? p.joints : 0;
function leaderOptions(followerId, peers) {
  const list = (peers ?? []).filter((p) => p && p.peer_id);
  const hosts = armHosts(list);
  const out = [];
  for (const p of list) {
    if (p.peer_id === followerId) continue;
    const host = hosts[p.peer_id];
    if (host) {
      out.push({ peer_id: p.peer_id, ok: false, why: host.why });
      continue;
    }
    if (!jointCount$2(p)) {
      out.push({
        peer_id: p.peer_id,
        ok: false,
        why: "reports no joints — it cannot publish a position it cannot read; its log (devices › logs) says why"
      });
      continue;
    }
    if (isChildOf(p.peer_id, followerId) || isChildOf(followerId, p.peer_id)) {
      out.push({ peer_id: p.peer_id, ok: false, why: "the same robot as the follower, under its process name" });
      continue;
    }
    if (isSim$1(p)) {
      out.push({
        peer_id: p.peer_id,
        ok: true,
        why: "simulated — nothing to hand-guide; its joints move when a task or replay drives it"
      });
      continue;
    }
    out.push({
      peer_id: p.peer_id,
      ok: true,
      why: p.role === "leader" && p.role_source === "measured" ? `measured as a leader (${p.role_volts ?? "?"}V)` : p.role === "follower" && p.role_source === "measured" ? `measured as a FOLLOWER (${p.role_volts ?? "?"}V) — check this is the arm you intend to hand-guide` : "role not measured — the dashboard cannot tell which arm this is wired as"
    });
  }
  return out;
}
function pairPlan(followerId, leaderId, peers) {
  if (!followerId || !leaderId || followerId === leaderId) return null;
  const list = (peers ?? []).filter((p) => p && p.peer_id);
  const follower = list.find((p) => p.peer_id === followerId);
  const leader = list.find((p) => p.peer_id === leaderId);
  const blockers = [];
  const notes = [];
  if (!follower) blockers.push(`${followerId} is not on the mesh — the dashboard cannot command an arm it cannot see`);
  if (!leader) blockers.push(`${leaderId} is not on the mesh`);
  if (follower && !jointCount$2(follower))
    blockers.push(`${followerId} reports no joints, so nothing could be applied to it (its log says why)`);
  if (leader && !jointCount$2(leader))
    blockers.push(`${leaderId} reports no joints, so it has no position to publish (its log says why)`);
  const hosts = armHosts(list);
  for (const id of [followerId, leaderId]) if (hosts[id]) blockers.push(`${id} ${hosts[id].why}`);
  const fj = jointCount$2(follower), lj = jointCount$2(leader);
  if (fj && lj && fj !== lj)
    notes.push(`${leaderId} reports ${lj} joints and ${followerId} reports ${fj} — only the names they share can be followed`);
  if ((follower == null ? void 0 : follower.role) === "leader" && follower.role_source === "measured")
    notes.push(`${followerId} measures as a LEADER (${follower.role_volts ?? "?"}V) — it is about to be DRIVEN, so make sure that is the arm you want moving`);
  const physical = !isSim$1(follower);
  if (!physical) notes.push(`${followerId} is simulated — nothing physical moves when it follows`);
  return {
    leader: leaderId,
    follower: followerId,
    blockers,
    notes,
    physical,
    // Grants asked for BEFORE anything is sent. agent_physical_motion only when frames move
    // metal; teleop_degree_units always, because an SO-101 leader publishes degrees into a
    // radian envelope (sim or real) and every frame would otherwise be refused — the failure
    // that took a log dive to find the first time.
    consents: physical ? ["agent_physical_motion", "teleop_degree_units"] : ["teleop_degree_units"]
  };
}
const isSim = (p) => (p == null ? void 0 : p.robot_type) === "sim";
const jointCount$1 = (p) => typeof (p == null ? void 0 : p.joints) === "number" && p.joints > 0 ? p.joints : 0;
function twinFollowerOf(peerId, peers) {
  const list = (peers ?? []).filter((p) => p && p.peer_id);
  const twinId = `${peerId}-twin`;
  const process = list.find((p) => p.peer_id === twinId) ?? null;
  const arm = list.find((p) => isChildOf(p.peer_id, twinId) && jointCount$1(p) > 0) ?? null;
  return { process, arm };
}
function mirrorPlan(peerId, peers) {
  const list = (peers ?? []).filter((p) => p && p.peer_id);
  const subject = list.find((p) => p.peer_id === peerId);
  const blockers = [];
  const notes = [];
  if (!subject) {
    return { follower: null, blockers: [`${peerId} is not on the mesh`], notes };
  }
  if (isSim(subject)) {
    return { follower: null, blockers: ["this peer is already a simulation — the mirror follows a REAL arm"], notes };
  }
  if (!jointCount$1(subject)) {
    blockers.push("this arm reports no joints, so it has no position to publish (its log — devices › logs — says why)");
  }
  const { process, arm } = twinFollowerOf(peerId, list);
  if (!process) {
    blockers.push("no sim twin on the mesh — spawn it first (the twin button on this card), then mirror");
  } else if (!arm) {
    blockers.push(`${process.peer_id} is up but no robot under it reports joints yet — it may still be loading; ask again in a few seconds`);
  }
  if (subject.role === "leader" && subject.role_source === "measured") {
    notes.push(`measured as a leader (${subject.role_volts ?? "?"}V) — hand-movable by design`);
  } else if (subject.role === "follower" && subject.role_source === "measured") {
    notes.push(
      `measured as a FOLLOWER (${subject.role_volts ?? "?"}V) — its motors may hold torque, so it will resist being hand-moved; relaxing torque is a physical change made at the arm, not from this screen`
    );
  } else {
    notes.push("role not measured — if the arm resists being hand-moved, its motors hold torque; relaxing it is done at the arm, not from this screen");
  }
  const fj = jointCount$1(arm ?? void 0);
  const lj = jointCount$1(subject);
  if (arm && fj && lj && fj !== lj) {
    notes.push(`this arm reports ${lj} joints and ${arm.peer_id} reports ${fj} — only the names they share can be mirrored`);
  }
  return { follower: blockers.length ? null : (arm == null ? void 0 : arm.peer_id) ?? null, blockers, notes };
}
function mirrorSentence(plan) {
  if (plan.blockers.length) return `cannot mirror: ${plan.blockers.join("; ")}`;
  return `hand-move this arm and ${plan.follower} copies it, joint for joint — nothing physical moves`;
}
const DISK_CRITICAL_GB = 2;
const DISK_TIGHT_GB = 12;
const BATTERY_LOW_PCT = 10;
function num(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function uptimeText(seconds) {
  const s = num(seconds);
  if (s === null || s < 0) return null;
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}
function hottest(temps) {
  if (!temps || typeof temps !== "object") return null;
  let best = null;
  for (const [name, raw] of Object.entries(temps)) {
    const c = num(raw);
    if (c === null) continue;
    if (best === null || c > best[1]) best = [name, c];
  }
  return best;
}
function healthLine(health) {
  if (health == null) return { text: "no health topic on this robot", tone: "none", detail: null };
  const battery = num(health.battery_pct);
  const charging = health.charging === true;
  const disk = num(health.disk_free_gb);
  const mem = num(health.mem_pct);
  const load2 = num(health.cpu_load);
  const hot = hottest(health.temps);
  const up = uptimeText(health.uptime_s);
  const extras = [];
  if (battery !== null) extras.push(`battery ${battery.toFixed(0)}%${charging ? " (charging)" : ""}`);
  if (disk !== null) extras.push(`${disk.toFixed(1)} GB free`);
  if (mem !== null) extras.push(`memory ${mem.toFixed(0)}%`);
  if (load2 !== null) extras.push(`load ${load2.toFixed(2)}`);
  if (hot) extras.push(`${hot[0]} ${hot[1].toFixed(0)}C`);
  if (up) extras.push(`up ${up}`);
  const detail = extras.length > 0 ? extras.join(" · ") : null;
  if (detail === null) {
    return { text: "health topic is arriving, but reported no readings", tone: "none", detail: null };
  }
  if (battery !== null && !charging && battery <= BATTERY_LOW_PCT) {
    return { text: `battery ${battery.toFixed(0)}% and discharging`, tone: "attention", detail };
  }
  if (disk !== null && disk < DISK_CRITICAL_GB) {
    return {
      text: `only ${disk.toFixed(1)} GB free - not enough to finish a recording`,
      tone: "attention",
      detail
    };
  }
  if (disk !== null && disk < DISK_TIGHT_GB) {
    return {
      text: `${disk.toFixed(1)} GB free - tight for a long session`,
      tone: "attention",
      detail
    };
  }
  if (battery !== null) {
    return { text: `battery ${battery.toFixed(0)}%${charging ? ", charging" : ""}`, tone: "ok", detail };
  }
  return { text: extras[0], tone: "ok", detail };
}
const SENSOR_QUIET_S = 10;
const SENSOR_KINDS = ["health", "pose", "odom", "imu", "lidar"];
function declaredKinds(topics) {
  if (!Array.isArray(topics)) return [];
  return SENSOR_KINDS.filter((k) => topics.includes(k));
}
function sensorVerdict(kind, reading, nowS, declared = false) {
  if (reading == null) {
    if (declared) {
      return { kind, tone: "waiting", text: "declared, waiting for the first reading", ageS: null };
    }
    return { kind, tone: "absent", text: "not published by this robot", ageS: null };
  }
  const t = reading.t;
  if (typeof t !== "number" || !Number.isFinite(t) || t <= 0) {
    return { kind, tone: "live", text: "arriving (no timestamp in the payload)", ageS: null };
  }
  const ageS2 = nowS - t;
  if (ageS2 < 0) return { kind, tone: "live", text: "arriving", ageS: 0 };
  if (ageS2 <= SENSOR_QUIET_S) return { kind, tone: "live", text: "arriving", ageS: ageS2 };
  return { kind, tone: "stale", text: `last reading ${agoText(ageS2)}`, ageS: ageS2 };
}
function rowsToShow(topics, sensors) {
  const declared = new Set(declaredKinds(topics));
  return SENSOR_KINDS.filter((k) => declared.has(k) || (sensors == null ? void 0 : sensors[k]) != null);
}
function stripSummary(verdicts) {
  const shown = verdicts.filter((v) => v.tone !== "absent");
  if (shown.length === 0) return null;
  const stale = shown.filter((v) => v.tone === "stale");
  if (stale.length > 0) {
    return { text: `${stale.map((v) => v.kind).join(", ")} went quiet`, tone: "stale" };
  }
  const live = shown.filter((v) => v.tone === "live");
  if (live.length === 0) {
    return { text: `${shown.map((v) => v.kind).join(", ")} declared, nothing yet`, tone: "waiting" };
  }
  const waiting = shown.length - live.length;
  const tail = waiting > 0 ? `, ${waiting} not started` : "";
  return { text: `${live.length} sensor${live.length === 1 ? "" : "s"} arriving${tail}`, tone: "live" };
}
function chipValue(kind, r) {
  const n = (v) => typeof v === "number" && Number.isFinite(v) ? v : null;
  if (kind === "pose") {
    const x = n(r.x), y = n(r.y);
    return x !== null && y !== null ? `${x.toFixed(2)}, ${y.toFixed(2)}` : null;
  }
  if (kind === "odom") {
    const vx = n(r.vx), wz = n(r.wz);
    if (vx === null && wz === null) return null;
    return `${vx !== null ? `${vx.toFixed(2)} m/s` : ""}${vx !== null && wz !== null ? " " : ""}${wz !== null ? `${wz.toFixed(2)} rad/s` : ""}`;
  }
  if (kind === "imu") {
    const g = Array.isArray(r.gyro) ? r.gyro.map(n).filter((v) => v !== null) : [];
    return g.length > 0 ? `gyro ${Math.max(...g.map(Math.abs)).toFixed(3)}` : null;
  }
  if (kind === "lidar") {
    const min = n(r.min_range);
    return min !== null ? `min ${min.toFixed(2)} m` : null;
  }
  return null;
}
function SensorStrip({ peer, nowS }) {
  var _a;
  const now = nowS ?? Date.now() / 1e3;
  const lidarReading = (() => {
    var _a2, _b;
    const s = (_a2 = peer.lidar) == null ? void 0 : _a2.summary, st = (_b = peer.lidar) == null ? void 0 : _b.state;
    if (!s && !st) return null;
    const ts = (r) => typeof (r == null ? void 0 : r.t) === "number" ? r.t : -Infinity;
    return ts(s) >= ts(st) ? s : st;
  })();
  const readings = {
    health: peer.health ?? null,
    pose: peer.pose ?? null,
    odom: peer.odom ?? null,
    imu: peer.imu ?? null,
    lidar: lidarReading
  };
  const topics = (_a = peer.presence) == null ? void 0 : _a.topics;
  const declared = new Set(declaredKinds(topics));
  const verdicts = SENSOR_KINDS.map((k) => sensorVerdict(k, readings[k], now, declared.has(k)));
  const summary = stripSummary(verdicts);
  if (summary === null) return null;
  const shown = rowsToShow(topics, readings);
  const health = healthLine(readings.health);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sensorstrip", children: [
    readings.health && /* @__PURE__ */ jsxRuntimeExports.jsx(
      "p",
      {
        className: `hint${health.tone === "attention" ? " warn" : health.tone === "ok" ? " ok" : ""}`,
        role: "status",
        title: health.detail ?? void 0,
        children: health.text
      }
    ),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "telemetry", children: shown.map((kind) => {
      const v = verdicts.find((x) => x.kind === kind);
      const reading = readings[kind];
      const value = reading ? chipValue(kind, reading) : null;
      return /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "span",
        {
          className: `metric${v.tone === "stale" ? " warn" : ""}`,
          title: `${kind}: ${v.text}`,
          children: [
            kind,
            value ? ` ${value}` : "",
            v.tone === "stale" && v.ageS !== null ? ` (${v.ageS.toFixed(0)}s old)` : "",
            v.tone === "waiting" ? " —" : ""
          ]
        },
        kind
      );
    }) })
  ] });
}
const HISTORY = 40;
function fmt(v) {
  if (typeof v === "number") return v.toFixed(3);
  if (Array.isArray(v)) return `[${v.map((x) => typeof x === "number" ? x.toFixed(2) : String(x)).join(", ")}]`;
  if (v && typeof v === "object") return JSON.stringify(v);
  return String(v);
}
function RobotDetail({ peer, twinLive = false, hostsChildren, fleet, onOpen, onClose }) {
  var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin } = useTask(peer);
  const cams = Object.keys(peer.cameras ?? {});
  const twin = twinButtonCopy({ peerId: peer.peer_id, twinLive, busy: twinBusy });
  const [cam, setCam] = reactExports.useState(null);
  const [teleop, setTeleop] = reactExports.useState(null);
  const askTeleop = async () => {
    setTeleop("asking");
    try {
      setTeleop(teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`)));
    } catch {
      setTeleop("unreachable");
    }
  };
  const [stopArmed, setStopArmed] = reactExports.useState(false);
  const [stopped, setStopped] = reactExports.useState(null);
  const [startArmed, setStartArmed] = reactExports.useState(null);
  const [started, setStarted] = reactExports.useState(null);
  const startTeleop = async (leaderId) => {
    setStartArmed(null);
    setStarted(null);
    setStopped(null);
    setTeleop("asking");
    try {
      await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop/publish`, { method: "POST", body: JSON.stringify({}) });
    } catch (e) {
      setTeleop("unreachable");
      setStarted({ ok: false, line: `${leaderId} could not start publishing, so nothing was pointed at it: ${e.message}` });
      return;
    }
    try {
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/receive`, { method: "POST", body: JSON.stringify({ source_peer_id: leaderId, confirmed: true }) });
    } catch (e) {
      setTeleop("unreachable");
      setStranded(leaderId);
      setStarted({ ok: false, line: `${leaderId} is publishing its joints, but ${peer.peer_id} would not follow it: ${e.message} — nothing is moving, and ${leaderId} is still on the wire until you stop it below` });
      return;
    }
    let after = null;
    try {
      after = teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`));
    } catch {
      after = null;
    }
    setTeleop(after ?? "unreachable");
    setStarted(startVerdict(after));
  };
  const [stranded, setStranded] = reactExports.useState(null);
  const [strandedResult, setStrandedResult] = reactExports.useState(null);
  const stopStranded = async (leaderId) => {
    setStrandedResult(null);
    try {
      await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop/stop`, { method: "POST" });
    } catch (e) {
      setStrandedResult({ ok: false, line: `${leaderId} would not stop publishing: ${e.message} — it is still on the wire` });
      return;
    }
    let after = null;
    try {
      after = teleopView(await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop`));
    } catch {
      after = null;
    }
    const v = stopVerdict(after);
    setStrandedResult({ ok: v.ok, line: `${leaderId}: ${v.line}` });
  };
  const stopTeleop = async () => {
    setStopArmed(false);
    setStopped(null);
    setTeleop("asking");
    try {
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/stop`, { method: "POST" });
    } catch (e) {
      setTeleop("unreachable");
      setStopped({ ok: false, line: `stop was refused: ${e.message}` });
      return;
    }
    let after = null;
    try {
      after = teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`));
    } catch {
      after = null;
    }
    setTeleop(after ?? "unreachable");
    setStopped(stopVerdict(after));
  };
  const mirror = ((_a = peer.presence) == null ? void 0 : _a.robot_type) === "robot" && ((fleet == null ? void 0 : fleet.length) ?? 0) > 0 ? mirrorPlan(peer.peer_id, fleet) : null;
  const [mirrorArmed, setMirrorArmed] = reactExports.useState(false);
  const [mirrorBusy, setMirrorBusy] = reactExports.useState(false);
  const [mirrorOn, setMirrorOn] = reactExports.useState(null);
  const [mirrorLine, setMirrorLine] = reactExports.useState(null);
  const askMirror = async (followerId) => {
    let after = null;
    try {
      after = teleopView(await api(`/api/robots/${encodeURIComponent(followerId)}/teleop`));
    } catch {
      after = null;
    }
    return after;
  };
  const startMirror = async (followerId) => {
    setMirrorArmed(false);
    setMirrorBusy(true);
    setMirrorLine(null);
    try {
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/publish`, { method: "POST", body: JSON.stringify({}) });
    } catch (e) {
      setMirrorBusy(false);
      setMirrorLine({ ok: false, line: `${peer.peer_id} could not start publishing its joints, so nothing was mirrored: ${e.message}` });
      return;
    }
    try {
      await api(`/api/robots/${encodeURIComponent(followerId)}/teleop/receive`, { method: "POST", body: JSON.stringify({ source_peer_id: peer.peer_id, confirmed: true }) });
    } catch (e) {
      setMirrorBusy(false);
      setMirrorOn(followerId);
      setMirrorLine({ ok: false, line: `${peer.peer_id} is publishing, but ${followerId} would not follow: ${e.message} — stop below removes the publisher` });
      return;
    }
    const after = await askMirror(followerId);
    setMirrorBusy(false);
    setMirrorOn(followerId);
    const v = startVerdict(after);
    setMirrorLine({ ok: v.ok, line: v.ok ? `mirror live — ${followerId} is copying this arm's joints` : v.line });
  };
  const refreshMirror = async (followerId) => {
    setMirrorBusy(true);
    const after = await askMirror(followerId);
    setMirrorBusy(false);
    if (!after) {
      setMirrorLine({ ok: false, line: `${followerId} did not answer when asked — nothing confirms the mirror state` });
      return;
    }
    setMirrorLine({ ok: after.streaming && after.tone === "ok", line: `${followerId}: ${after.headline}` });
  };
  const stopMirror = async (followerId) => {
    setMirrorBusy(true);
    setMirrorLine(null);
    let failed = "";
    try {
      await api(`/api/robots/${encodeURIComponent(followerId)}/teleop/stop`, { method: "POST" });
    } catch (e) {
      failed = `${followerId} refused stop: ${e.message}`;
    }
    try {
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/stop`, { method: "POST" });
    } catch (e) {
      failed = failed ? `${failed}; ` : "";
      failed += `${peer.peer_id} refused stop: ${e.message}`;
    }
    const after = await askMirror(followerId);
    setMirrorBusy(false);
    const v = stopVerdict(after);
    if (v.ok && !failed) setMirrorOn(null);
    setMirrorLine(failed ? { ok: false, line: failed } : { ok: v.ok, line: v.ok ? "mirror stopped — nothing is being copied" : v.line });
  };
  const sheetRef = reactExports.useRef(null);
  useDialogFocus(sheetRef);
  const [camConfig, setCamConfig] = reactExports.useState(null);
  const canConfig = ((_b = peer.presence) == null ? void 0 : _b.robot_type) === "robot" && peer.origin !== "external";
  const [steps, setSteps] = reactExports.useState([]);
  const lastStep = reactExports.useRef(null);
  const active = cam && cams.includes(cam) ? cam : cams[0] ?? null;
  reactExports.useEffect(() => {
    const s = peer.stream;
    if (!s || s.step === lastStep.current) return;
    lastStep.current = s.step;
    setSteps((prev) => [s, ...prev].slice(0, HISTORY));
  }, [peer.stream]);
  reactExports.useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const p = peer.presence;
  const offline = !!peer.stale;
  const joints = Object.entries(((_c = peer.state) == null ? void 0 : _c.joints) ?? {});
  const { line: whyNoJoints } = useJointFailure(peer.peer_id, joints.length === 0);
  const telemetry = useTelemetry(peer);
  const status = ((p == null ? void 0 : p.robot_type) ?? "?") === "robot" ? statusSentence(peerStatusFields(peer, telemetry, hostsChildren)) : null;
  const stepHz = reactExports.useMemo(() => {
    if (steps.length < 2) return 0;
    const span = steps[0].t - steps[steps.length - 1].t;
    return span > 0 ? (steps.length - 1) / span : 0;
  }, [steps]);
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "detail-backdrop", onClick: onClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "section",
    {
      ref: sheetRef,
      className: `detail${offline ? " stale" : ""}`,
      role: "dialog",
      "aria-label": `Robot ${peer.peer_id}`,
      onClick: (e) => e.stopPropagation(),
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("header", { className: "detail-head", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `typebadge ${(p == null ? void 0 : p.robot_type) ?? "?"}`, children: (p == null ? void 0 : p.robot_type) ?? "?" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: peer.peer_id }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "span",
            {
              className: offline ? "dot off" : running ? "dot busy" : "dot on",
              role: "img",
              "aria-label": offline ? "no heartbeat for 15s" : running ? "task running" : "idle",
              title: offline ? "no heartbeat for 15s" : running ? "task running" : "idle"
            }
          ),
          (p == null ? void 0 : p.hostname) && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "host", children: p.hostname }),
          (p == null ? void 0 : p.robot_type) === "robot" && !peer.peer_id.endsWith("-twin") && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: `twinbtn${twin.cls ? ` ${twin.cls}` : ""}`,
              onClick: toggleTwin,
              disabled: twinBusy,
              title: twin.title,
              "aria-label": twin.aria,
              "aria-pressed": twin.pressed,
              children: twin.label
            }
          ),
          (p == null ? void 0 : p.robot_type) === "robot" && peer.origin === "external" && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "span",
            {
              className: "originbadge",
              title: "started outside this dashboard (your own script, or another machine).\nEverything here works normally except the three things that need a local\nchild process: logs, camera reconfigure and despawn.",
              children: "external"
            }
          ),
          (p == null ? void 0 : p.robot_type) === "robot" && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost",
              onClick: () => setCamConfig({ cam: null, add: false }),
              disabled: peer.origin === "external",
              title: peer.origin === "external" ? "this robot was started outside the dashboard, so it has no local process to restart — change its cameras where it was launched (its own script or machine)" : "attach / detach cameras, change fps and resolution (restarts the robot)",
              children: "cameras"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost",
              onClick: askTeleop,
              disabled: teleop === "asking",
              title: "is this arm following another arm, or publishing its own joints? (reads only)",
              children: teleop === "asking" ? "asking…" : "teleop"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, "aria-label": "close this robot", title: "Escape", children: "✕" })
        ] }),
        teleop === "unreachable" && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint warn", role: "status", children: "could not ask this arm about teleop — it may be busy or gone; its own log (devices › logs) is where a refusal appears" }),
        teleop && typeof teleop === "object" && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `hint ${teleop.tone === "warn" ? "warn" : ""}`, role: "status", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "teleop:" }),
          " ",
          teleop.headline,
          teleop.streaming && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "muted small", children: " · frames are on the wire" }),
          teleop.detail && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "muted small", children: teleop.detail }),
          teleop.streaming && (stopArmed ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row small", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn danger", onClick: stopTeleop, children: [
              "confirm — stop teleop on ",
              peer.peer_id
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setStopArmed(false), children: "keep it running" })
          ] }) : /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost small",
              onClick: () => setStopArmed(true),
              title: "stop the teleop stream on this arm — it only removes commands, it cannot move anything",
              children: "stop teleop"
            }
          )),
          !teleop.streaming && ((fleet == null ? void 0 : fleet.length) ?? 0) > 0 && (() => {
            const subject = teleopSubject(peer.peer_id, fleet);
            if (subject) {
              return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small muted", children: [
                subject.why,
                /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row small", children: subject.children.map((c) => /* @__PURE__ */ jsxRuntimeExports.jsxs(
                  "button",
                  {
                    className: "btn ghost small",
                    onClick: () => onOpen == null ? void 0 : onOpen(c),
                    disabled: !onOpen,
                    title: `open ${c} — teleop starts from the arm itself`,
                    children: [
                      "open ",
                      c
                    ]
                  },
                  c
                )) })
              ] });
            }
            const opts = leaderOptions(peer.peer_id, fleet);
            const usable = opts.filter((o) => o.ok);
            if (usable.length) {
              const plan = pairPlan(peer.peer_id, usable[0].peer_id, fleet);
              return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small muted", children: [
                "could follow: ",
                usable.map((o) => `${o.peer_id} (${o.why})`).join(", "),
                plan && !plan.blockers.length && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                  " · starting asks for ",
                  plan.consents.join(" + "),
                  " first, ",
                  plan.physical ? "because frames move a real arm" : "because raw degree frames are refused by the unit envelope even in sim"
                ] }),
                plan == null ? void 0 : plan.notes.map((n, i) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: n }, i)),
                /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row small", children: usable.map((o) => startArmed === o.peer_id ? /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "row small", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: (plan == null ? void 0 : plan.physical) === false ? "btn" : "btn danger", onClick: () => startTeleop(o.peer_id), children: (plan == null ? void 0 : plan.physical) === false ? `confirm — ${peer.peer_id} (sim) follows ${o.peer_id}; nothing physical moves` : `confirm — hand-guide ${o.peer_id}, and ${peer.peer_id} MOVES with it` }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setStartArmed(null), children: "cancel" })
                ] }, o.peer_id) : /* @__PURE__ */ jsxRuntimeExports.jsxs(
                  "button",
                  {
                    className: "btn ghost small",
                    onClick: () => setStartArmed(o.peer_id),
                    title: (plan == null ? void 0 : plan.physical) === false ? `${peer.peer_id} is simulated — it will mirror ${o.peer_id}'s joints in the sim` : `${peer.peer_id} will follow ${o.peer_id}'s joints and move`,
                    children: [
                      "follow ",
                      o.peer_id
                    ]
                  },
                  o.peer_id
                )) })
              ] });
            }
            return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small muted", children: [
              "no arm on this fleet can lead it yet:",
              opts.map((o) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
                "· ",
                o.peer_id,
                " — ",
                o.why
              ] }, o.peer_id))
            ] });
          })(),
          teleop.consentKind && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "muted small", children: [
            "every frame is outside the safety envelope — settings › consent › ",
            teleop.consentKind,
            " is where that bound is widened, deliberately and by you"
          ] })
        ] }),
        stopped && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: `small ${stopped.ok ? "muted" : "warn"}`, role: "status", children: stopped.line }),
        started && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: `small ${started.ok ? "muted" : "warn"}`, role: "status", children: started.line }),
        stranded && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row small", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "button",
            {
              className: "btn ghost small",
              onClick: () => stopStranded(stranded),
              disabled: strandedResult == null ? void 0 : strandedResult.ok,
              title: `stop ${stranded} publishing its joints`,
              children: [
                "stop ",
                stranded,
                " publishing"
              ]
            }
          ),
          strandedResult && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: strandedResult.ok ? "muted" : "warn", role: "status", children: strandedResult.line })
        ] }),
        mirror && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "hint", role: "status", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "mirror to sim:" }),
          " ",
          mirrorOn ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: (mirrorLine == null ? void 0 : mirrorLine.ok) ? "" : "muted", children: (mirrorLine == null ? void 0 : mirrorLine.line) ?? `mirror started on ${mirrorOn}` }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row small", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn danger small", onClick: () => stopMirror(mirrorOn), disabled: mirrorBusy, children: "stop mirror" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "button",
                {
                  className: "btn ghost small",
                  onClick: () => refreshMirror(mirrorOn),
                  disabled: mirrorBusy,
                  title: "ask the twin what it is actually doing — the indicator is measured, not assumed",
                  children: mirrorBusy ? "asking…" : "check"
                }
              )
            ] })
          ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "muted small", children: mirrorSentence(mirror) }),
            mirror.notes.map((n, i) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "muted small", children: n }, i)),
            mirror.follower && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row small", children: mirrorArmed ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn", onClick: () => startMirror(mirror.follower), disabled: mirrorBusy, children: [
                "confirm — publish this arm's joints; ",
                mirror.follower,
                " (sim) copies them"
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setMirrorArmed(false), children: "cancel" })
            ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                className: "btn ghost small",
                onClick: () => setMirrorArmed(true),
                disabled: mirrorBusy,
                title: "start the joint stream: this arm publishes (read-only), the twin follows — nothing physical moves",
                children: [
                  "mirror to ",
                  mirror.follower
                ]
              }
            ) }),
            mirrorLine && !mirrorLine.ok && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "warn small", role: "status", children: mirrorLine.line })
          ] })
        ] }),
        whyNoJoints && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint warn", role: "status", children: whyNoJoints }),
        status && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `status-ribbon ${status.severity}`, role: "status", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: status.word }),
          " ",
          status.text
        ] }),
        offline && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "stale-note", children: [
          "no heartbeat — last seen",
          " ",
          peer.last_seen ? `${Math.round(Date.now() / 1e3 - peer.last_seen)}s ago` : "unknown",
          ". Commands will time out."
        ] }),
        camConfig && /* @__PURE__ */ jsxRuntimeExports.jsx(
          CameraConfigSheet,
          {
            peerId: peer.peer_id,
            focusCam: camConfig.cam,
            startAdding: camConfig.add,
            onClose: () => setCamConfig(null)
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "detail-body", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "detail-stage", children: [
            active ? /* @__PURE__ */ jsxRuntimeExports.jsx(CameraTile, { peerId: peer.peer_id, cam: active, meta: (_d = peer.cameras) == null ? void 0 : _d[active], big: true }) : (() => {
              var _a2;
              const ph = cameraPlaceholder(cameraEvidence(peer.peer_id, (_a2 = peer.presence) == null ? void 0 : _a2.cameras, cams, peer.cameras_requested));
              return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "camtile big", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camstate", title: ph == null ? void 0 : ph.title, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: (ph == null ? void 0 : ph.head) ?? "no camera" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: (ph == null ? void 0 : ph.sub) ?? "" })
              ] }) });
            })(),
            (cams.length > 1 || canConfig) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camswitch", children: [
              cams.length > 1 && cams.map((c) => (
                /**
                 * aria-pressed, not colour alone: `.chip.on` is the ONLY thing that said which camera is on
                 * screen, so a screen reader announced two identical "wrist, button" controls and voice
                 * control could not confirm a switch.
                 */
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "button",
                  {
                    className: c === active ? "chip on" : "chip",
                    "aria-pressed": c === active,
                    onClick: () => setCam(c),
                    children: c
                  },
                  c
                )
              )),
              canConfig && /* @__PURE__ */ jsxRuntimeExports.jsx(
                "button",
                {
                  className: "chip addcam",
                  onClick: () => setCamConfig({ cam: null, add: true }),
                  title: "attach another camera to this robot (applying restarts it)",
                  children: "+ add camera"
                }
              )
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(TelemetryStrip, { peer }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              RunForm,
              {
                peerId: peer.peer_id,
                presence: p,
                running,
                busy,
                disabled: offline,
                onRun: run,
                onStop: stop
              }
            ),
            outcome && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: outcome.ok ? "result ok" : "result bad", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
                outcome.ok ? "✓" : outcome.ambiguous ? "⚠ unknown —" : "✗",
                " ",
                outcome.text
              ] }),
              outcome.detail && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "details" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: outcome.detail })
              ] })
            ] }),
            phase === "stopping" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: "stop sent, waiting for the peer to confirm…" })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "side", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(SensorStrip, { peer }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
              "Joints (",
              joints.length,
              ")"
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(JointStrip, { state: peer.state, presence: p, problem: peer.joint_problem, peerStale: peer.stale }),
            joints.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("table", { className: "jointtable", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("thead", { children: /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "joint" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "pos" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "vel" })
              ] }) }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("tbody", { children: joints.map(([name, v]) => {
                const obj = v && typeof v === "object" && !Array.isArray(v) ? v : null;
                const pos = (obj == null ? void 0 : obj.position) ?? (typeof v === "number" ? v : Array.isArray(v) ? v[0] : void 0);
                return /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { children: name }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: typeof pos === "number" ? pos.toFixed(4) : "—" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: (obj == null ? void 0 : obj.velocity) !== void 0 ? obj.velocity.toFixed(3) : "—" })
                ] }, name);
              }) })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Peer" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("dl", { className: "kv", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "tool" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { children: (p == null ? void 0 : p.tool_name) ?? "—" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "hardware" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: (p == null ? void 0 : p.connected) === false ? "bad" : "", children: (p == null ? void 0 : p.connected) === false ? "not connected" : (p == null ? void 0 : p.hw) ?? ((p == null ? void 0 : p.connected) ? "connected" : "—") }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "task" }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("dd", { children: [
                ((_f = (_e = peer.state) == null ? void 0 : _e.task) == null ? void 0 : _f.status) ?? (p == null ? void 0 : p.task_status) ?? "idle",
                ((_h = (_g = peer.state) == null ? void 0 : _g.task) == null ? void 0 : _h.instruction) ? ` — ${peer.state.task.instruction}` : ""
              ] }),
              ((_i = p == null ? void 0 : p.sim_robots) == null ? void 0 : _i.length) ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "sim bodies" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "mono", children: p.sim_robots.join(", ") })
              ] }) : null,
              ((_j = p == null ? void 0 : p.action_keys) == null ? void 0 : _j.length) ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "action keys" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "mono", children: p.action_keys.join(", ") })
              ] }) : null,
              ((_k = p == null ? void 0 : p.topics) == null ? void 0 : _k.length) ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "topics" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "mono small", children: p.topics.join("\n") })
              ] }) : null
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
              "Policy steps ",
              steps.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: stepHz > 0 ? `${stepHz.toFixed(1)} Hz` : "" })
            ] }),
            steps.length === 0 ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Nothing yet. Steps only arrive while a policy is running and only from the moment this view opened — the stream topic is not replayed." }) : /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "steps", children: steps.map((s) => /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "stepno", children: [
                "#",
                s.step
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "mono", children: Object.entries(s.action ?? {}).slice(0, 8).map(([k, v]) => `${k}=${fmt(v)}`).join("  ") || "(no action)" })
            ] }, `${s.step}-${s.t}`)) })
          ] })
        ] })
      ]
    }
  ) });
}
const DEFAULT_RATE = 24e3;
function words(value, fallback, cap = 200) {
  if (value === void 0 || value === null || value === "") return fallback;
  const text = typeof value === "string" ? value : (() => {
    try {
      return JSON.stringify(value) ?? String(value);
    } catch {
      return String(value);
    }
  })();
  return text.slice(0, cap);
}
function interpretVoiceEvent(ev) {
  if (!ev || typeof ev !== "object") return {};
  switch (ev.type) {
    case "voice_meta": {
      const rate = Number(ev.rate);
      return { rate: Number.isFinite(rate) && rate > 0 ? rate : DEFAULT_RATE };
    }
    case "audio":
      return typeof ev.data === "string" && ev.data ? { play: ev.data } : {};
    case "transcript":
      if (!ev.text) return {};
      return { transcript: `${ev.role === "user" ? "🎙" : "🤖"} ${words(ev.text, "", 4e3)}` };
    case "needs_consent": {
      const effect = { need: ev.need ?? null };
      if (ev.spoken) effect.transcript = `⚠ ${words(ev.spoken, "")}`;
      else if (!ev.need) effect.transcript = "⚠ that voice turn was refused — open the dock for details";
      return effect;
    }
    case "error":
      return {
        transcript: `⚠ ${words(ev.error, "the voice session reported an error")}`,
        state: "error"
      };
    default:
      return {};
  }
}
function voiceCloseState(code, current) {
  if (code === 1008 || code === 4401) {
    return { state: "error", transcript: "⚠ unauthorized - sign in again, or set the dashboard token in Settings" };
  }
  return { state: current === "error" ? current : "idle" };
}
function useVoice() {
  const [state, setState] = reactExports.useState("idle");
  const [transcript, setTranscript] = reactExports.useState("");
  const [need, setNeed] = reactExports.useState(null);
  const wsRef = reactExports.useRef(null);
  const ctxRef = reactExports.useRef(null);
  const nodesRef = reactExports.useRef({});
  const playRef = reactExports.useRef({ nextT: 0, rate: 24e3 });
  const releaseMic = reactExports.useCallback(() => {
    var _a, _b, _c, _d;
    (_a = nodesRef.current.proc) == null ? void 0 : _a.disconnect();
    (_b = nodesRef.current.src) == null ? void 0 : _b.disconnect();
    (_c = nodesRef.current.stream) == null ? void 0 : _c.getTracks().forEach((t) => t.stop());
    nodesRef.current = {};
    (_d = ctxRef.current) == null ? void 0 : _d.close();
    ctxRef.current = null;
  }, []);
  const stop = reactExports.useCallback(() => {
    var _a, _b, _c;
    if (((_a = wsRef.current) == null ? void 0 : _a.readyState) === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: "stop" }));
    (_b = wsRef.current) == null ? void 0 : _b.close();
    wsRef.current = null;
    releaseMic();
    (_c = playRef.current.ctx) == null ? void 0 : _c.close();
    playRef.current.ctx = void 0;
    setState("idle");
  }, [releaseMic]);
  const start = reactExports.useCallback(async () => {
    if (state === "live" || state === "connecting") {
      stop();
      return;
    }
    setState("connecting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      nodesRef.current = { stream };
      const ws = new WebSocket(wsUrl("/ws/voice"));
      wsRef.current = ws;
      ws.onerror = () => releaseMic();
      ws.onmessage = (msg) => {
        try {
          const eff = interpretVoiceEvent(JSON.parse(msg.data));
          if (eff.rate !== void 0) playRef.current.rate = eff.rate;
          if (eff.play !== void 0) playPcm(eff.play);
          if (eff.transcript !== void 0) setTranscript(eff.transcript);
          if ("need" in eff) setNeed(eff.need);
          if (eff.state !== void 0) setState(eff.state);
        } catch {
        }
      };
      ws.onclose = (e) => {
        releaseMic();
        setState((s) => {
          const v = voiceCloseState(e.code, s);
          if (v.transcript) setTranscript(v.transcript);
          return v.state;
        });
      };
      ws.onopen = () => {
        setState("live");
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        const src = ctx.createMediaStreamSource(stream);
        const proc = ctx.createScriptProcessor(4096, 1, 1);
        nodesRef.current = { ...nodesRef.current, src, proc, stream };
        const inRate = ctx.sampleRate;
        src.connect(proc);
        proc.connect(ctx.destination);
        proc.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const f32 = e.inputBuffer.getChannelData(0);
          const ratio = inRate / 16e3;
          const outLen = Math.floor(f32.length / ratio);
          const pcm = new Int16Array(outLen);
          for (let i = 0; i < outLen; i++) {
            const v = f32[Math.floor(i * ratio)];
            pcm[i] = Math.max(-32768, Math.min(32767, v * 32767));
          }
          ws.send(pcm.buffer);
        };
      };
    } catch (e) {
      releaseMic();
      setTranscript(`⚠ ${e}`);
      setState("error");
    }
  }, [state, stop, releaseMic]);
  const playPcm = (b64) => {
    const p = playRef.current;
    if (!p.ctx) {
      p.ctx = new AudioContext();
      p.nextT = p.ctx.currentTime;
    }
    const raw = atob(b64);
    const pcm = new Int16Array(raw.length / 2);
    for (let i = 0; i < pcm.length; i++) pcm[i] = (raw.charCodeAt(2 * i) | raw.charCodeAt(2 * i + 1) << 8) << 16 >> 16;
    const buf = p.ctx.createBuffer(1, pcm.length, p.rate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const srcN = p.ctx.createBufferSource();
    srcN.buffer = buf;
    srcN.connect(p.ctx.destination);
    const t = Math.max(p.ctx.currentTime, p.nextT);
    srcN.start(t);
    p.nextT = t + buf.duration;
  };
  return { state, transcript, toggle: start, stop, need, clearNeed: () => setNeed(null) };
}
function sendFailureVerdict(f) {
  const why2 = String(f.error ?? "").trim() || "the agent socket could not be opened";
  return {
    text: `⚠ not sent: ${why2}. Nothing reached the agent, so nothing ran — your message is back in the box, press ↑ to try again.`,
    retrySafe: true
  };
}
const AUTH_TEXT = "the server refused the session - sign in again, or set the token in Settings";
function interruptionNotice(i) {
  const auth = i.code === 1008 || i.code === 4401;
  if (!i.wasBusy) {
    return auth ? { text: `⚠ ${AUTH_TEXT}`, tone: "bad", doubleRunRisk: false } : null;
  }
  const tools = (i.runningTools ?? []).filter((t) => String(t ?? "").trim());
  const partial = Math.max(0, Number(i.partialChars ?? 0) || 0);
  const head = auth ? `⚠ the turn was cut off: ${AUTH_TEXT}.` : "⚠ the connection dropped before the answer finished.";
  const parts = [head];
  if (tools.length) {
    const named = tools.slice(0, 3).join(", ");
    const more = tools.length > 3 ? ` +${tools.length - 3} more` : "";
    parts.push(
      `${tools.length === 1 ? "The tool" : "The tools"} ${named}${more} had already started, so the agent may have ACTED on the fleet — check Activity before assuming nothing happened.`
    );
  }
  parts.push(
    partial > 0 ? "The text above stops mid-answer; it is not a complete reply." : "Nothing came back, but your message had already been sent — re-sending could run it a second time."
  );
  if (partial > 0 || tools.length) {
    parts.push("Re-sending repeats the request.");
  }
  return {
    text: parts.join(" "),
    tone: "bad",
    // Delivered-then-dropped is exactly the case where a blind retry is unsafe.
    doubleRunRisk: true
  };
}
function bubbleLabel(delivered) {
  return delivered === false ? "not sent" : null;
}
const SPOKEN_MAX = 600;
function clip(text, max = SPOKEN_MAX) {
  const t = text.trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const at = cut.lastIndexOf(" ");
  return `${(at > max * 0.6 ? cut.slice(0, at) : cut).trimEnd()}… the rest is in the conversation`;
}
function turnAnnouncement({ busy, last, error }) {
  if (error) return `the fleet agent could not be reached: ${error}`;
  if (busy) return "";
  if (!last) return "";
  if (last.role === "notice") return `notice from the fleet: ${clip(last.text)}`;
  if (last.role === "user") {
    return last.delivered === false ? 'your message was not delivered — use "send again" to retry it' : "";
  }
  if (last.role !== "agent") return "";
  const ran = (last.tools ?? []).length;
  const body = last.text.trim();
  if (!body) return ran ? `the agent ran ${ran === 1 ? "1 tool" : `${ran} tools`} and said nothing` : "";
  const prefix = ran ? `the agent replied after ${ran === 1 ? "1 tool" : `${ran} tools`}: ` : "the agent replied: ";
  return prefix + clip(body);
}
const ALL_PEERS = "*ALL_PEERS*";
function str$1(v) {
  return typeof v === "string" ? v.trim() : "";
}
function parseInterruptEvent(ev) {
  if (!ev || ev.type !== "interrupt" || typeof ev.id !== "string" || !ev.id.trim()) return null;
  const r = ev.reason && typeof ev.reason === "object" ? ev.reason : {};
  const dur = typeof r.duration === "number" && isFinite(r.duration) && r.duration > 0 ? r.duration : null;
  const rawTarget = str$1(r.target) || (str$1(r.session_id) ? `sim session ${str$1(r.session_id)}` : "");
  let command = "";
  if (r.command != null && r.command !== "") {
    try {
      command = typeof r.command === "string" ? r.command : JSON.stringify(r.command);
    } catch {
      command = "";
    }
  }
  return {
    id: ev.id.trim(),
    target: rawTarget === ALL_PEERS ? "every robot on the mesh" : rawTarget || "a robot",
    fleetWide: rawTarget === ALL_PEERS,
    action: str$1(r.action) || str$1(r.tool),
    instruction: str$1(r.instruction) || str$1(r.detail),
    func: str$1(r.function),
    command,
    duration: dur,
    whyPhysical: str$1(r.why_physical),
    warning: str$1(r.warning)
  };
}
function parseStatusInterrupt(interrupt) {
  if (!interrupt || typeof interrupt !== "object") return null;
  return parseInterruptEvent({ type: "interrupt", id: interrupt.id, reason: interrupt.reason });
}
function fmtDuration(s) {
  if (s >= 60) {
    const m = Math.floor(s / 60);
    const rest = Math.round(s % 60);
    return rest ? `${m}m ${rest}s` : `${m}m`;
  }
  return `${Math.round(s * 10) / 10}s`;
}
function deed(c) {
  if (c.action === "sim_set_joints") return c.instruction ? `move joints to ${c.instruction}` : "move its joints";
  if (c.action === "sim_reset") return "return to its home pose";
  if (c.instruction) return `run "${c.instruction}"`;
  if (c.func) return `invoke ${c.func}()`;
  if (c.command) return `receive ${c.command}`;
  if (c.action === "stop" || c.action === "emergency_stop") return "stop";
  if (c.action) return `do "${c.action}"`;
  return "start a task";
}
function confirmQuestion(c) {
  const span = c.duration != null ? ` for ${fmtDuration(c.duration)}` : "";
  const stopping = c.action === "stop" || c.action === "emergency_stop";
  const sim = c.target.startsWith("sim session");
  const tail = stopping || sim ? "" : " - real motion";
  return `The agent wants ${c.target} to ${deed(c)}${span}${tail}.`;
}
function confirmDetail(c) {
  if (c.whyPhysical) return `${c.target}: ${c.whyPhysical}.`;
  if (c.warning) return c.warning;
  return "";
}
function interruptResponseBody(id, approve, always = false) {
  return { type: "resume", id, approve, always };
}
function answerNotice(c, approve, always = false) {
  return approve ? `✓ approved - ${c.target} may ${deed(c)}${always ? " for the rest of this conversation" : " this once"}` : `✗ declined — nothing was sent to ${c.target}`;
}
function AgentDock({ onSettings, startOpen = false, exampleRobot }) {
  var _a;
  const [open, setOpen] = reactExports.useState(startOpen);
  const [input, setInput] = reactExports.useState("");
  const [msgs, setMsgs] = reactExports.useState([]);
  const [busy, setBusy] = reactExports.useState(false);
  const [connError, setConnError] = reactExports.useState(null);
  const wsRef = reactExports.useRef(null);
  const busyRef = reactExports.useRef(false);
  const lastAgentRef = reactExports.useRef({ chars: 0, running: [] });
  const scrollRef = reactExports.useRef(null);
  const [need, setNeed] = reactExports.useState(null);
  const [confirm, setConfirm] = reactExports.useState(null);
  const refusedPrompt = reactExports.useRef("");
  const voice = useVoice();
  const { config } = useConfig();
  const agent = config == null ? void 0 : config.agent;
  const [info, setInfo] = reactExports.useState(null);
  reactExports.useEffect(() => {
    api("/api/agent").then(setInfo).catch(() => setInfo(null));
  }, []);
  reactExports.useEffect(() => {
    var _a2;
    (_a2 = scrollRef.current) == null ? void 0 : _a2.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    const last = msgs[msgs.length - 1];
    lastAgentRef.current = (last == null ? void 0 : last.role) === "agent" ? { chars: last.text.length, running: (last.tools ?? []).filter((t) => t.status === "running").map((t) => t.name) } : { chars: 0, running: [] };
  }, [msgs]);
  reactExports.useEffect(() => {
    busyRef.current = busy;
  }, [busy]);
  const answeredIds = reactExports.useRef(/* @__PURE__ */ new Set());
  const statusInterrupt = agent == null ? void 0 : agent.interrupt;
  reactExports.useEffect(() => {
    if (!statusInterrupt || busy) return;
    const c = parseStatusInterrupt(statusInterrupt);
    if (c && !answeredIds.current.has(c.id)) {
      setConfirm((prev) => prev ?? c);
      setOpen(true);
    }
  }, [statusInterrupt, busy]);
  const answer = async (approve, always = false) => {
    const c = confirm;
    if (!c) return;
    answeredIds.current.add(c.id);
    setConfirm(null);
    setMsgs((prev) => [...prev, { role: "notice", text: answerNotice(c, approve, always) }]);
    setBusy(true);
    try {
      const ws = await ensureWs();
      ws.send(JSON.stringify(interruptResponseBody(c.id, approve, always)));
    } catch (e) {
      setBusy(false);
      setConfirm(c);
      setConnError((e == null ? void 0 : e.message) ?? String(e));
    }
  };
  const patchAgent = (fn) => {
    setMsgs((prev) => {
      const next = [...prev];
      let last = next[next.length - 1];
      if (!last || last.role !== "agent") {
        last = { role: "agent", text: "", tools: [] };
        next.push(last);
      } else {
        last = { ...last, tools: [...last.tools ?? []] };
        next[next.length - 1] = last;
      }
      fn(last);
      return next;
    });
  };
  const ensureWs = () => new Promise((resolve, reject) => {
    var _a2;
    if (((_a2 = wsRef.current) == null ? void 0 : _a2.readyState) === WebSocket.OPEN) return resolve(wsRef.current);
    const ws = new WebSocket(wsUrl("/ws/agent"));
    ws.onopen = () => {
      wsRef.current = ws;
      setConnError(null);
      resolve(ws);
    };
    ws.onerror = () => reject(new Error(
      authRefusedRecently() ? "the agent socket was refused — this page is not signed in any more, sign in again" : "could not reach the agent socket"
    ));
    ws.onmessage = (msg) => {
      let ev;
      try {
        ev = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (ev.type === "notice") {
        setMsgs((prev) => [...prev, { role: "notice", text: ev.text }]);
        setOpen(true);
        return;
      }
      if (ev.type === "pong") return;
      if (ev.type === "interrupt") {
        const c = parseInterruptEvent(ev);
        if (c) {
          setConfirm(c);
          setBusy(false);
          setOpen(true);
        }
        return;
      }
      if (ev.type === "tool_result" && ev.needs_consent) setNeed(ev.needs_consent);
      patchAgent((last) => {
        if (ev.type === "text") last.text += ev.text;
        else if (ev.type === "reasoning") last.reasoning = (last.reasoning ?? "") + (ev.text ?? ev.data ?? "");
        else if (ev.type === "tool_use") last.tools.push({ name: ev.name, status: "running" });
        else if (ev.type === "tool_result") {
          const t = last.tools.find((t2) => t2.status === "running");
          if (t) t.status = ev.status === "error" ? "error" : "done";
        } else if (ev.type === "done") setBusy(false);
        else if (ev.type === "error") {
          last.text += `
⚠ ${ev.message ?? ev.error}`;
          setBusy(false);
        }
      });
      setOpen(true);
    };
    ws.onclose = (ev) => {
      wsRef.current = null;
      setBusy(false);
      if (ev.code === 4401) noteAuthRefusal(401);
      const verdict2 = interruptionNotice({
        code: ev.code,
        wasBusy: busyRef.current,
        partialChars: lastAgentRef.current.chars,
        runningTools: lastAgentRef.current.running
      });
      if (!verdict2) return;
      if (busyRef.current) {
        setMsgs((prev) => [...prev, { role: "notice", text: verdict2.text, bad: true }]);
        setOpen(true);
      } else {
        setConnError(verdict2.text.replace(/^⚠ /, ""));
      }
    };
  });
  const send = async (retryText) => {
    const text = (retryText ?? input).trim();
    if (!text || busy) return;
    if (!retryText) setInput("");
    refusedPrompt.current = text;
    setMsgs((prev) => [...prev, { role: "user", text }]);
    setBusy(true);
    setConnError(null);
    try {
      const ws = await ensureWs();
      ws.send(JSON.stringify({ type: "say", text }));
    } catch (e) {
      setBusy(false);
      const verdict2 = sendFailureVerdict({ error: (e == null ? void 0 : e.message) ?? String(e) });
      setMsgs((prev) => prev.map((m, i) => i === prev.length - 1 && m.role === "user" ? { ...m, delivered: false } : m));
      setInput(text);
      setConnError(verdict2.text.replace(/^⚠ /, ""));
    }
  };
  const clearHistory = async () => {
    const ws = wsRef.current;
    wsRef.current = null;
    busyRef.current = false;
    try {
      ws == null ? void 0 : ws.close(1e3, "cleared");
    } catch {
    }
    setMsgs([]);
    setConfirm(null);
    setBusy(false);
    setConnError(null);
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
    open && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dock-panel", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dock-head", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "chip static", title: "the model the console answers with", children: [
          "🤖 ",
          (info == null ? void 0 : info.model) || (agent == null ? void 0 : agent.model_id) || "default model"
        ] }),
        busy && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "chip static", children: "turn in flight" }),
        ((_a = info == null ? void 0 : info.asks_first) == null ? void 0 : _a.length) ? /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "chip static", title: `asks before: ${info.asks_first.join(", ")}`, children: [
          info.asks_first.length,
          " tools ask first"
        ] }) : null,
        (agent == null ? void 0 : agent.bridge_online) === false && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge danger", title: "the agent has no mesh bridge - its fleet tools cannot reach any robot", children: "no mesh" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "spacer" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: clearHistory, title: "Forget the conversation", children: "clear" }),
        onSettings && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onSettings, title: "Model & prompt", children: "⚒" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "div",
        {
          className: "dock-scroll",
          ref: scrollRef,
          role: "log",
          "aria-label": "conversation with the fleet agent",
          "aria-live": "off",
          "aria-busy": busy || void 0,
          children: [
            msgs.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dock-hint", children: [
              "Ask the fleet agent anything:",
              /* @__PURE__ */ jsxRuntimeExports.jsx("br", {}),
              /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: '"what robots are online?"' }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("br", {}),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { children: [
                '"tell ',
                exampleRobot ?? "so101-arm-1",
                ' to wave hello"'
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("br", {}),
              /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: '"everyone stop" — the safety brake, it halts every robot' }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "It can stop any robot, and start tasks in simulation. Starting a real arm stays with you — press ▶ on its card. Everything it does is recorded in Activity." })
            ] }),
            msgs.map((m, i) => {
              var _a2;
              return m.role === "notice" ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: m.bad ? "dock-notice bad" : "dock-notice", children: [
                m.bad ? "" : "ⓘ ",
                m.text
              ] }, i) : /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `bubble ${m.role}${m.delivered === false ? " undelivered" : ""}`, children: [
                (_a2 = m.tools) == null ? void 0 : _a2.map((t, j) => /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: `toolchip ${t.status}`, children: [
                  "⚙ ",
                  t.name
                ] }, j)),
                m.reasoning && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "reasoning", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "thinking" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: m.reasoning })
                ] }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: m.text || (busy && i === msgs.length - 1 ? "…" : "") }),
                bubbleLabel(m.delivered) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "bubble-foot", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge warn", children: bubbleLabel(m.delivered) }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn tiny", onClick: () => void send(m.text), disabled: busy, children: "send again" })
                ] })
              ] }, i);
            })
          ]
        }
      ),
      confirm && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dock-notice confirm", role: "alertdialog", "aria-label": "motion confirmation", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: /* @__PURE__ */ jsxRuntimeExports.jsxs("strong", { children: [
          "▶ ",
          confirmQuestion(confirm)
        ] }) }),
        confirmDetail(confirm) && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: confirmDetail(confirm) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "bubble-foot", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn", onClick: () => void answer(true), autoFocus: true, children: "yes, run it" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost",
              onClick: () => void answer(true, true),
              title: "approve this and every later motion on the same target for this conversation",
              children: "yes, and stop asking for this one"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void answer(false), children: "no - nothing moves" })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sr-only", role: "status", "aria-live": "polite", "aria-atomic": "true", children: turnAnnouncement({ busy, last: msgs[msgs.length - 1], error: connError }) }),
      connError && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dock-notice bad", children: [
        "⚠ ",
        connError
      ] }),
      voice.transcript && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "voice-transcript", children: voice.transcript })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dock-bar", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: `mic ${voice.state}`,
          onClick: () => {
            setOpen(true);
            voice.toggle();
          },
          title: "Speech-to-speech fleet control",
          children: voice.state === "live" ? "🔴" : voice.state === "connecting" ? "⏳" : "🎙"
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "input",
        {
          placeholder: confirm ? "answer the motion confirm above first" : `ask the fleet agent… (e.g. '${exampleRobot ?? "so101-arm-1"}, wave hello')`,
          "aria-label": "message to the fleet agent",
          value: input,
          onFocus: () => setOpen(true),
          onChange: (e) => setInput(e.target.value),
          onKeyDown: (e) => {
            if (e.key === "Enter") void send();
          },
          disabled: busy || !!confirm
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "dock-send",
          onClick: () => void send(),
          disabled: busy || !!confirm || !input.trim(),
          "aria-label": "send to the agent",
          title: "send to the agent",
          children: "↑"
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "dock-min",
          onClick: () => setOpen((o) => !o),
          "aria-label": open ? "hide the conversation" : "show the conversation",
          title: open ? "hide the conversation" : "show the conversation",
          children: open ? "▾ hide" : `▴ chat${msgs.length ? ` (${msgs.length})` : ""}`
        }
      )
    ] }),
    need ?? voice.need ? /* @__PURE__ */ jsxRuntimeExports.jsx(
      ConsentSheet,
      {
        need: need ?? voice.need,
        target: "spawn",
        onCancel: () => {
          setNeed(null);
          voice.clearNeed();
        },
        onRetry: () => {
          const again = refusedPrompt.current;
          setNeed(null);
          if (voice.need) voice.clearNeed();
          else void send(again);
        }
      }
    ) : null
  ] });
}
const LOCAL$1 = /* @__PURE__ */ new Set(["localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0", ""]);
function isLocalHost(host) {
  return LOCAL$1.has((host || "").trim().toLowerCase());
}
function authRemovalWarning(facts) {
  const host = (facts.host || "").trim();
  const remote = !isLocalHost(host);
  const lines = [];
  if (remote) {
    lines.push(
      `You are viewing this dashboard as ${host}, not localhost — so it is reachable from outside this machine, and removing the token opens it to everyone who can reach that address.`
    );
  } else {
    lines.push(
      "Anyone who can reach this port gets full control — including from another machine on this network, or through any tunnel that forwards it."
    );
  }
  const n = facts.peerCount ?? 0;
  lines.push(
    n > 0 ? `${n} robot${n === 1 ? "" : "s"} on this fleet can then be commanded, and motors moved, without a token.` : "Any robot that joins this fleet can then be commanded, and its motors moved, without a token."
  );
  if ((facts.corsOrigins ?? "").trim() === "*") {
    lines.push("CORS is set to * , so a web page on any site can call this API from a browser too.");
  }
  lines.push("Setting a token again re-locks it immediately — this is reversible, but not automatic.");
  return {
    severity: remote ? "exposed" : "local",
    lines,
    confirmLabel: remote ? "yes — leave it open to the network" : "yes — remove the token"
  };
}
const LOCAL = /* @__PURE__ */ new Set(["localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"]);
function hostOf(base, pageHost = "") {
  const v = (base ?? "").trim();
  if (!v) return (pageHost || "").toLowerCase();
  try {
    return new URL(/^[a-z]+:\/\//i.test(v) ? v : `http://${v}`).host.toLowerCase();
  } catch {
    return "";
  }
}
function isLocal(host) {
  return LOCAL.has(host.replace(/:\d+$/, ""));
}
function connectionChange(c) {
  const nextRaw = (c.nextBase ?? "").trim();
  const nextToken = (c.nextToken ?? "").trim();
  const currentToken = (c.currentToken ?? "").trim();
  if (nextRaw && hostOf(nextRaw, c.pageHost) === "") {
    return {
      kind: "unparseable",
      detail: `"${nextRaw}" is not an address this browser can dial. Use host:port or a full URL (https://robot.lan:8090); leave it empty to talk to the origin that served this page`
    };
  }
  const from = hostOf(c.currentBase, c.pageHost);
  const to = hostOf(nextRaw, c.pageHost);
  const carryingOldToken = nextToken !== "" && nextToken === currentToken;
  if (carryingOldToken && from !== to) {
    return {
      kind: "token_follows_host",
      fromHost: from || "(this origin)",
      toHost: to || "(this origin)",
      detail: `The token in this browser was given for ${from || "this origin"}, and connecting to ${to || "this origin"} will send it there — a credential for one machine handed to another. If that address is a typo or not the robot you think it is, the secret is gone.`,
      alternative: "connect without a token"
    };
  }
  if (nextToken !== "" && to && !isLocal(to)) {
    const scheme = /^https:\/\//i.test(nextRaw) ? "https" : /^[a-z]+:\/\//i.test(nextRaw) ? "http" : "http";
    if (scheme === "http") {
      return {
        kind: "cleartext_token",
        toHost: to,
        detail: `http://${to} is not encrypted, so this token crosses the network in clear text — anyone on the path can read it and use it to move motors. https:// keeps it private.`,
        alternative: "connect without a token"
      };
    }
  }
  return { kind: "ok" };
}
function needsConfirm(v) {
  return v.kind === "token_follows_host" || v.kind === "cleartext_token";
}
function syncDrafts(current, lastServer, nextServer) {
  const next = { ...current };
  const kept = [];
  const conflicts = [];
  const adopted = [];
  for (const key of Object.keys(nextServer)) {
    const server = nextServer[key];
    const seeded = lastServer[key];
    const draft = current[key];
    if (seeded === void 0) {
      next[key] = server;
      if (draft !== void 0 && draft !== server) kept.push(key), next[key] = draft;
      else adopted.push(key);
      continue;
    }
    const touched = draft !== void 0 && draft !== seeded;
    if (!touched) {
      if (server !== draft) adopted.push(key);
      next[key] = server;
      continue;
    }
    next[key] = draft;
    if (server === draft) adopted.push(key);
    else if (server !== seeded) conflicts.push(key);
    else kept.push(key);
  }
  return { next, kept, conflicts, adopted };
}
function dirtyFields(current, server) {
  return Object.keys(server).filter((k) => current[k] !== void 0 && current[k] !== server[k]).sort();
}
function unsavedSummary(dirty, labels = {}) {
  if (!dirty.length) return "";
  const names = dirty.map((k) => labels[k] ?? k);
  const list = names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
  return `Unsaved ${names.length === 1 ? "change" : "changes"} to ${list}`;
}
function ConsentSettings() {
  const [state, setState] = reactExports.useState(null);
  const [busy, setBusy] = reactExports.useState(null);
  const [note, setNote] = reactExports.useState(null);
  const [error, setError] = reactExports.useState(null);
  const load2 = reactExports.useCallback(async () => {
    try {
      setState(await api("/api/consent"));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  reactExports.useEffect(() => {
    void load2();
  }, [load2]);
  const setLock = async (on) => {
    var _a;
    const key = ((_a = state == null ? void 0 : state.locks) == null ? void 0 : _a.task_requires_confirm_env) ?? "STRANDS_DASH_TASK_REQUIRES_CONFIRM";
    setBusy("lock");
    setNote(null);
    setError(null);
    try {
      await post("/api/config", { env: { [key]: on ? "1" : "" } });
      setNote(on ? "on — a task that would move a real robot now needs the ▶ confirmation" : "off — any caller with the API token can start a real task again");
      await load2();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const revoke = async (kind, subject, label2) => {
    setBusy(label2);
    setNote(null);
    setError(null);
    try {
      const r = await post("/api/consent/revoke", { kind, subject });
      setNote(r.note ?? (r.revoked ? `revoked ${label2}` : `nothing to revoke for ${label2}`));
      await load2();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  if (error && !state) return /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
    "could not read permissions: ",
    error
  ] });
  const envelope = state == null ? void 0 : state.teleop_degree_units;
  const nothing = state && nothingGranted(state);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "consent-settings", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Permissions you granted" }),
    nothing ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Nothing extra is allowed here. When a safety guard refuses something, the dashboard asks you first and what you approve shows up in this list." }) : null,
    (state == null ? void 0 : state.trust_remote_code) ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "Run model code from HuggingFace" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: "Any policy load may execute code from the model repository — this is the widest permission on the list." })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn ghost danger",
          disabled: busy === "trust",
          onClick: () => revoke("trust_remote_code", null, "trust"),
          children: busy === "trust" ? "…" : "revoke"
        }
      )
    ] }) : null,
    (state == null ? void 0 : state.agent_physical_motion) ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "The agent may start motion on real robots" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: "A chat sentence or a voice command can put any real robot on this mesh in motion, with no confirmation and without the check that the policy fits that robot — the one the ▶ button does. Revoking leaves the agent able to stop robots, answer questions and run tasks in simulation; starting a real arm comes back to you." })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn ghost danger",
          disabled: busy === "agent motion",
          onClick: () => revoke("agent_physical_motion", null, "agent motion"),
          children: busy === "agent motion" ? "…" : "revoke"
        }
      )
    ] }) : null,
    (state == null ? void 0 : state.locks) ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "Require the ▶ confirmation before real motion" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: state.locks.task_requires_confirm ? "On. A task that would move a real robot is refused unless it comes from the ▶ button (or a script that says so explicitly). Simulated robots and stopping are never affected." : "Off. Anything holding this dashboard’s API token — a script, a terminal, whoever finds the token if this dashboard is reachable from the internet — can start a real robot with one request and no confirmation. Turning this on does not change the ▶ button." })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: `btn ghost${state.locks.task_requires_confirm ? "" : " danger"}`,
          disabled: busy === "lock",
          onClick: () => void setLock(!state.locks.task_requires_confirm),
          children: busy === "lock" ? "…" : state.locks.task_requires_confirm ? "turn off" : "turn on"
        }
      )
    ] }) : null,
    (envelope == null ? void 0 : envelope.granted) ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
          "Teleop envelope widened",
          envelope.is_degree_preset ? " to degrees" : ""
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "hint", children: [
          "Every teleop stream on this machine, not one arm: a single frame may command a reach of ",
          envelope.value_abs ?? "the default",
          " units",
          envelope.slew_abs ? ` at up to ${envelope.slew_abs} units/s` : " (speed bound left at the default)",
          ".",
          envelope.is_degree_preset ? " This is the degrees preset, which an SO-101 needs — a runaway far outside it is still refused." : " This is a hand-set bound, not the degrees preset — check it is the one you meant."
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn ghost danger",
          disabled: busy === "teleop",
          onClick: () => revoke("teleop_degree_units", null, "the teleop envelope"),
          children: busy === "teleop" ? "…" : "revoke"
        }
      )
    ] }) : null,
    ((state == null ? void 0 : state.policy_type_allow) ?? []).map((entry) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { className: "rc-mono", children: entry }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: "this policy may be built and run — one name, no wildcard" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn ghost danger",
          disabled: busy === entry,
          onClick: () => revoke("policy_type_allow", entry, entry),
          children: busy === entry ? "…" : "revoke"
        }
      )
    ] }, `type-${entry}`)),
    ((state == null ? void 0 : state.policy_host_allow) ?? []).map((entry) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { className: "rc-mono", children: entry }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: entry.includes("/") ? "every address in this range may run policies for your robots — wider than one host" : "policies may run on this host: it receives camera frames and joint states, and what it returns drives the arms" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn ghost danger",
          disabled: busy === entry,
          onClick: () => revoke("policy_host_allow", entry, entry),
          children: busy === entry ? "…" : "revoke"
        }
      )
    ] }, `host-${entry}`)),
    state == null ? void 0 : state.hf_repo_allow.map((entry) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "cg-row", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { className: "rc-mono", children: entry }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: entry.includes("/") ? "this model only" : "every model under this organisation — wider than a single approval" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn ghost danger",
          disabled: busy === entry,
          onClick: () => revoke("hf_repo_allow", entry, entry),
          children: busy === entry ? "…" : "revoke"
        }
      )
    ] }, entry)),
    note ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "cs-note", role: "status", children: note }) : null,
    error ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "cs-error", role: "alert", children: error }) : null,
    /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Revoking applies to robots started from now on: a peer that is already running keeps the permission it was started with until you respawn it." })
  ] });
}
function whenText(created, nowMs) {
  if (created === null || created === void 0 || created === "") return "";
  const n = typeof created === "number" ? created : Number(created);
  if (!Number.isFinite(n) || n <= 0) return "";
  const ms = n > 1e12 ? n : n * 1e3;
  const secs = (nowMs - ms) / 1e3;
  if (secs < 0) return "just now";
  return `added ${agoText(secs)}`;
}
function passkeyRows(creds, nowMs = Date.now()) {
  const list = Array.isArray(creds) ? creds.filter((c) => c && typeof c.id === "string" && c.id) : [];
  const last = list.length <= 1;
  return list.map((c) => ({
    id: c.id,
    label: (c.name ?? "").trim() || "passkey",
    when: whenText(c.created, nowMs),
    revocable: !last,
    reason: last ? "this is the only key to this dashboard — enroll another device first, or removing it would re-open the setup flow to anyone who can reach this page" : ""
  }));
}
function revokeRefusal(status, detail) {
  const text = (detail || "").trim();
  if (status === 409) return text || "cannot remove the last passkey — enroll another first";
  if (status === 404) return "that passkey is already gone — reload to see the current list";
  if (status === 401 || status === 403) return "your session expired — reload and sign in again";
  return text || "could not remove that passkey";
}
function passkeySummary(rows, authRequired) {
  if (rows.length === 0) {
    return authRequired ? "no passkey is enrolled, yet this dashboard demands one — nobody can sign in until a device is enrolled" : "no passkey is enrolled: anyone who can reach this page can use this dashboard";
  }
  if (rows.length === 1) return "1 device can sign in to this dashboard";
  return `${rows.length} devices can sign in to this dashboard`;
}
function PasskeyList({ authRequired }) {
  const [creds, setCreds] = reactExports.useState(null);
  const [msg, setMsg] = reactExports.useState("");
  const [busy, setBusy] = reactExports.useState("");
  const [absent, setAbsent] = reactExports.useState(false);
  const load2 = reactExports.useCallback(async () => {
    try {
      const r = await api("/api/auth/credentials");
      setCreds(r.credentials ?? []);
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) setAbsent(true);
      else setMsg(e instanceof Error ? e.message : String(e));
    }
  }, []);
  reactExports.useEffect(() => {
    void load2();
  }, [load2]);
  if (absent) return null;
  const rows = passkeyRows(creds);
  const revoke = async (id, label2) => {
    setMsg("");
    setBusy(id);
    try {
      await del(`/api/auth/credentials/${encodeURIComponent(id)}`);
      setMsg(`removed ${label2}`);
      await load2();
    } catch (e) {
      setMsg(`✗ ${revokeRefusal(
        e instanceof HttpError ? e.status : 0,
        e instanceof Error ? e.message : String(e)
      )}`);
    } finally {
      setBusy("");
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("h4", { children: "Devices that can sign in" }),
    creds === null ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "reading…" }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: passkeySummary(rows, authRequired) }),
      rows.map((r) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row between cg-row", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "muted small", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: r.label }),
          r.when ? ` · ${r.when}` : "",
          !r.revocable && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint small", children: [
            " — ",
            r.reason
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "btn ghost tiny danger",
            disabled: !r.revocable || busy === r.id,
            title: r.revocable ? "this device can no longer sign in" : r.reason,
            onClick: () => void revoke(r.id, r.label),
            children: busy === r.id ? "removing…" : "remove"
          }
        )
      ] }, r.id))
    ] }),
    msg && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: msg.startsWith("✗") ? "warn small" : "hint small", children: msg }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Removing a passkey takes effect immediately; a browser already signed in keeps its session until it reloads." })
  ] });
}
const APPLY_LABEL = {
  live: "applies immediately",
  "next-turn": "applies on the next agent turn",
  "mesh-restart": "needs a mesh restart",
  respawn: "applies to robots spawned from now on",
  startup: "applies at next server start"
};
function finiteNumber(raw, opts = {}) {
  const s = raw.trim();
  if (s === "") return null;
  const n = Number(s);
  if (!Number.isFinite(n)) return `${opts.name ?? "value"} must be a number`;
  if (opts.integer && !Number.isInteger(n)) return `${opts.name ?? "value"} must be a whole number`;
  if (opts.min !== void 0 && n < opts.min) return `minimum is ${opts.min}`;
  if (opts.max !== void 0 && n > opts.max) return `maximum is ${opts.max}`;
  return null;
}
const TLS_BEARING = ["tls", "quic", "wss", "unixsock"];
const PLAINTEXT = ["tcp", "udp"];
const ALL_SCHEMES = [...TLS_BEARING, ...PLAINTEXT];
function endpointError(ep, dialable) {
  const [addr] = ep.split("#");
  const slash = addr.indexOf("/");
  if (slash <= 0) return `"${ep}" is not proto/host:port (e.g. tls/robot.lan:7447)`;
  const scheme = addr.slice(0, slash).toLowerCase();
  const rest = addr.slice(slash + 1);
  if (!ALL_SCHEMES.includes(scheme)) {
    return `"${scheme}" is not a mesh transport — use one of ${ALL_SCHEMES.join(", ")}`;
  }
  if (scheme === "unixsock") {
    return rest.trim() === "" ? `"${ep}" needs a socket path (unixsock/tmp/zenoh.sock)` : null;
  }
  const colon = rest.lastIndexOf(":");
  if (colon <= 0) return `"${ep}" is missing a port (e.g. ${scheme}/robot.lan:7447)`;
  const host = rest.slice(0, colon);
  const portRaw = rest.slice(colon + 1);
  if (/\s/.test(host) || host === "") return `"${ep}" has no host`;
  if (!/^\d{1,5}$/.test(portRaw)) return `"${ep}" has a non-numeric port`;
  const port = Number(portRaw);
  if (port > 65535) return `"${ep}" has an out-of-range port`;
  if (port === 0 && dialable) return `"${ep}" cannot be dialled — port 0 means "any free port", which only a listen endpoint can use`;
  if (!TLS_BEARING.includes(scheme)) {
    return `"${ep}" uses ${scheme}, which the default mTLS posture refuses when the mesh restarts — use ${TLS_BEARING.join("/")}, or set STRANDS_MESH_AUTH_MODE=none for the insecure development posture`;
  }
  return null;
}
function endpointsError(raw, dialable) {
  const s = raw.trim();
  if (s === "") return null;
  for (const part of s.split(",")) {
    const ep = part.trim();
    if (ep === "") continue;
    const err = endpointError(ep, dialable);
    if (err) return err;
  }
  return null;
}
function connectEndpoints(raw) {
  return endpointsError(raw, true);
}
function listenEndpoints(raw) {
  return endpointsError(raw, false);
}
const SETTINGS = [
  {
    key: "agent.temperature",
    label: "Temperature",
    effect: "Higher = more varied agent replies; lower = more deterministic. Empty uses the model default.",
    safeDefault: "",
    apply: "next-turn",
    validate: (raw) => finiteNumber(raw, { min: 0, max: 2, name: "temperature" })
  },
  {
    key: "agent.max_tokens",
    label: "Max tokens",
    effect: "Caps the length of one agent reply. Too low truncates tool use mid-thought.",
    unit: "tokens",
    safeDefault: "",
    apply: "next-turn",
    validate: (raw) => finiteNumber(raw, { min: 1, max: 2e5, integer: true, name: "max tokens" })
  },
  {
    key: "agent.model_id",
    label: "Model id",
    effect: "Which model answers chat and fleet commands. Empty uses the provider default.",
    safeDefault: "",
    apply: "next-turn",
    validate: () => null
  },
  {
    key: "mesh.port",
    label: "Mesh port",
    effect: "UDP/TCP port the zenoh mesh binds. Every robot on the desk must agree on it.",
    unit: "port",
    safeDefault: "7447",
    apply: "mesh-restart",
    validate: (raw) => finiteNumber(raw, { min: 1, max: 65535, integer: true, name: "port" })
  },
  {
    key: "mesh.camera_hz",
    label: "Camera rate",
    effect: "How many frames per second each robot publishes. Higher is smoother but costs LAN bandwidth. Each robot reads it when IT starts, so a running robot keeps its rate until you respawn it.",
    unit: "Hz",
    safeDefault: "5",
    apply: "respawn",
    validate: (raw) => finiteNumber(raw, { min: 0.1, max: 60, name: "camera rate" })
  },
  {
    key: "mesh.connect",
    label: "Connect endpoints",
    effect: "Other mesh routers this dashboard dials out to. Empty relies on multicast discovery.",
    safeDefault: "",
    apply: "mesh-restart",
    validate: connectEndpoints
  },
  {
    key: "mesh.listen",
    label: "Listen endpoints",
    effect: "Addresses this dashboard accepts mesh connections on. Empty uses zenoh defaults; port 0 means any free port.",
    safeDefault: "",
    apply: "mesh-restart",
    validate: listenEndpoints
  },
  {
    key: "voice.provider",
    label: "Voice provider",
    effect: "Which service speaks and listens. Each provider needs its credential in the Env tab.",
    safeDefault: "openai",
    apply: "live",
    validate: () => null
  },
  {
    key: "voice.voice_name",
    label: "Voice name",
    effect: "Which of the provider's voices answers. Empty uses the provider default.",
    safeDefault: "",
    apply: "live",
    validate: () => null
  }
];
const byKey = new Map(SETTINGS.map((s) => [s.key, s]));
const settingMeta = (key) => byKey.get(key);
function validateSetting(key, raw) {
  var _a;
  return ((_a = byKey.get(key)) == null ? void 0 : _a.validate(raw)) ?? null;
}
function envKeyError(raw) {
  const s = raw.trim();
  if (s === "") return null;
  if (/[\r\n]/.test(raw)) return "key must be a single line";
  if (!/^[A-Z_][A-Z0-9_]*$/.test(s)) {
    return "keys are UPPER_SNAKE_CASE: letters, digits and _ only, not starting with a digit";
  }
  return null;
}
function envValueError(raw) {
  if (/[\r\n]/.test(raw)) return "value must be a single line - a line break would write a second variable";
  return null;
}
const TAB_OF = {
  "agent.temperature": "agent",
  "agent.max_tokens": "agent",
  "agent.model_id": "agent",
  "mesh.port": "mesh",
  "mesh.camera_hz": "mesh",
  "mesh.connect": "mesh",
  "mesh.listen": "mesh",
  "voice.provider": "voice",
  "voice.voice_name": "voice"
};
const EXTRA_ENTRIES = [
  { key: "connection.base", label: "API base URL", tab: "connection", keywords: "backend server address host remote", effect: "Which dashboard server this browser talks to." },
  { key: "connection.token", label: "Auth token (this browser)", tab: "connection", keywords: "login password bearer", effect: "Credential this browser sends with every request." },
  { key: "agent.system_prompt", label: "System prompt", tab: "agent", keywords: "instructions personality behavior", effect: "Standing instructions for the fleet agent." },
  { key: "env.vars", label: "Environment variables", tab: "env", keywords: "api key secret credential openai huggingface hf token .env", effect: "Credentials and flags written to the server .env file." },
  { key: "env.trust_remote_code", label: "HuggingFace trust_remote_code", tab: "env", keywords: "lerobot kimodo model repo security allow", effect: "Allows model repos to execute their own code when loaded." },
  { key: "security.auth_token", label: "Server auth token", tab: "security", keywords: "password protect lock api", effect: "Token every client must present on /api and /ws." },
  { key: "security.cors_origins", label: "CORS origins", tab: "security", keywords: "browser cross origin websites", effect: "Which websites a browser may call this API from. Adding one needs a server restart; removing one is refused for writes and websockets straight away." },
  { key: "mesh.restart", label: "Restart mesh", tab: "mesh", keywords: "re-point reconnect zenoh session", effect: "Re-opens the shared mesh session." }
];
const KEYWORDS_OF = {
  "agent.temperature": "sampling randomness creativity",
  "agent.max_tokens": "length limit reply cutoff",
  "agent.model_id": "llm claude bedrock provider",
  "mesh.port": "zenoh network 7447",
  "mesh.camera_hz": "fps frames rate video bandwidth",
  "mesh.connect": "endpoints dial router zenoh peer",
  "mesh.listen": "endpoints bind accept zenoh",
  "voice.provider": "speech tts openai nova sonic",
  "voice.voice_name": "speaker tts voice"
};
const SEARCH_INDEX = [
  ...SETTINGS.filter((s) => TAB_OF[s.key]).map((s) => ({
    key: s.key,
    label: s.label,
    tab: TAB_OF[s.key],
    keywords: KEYWORDS_OF[s.key] ?? "",
    effect: s.effect
  })),
  ...EXTRA_ENTRIES
];
function searchSettings(query, limit = 8) {
  const q = query.trim().toLowerCase();
  if (q === "") return [];
  const terms = q.split(/\s+/);
  const scored = [];
  for (const e of SEARCH_INDEX) {
    const label2 = e.label.toLowerCase();
    const hay = `${label2} ${e.key.toLowerCase()} ${e.keywords.toLowerCase()} ${e.effect.toLowerCase()}`;
    if (!terms.every((t) => hay.includes(t))) continue;
    let score = 1;
    if (label2.includes(q)) score = 2;
    if (label2.startsWith(q)) score = 3;
    scored.push({ e, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit).map((s) => s.e);
}
function FieldMeta({ k, raw }) {
  const meta = settingMeta(k);
  const err = validateSetting(k, raw);
  if (err) return /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "field-err", role: "alert", children: [
    "⚠ ",
    err
  ] });
  if (!meta) return null;
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "field-meta", children: [
    meta.effect,
    meta.safeDefault !== "" && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      " · default ",
      meta.safeDefault,
      meta.unit ? ` ${meta.unit}` : ""
    ] }),
    " · ",
    /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `apply-chip ${meta.apply}`, children: APPLY_LABEL[meta.apply] })
  ] });
}
const DRAFT_LABELS = {
  modelId: "the model id",
  prompt: "the system prompt",
  temperature: "temperature",
  maxTokens: "max tokens",
  voiceProvider: "the voice provider",
  voiceName: "the voice",
  connect: "mesh connect",
  listen: "mesh listen",
  meshPort: "the mesh port",
  meshBackend: "the mesh transport",
  cameraHz: "camera Hz",
  corsOrigins: "CORS origins"
};
const TAB_OF_FIELD = {
  modelId: "agent",
  prompt: "agent",
  temperature: "agent",
  maxTokens: "agent",
  voiceProvider: "voice",
  voiceName: "voice",
  connect: "mesh",
  listen: "mesh",
  meshPort: "mesh",
  meshBackend: "mesh",
  cameraHz: "mesh",
  corsOrigins: "security"
};
const TABS = [
  { id: "connection", label: "Connection" },
  { id: "agent", label: "Agent" },
  { id: "voice", label: "Voice" },
  { id: "mesh", label: "Mesh" },
  { id: "env", label: "Env" },
  { id: "security", label: "Security" }
];
function SettingsDrawer({ open, onClose, mesh, initialTab }) {
  var _a, _b, _c;
  const { config, loading, error, reload, save } = useConfig();
  const [tab, setTab] = reactExports.useState("connection");
  const sheetRef = reactExports.useRef(null);
  useDialogFocus(sheetRef, open);
  const [query, setQuery] = reactExports.useState("");
  reactExports.useEffect(() => {
    if (open && initialTab) setTab(initialTab);
  }, [open, initialTab]);
  const [status, setStatus] = reactExports.useState(null);
  const [saving, setSaving] = reactExports.useState(false);
  const [base, setBase] = reactExports.useState(backendBase());
  const [token, setToken] = reactExports.useState(authToken());
  const [modelId, setModelId] = reactExports.useState("");
  const [prompt, setPrompt] = reactExports.useState("");
  const [temperature, setTemperature] = reactExports.useState("");
  const [maxTokens, setMaxTokens] = reactExports.useState("");
  const [voiceProvider, setVoiceProvider] = reactExports.useState("");
  const [voiceName, setVoiceName] = reactExports.useState("");
  const [connect, setConnect] = reactExports.useState("");
  const [listen, setListen] = reactExports.useState("");
  const [meshPort, setMeshPort] = reactExports.useState("");
  const [meshBackend, setMeshBackend] = reactExports.useState("");
  const [cameraHz, setCameraHz] = reactExports.useState("");
  const [trustRemote, setTrustRemote] = reactExports.useState(false);
  const [envDraft, setEnvDraft] = reactExports.useState({});
  const [newKey, setNewKey] = reactExports.useState("");
  const [newValue, setNewValue] = reactExports.useState("");
  const [serverToken, setServerToken] = reactExports.useState("");
  const [envUnset, setEnvUnset] = reactExports.useState([]);
  const [removeArmed, setRemoveArmed] = reactExports.useState(false);
  const [discardArmed, setDiscardArmed] = reactExports.useState(false);
  const [corsOrigins, setCorsOrigins] = reactExports.useState("");
  const [connVerdict, setConnVerdict] = reactExports.useState(null);
  const serverDrafts = reactExports.useMemo(() => {
    if (!config) return {};
    const ms = config.mesh.settings ?? {};
    const out = {
      modelId: config.agent.model_id ?? "",
      prompt: config.agent.system_prompt ?? "",
      temperature: config.agent.temperature === null ? "" : String(config.agent.temperature),
      maxTokens: config.agent.max_tokens === null ? "" : String(config.agent.max_tokens),
      voiceProvider: config.voice.provider,
      voiceName: config.voice.voice_name ?? "",
      connect: (ms.connect ?? []).join(", "),
      listen: (ms.listen ?? []).join(", "),
      meshPort: ms.port ? String(ms.port) : "",
      meshBackend: ms.backend ?? "",
      cameraHz: ms.camera_hz ? String(ms.camera_hz) : "",
      corsOrigins: (config.security.cors_origins ?? []).join(", ")
    };
    return out;
  }, [config]);
  const SETTERS = {
    modelId: setModelId,
    prompt: setPrompt,
    temperature: setTemperature,
    maxTokens: setMaxTokens,
    voiceProvider: setVoiceProvider,
    voiceName: setVoiceName,
    connect: setConnect,
    listen: setListen,
    meshPort: setMeshPort,
    meshBackend: setMeshBackend,
    cameraHz: setCameraHz,
    corsOrigins: setCorsOrigins
  };
  const currentDrafts = {
    modelId,
    prompt,
    temperature,
    maxTokens,
    voiceProvider,
    voiceName,
    connect,
    listen,
    meshPort,
    meshBackend,
    cameraHz,
    corsOrigins
  };
  const seededRef = reactExports.useRef({});
  const currentRef = reactExports.useRef(currentDrafts);
  currentRef.current = currentDrafts;
  reactExports.useEffect(() => {
    var _a2;
    if (!config) return;
    const r = syncDrafts(currentRef.current, seededRef.current, serverDrafts);
    for (const [key, value] of Object.entries(r.next)) {
      if (currentRef.current[key] !== value) (_a2 = SETTERS[key]) == null ? void 0 : _a2.call(SETTERS, value);
    }
    seededRef.current = serverDrafts;
    if (r.conflicts.length) {
      setStatus(
        `⚠ changed on the server while you were editing: ${r.conflicts.join(", ")} — your version is still in the field, saving will overwrite theirs`
      );
    }
    setTrustRemote(config.runtime.trust_remote_code);
    setEnvDraft({});
    setServerToken("");
  }, [config, serverDrafts]);
  if (!open) return null;
  const dirty = dirtyFields(currentDrafts, serverDrafts);
  const unsaved = unsavedSummary(dirty, DRAFT_LABELS);
  const requestClose = () => {
    if (dirty.length) {
      setDiscardArmed(true);
      return;
    }
    onClose();
  };
  const agentValid = validateSetting("agent.temperature", temperature) === null && validateSetting("agent.max_tokens", maxTokens) === null;
  const meshValid = validateSetting("mesh.port", meshPort) === null && validateSetting("mesh.camera_hz", cameraHz) === null && validateSetting("mesh.connect", connect) === null && validateSetting("mesh.listen", listen) === null;
  const envValid = envKeyError(newKey) === null && envValueError(newValue) === null && Object.values(envDraft).every((v) => envValueError(v) === null);
  const results = searchSettings(query);
  const report = (r) => {
    var _a2, _b2, _c2, _d, _e;
    const parts = [];
    if (r.applied.length) parts.push(`applied ${r.applied.join(", ")}`);
    if (r.env_written.length) parts.push(`wrote ${r.env_written.join(", ")} to .env`);
    if ((_a2 = r.env_removed) == null ? void 0 : _a2.length) {
      parts.push(
        `removed ${r.env_removed.join(", ")} from .env (gone for this process and every robot spawned from now on; already-running robots keep it)`
      );
    }
    if (r.agent_reset) parts.push("agent will rebuild on the next turn");
    if (r.skipped_masked.length) parts.push(`skipped unchanged secrets: ${r.skipped_masked.join(", ")}`);
    if (r.restart_required.length) parts.push(`needs a mesh restart: ${r.restart_required.join(", ")}`);
    if ((_b2 = r.startup_required) == null ? void 0 : _b2.length) {
      parts.push(
        `saved, takes effect at the next server start: ${r.startup_required.join(", ")} (a removed origin is already refused for writes and websockets)`
      );
    }
    if ((_c2 = r.respawn_required) == null ? void 0 : _c2.length) {
      parts.push(
        `saved for robots spawned from now on: ${r.respawn_required.join(", ")} (respawn a robot to change its rate)`
      );
    }
    if (r.mesh_restart) {
      parts.push(r.mesh_restart.mesh_online ? "mesh re-pointed" : "mesh re-point FAILED (offline)");
      if ((_d = r.mesh_restart.orphaned) == null ? void 0 : _d.length) parts.push(`orphaned local robots: ${r.mesh_restart.orphaned.join(", ")}`);
    }
    if (r.errors.length) parts.push(`errors: ${r.errors.join("; ")}`);
    if ((_e = r.ignored) == null ? void 0 : _e.length) parts.push(`⚠ not recognised, so not saved: ${r.ignored.join(", ")}`);
    setStatus(parts.join(" · ") || "nothing changed");
  };
  const apply = async (body) => {
    setSaving(true);
    setStatus(null);
    try {
      report(await save(body));
    } catch (e) {
      setStatus(`⚠ ${(e == null ? void 0 : e.message) ?? String(e)}`);
    } finally {
      setSaving(false);
    }
  };
  const goConnect = (tokenToSend) => {
    setBackendBase(base);
    setAuthToken(tokenToSend);
    location.reload();
  };
  const applyConnection = () => {
    const v = connectionChange({
      currentBase: backendBase(),
      currentToken: authToken(),
      nextBase: normalize(base) || base,
      nextToken: token,
      pageHost: typeof location !== "undefined" ? location.host : ""
    });
    if (v.kind === "ok") {
      setConnVerdict(null);
      goConnect(token);
      return;
    }
    setConnVerdict(v);
  };
  const restartMesh = async (force = false) => {
    setSaving(true);
    setStatus(null);
    try {
      const r = await post("/api/mesh/restart", { force });
      setStatus(r.mesh_online ? "mesh re-opened" : "⚠ mesh is offline after restart");
      await reload();
    } catch (e) {
      setStatus(`⚠ ${(e == null ? void 0 : e.message) ?? String(e)}`);
    } finally {
      setSaving(false);
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "drawer-backdrop", onClick: requestClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("aside", { ref: sheetRef, className: "drawer", onClick: (e) => e.stopPropagation(), children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("header", { className: "drawer-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "Settings" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: requestClose, "aria-label": "close settings", title: "Escape", children: "✕" })
    ] }),
    discardArmed && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", role: "alert", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: unsaved }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { children: [
        "Closing now discards ",
        dirty.length === 1 ? "it" : "them",
        ". Save the tab you were editing first, or discard."
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: () => setDiscardArmed(false), children: "keep editing" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost danger", onClick: () => {
          setDiscardArmed(false);
          onClose();
        }, children: "discard and close" })
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "settings-search", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "input",
        {
          type: "search",
          placeholder: "Search settings… (fps, token, prompt)",
          "aria-label": "Search settings",
          value: query,
          onChange: (e) => setQuery(e.target.value)
        }
      ),
      results.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "search-results", role: "listbox", children: results.map((r) => /* @__PURE__ */ jsxRuntimeExports.jsx("li", { children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          role: "option",
          onClick: () => {
            setTab(r.tab);
            setQuery("");
          },
          children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: r.label }),
            " ",
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "tabname", children: r.tab }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: r.effect })
          ]
        }
      ) }, r.key)) }),
      query.trim() !== "" && results.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        'no setting matches "',
        query.trim(),
        '"'
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("nav", { className: "tabs", children: TABS.map((t) => {
      const tabDirty = dirty.some((k) => (TAB_OF_FIELD[k] ?? "") === t.id);
      return /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: tab === t.id ? "tab on" : "tab",
          "aria-pressed": tab === t.id,
          onClick: () => setTab(t.id),
          title: tabDirty ? "unsaved changes on this tab" : void 0,
          children: [
            t.label,
            tabDirty && /* @__PURE__ */ jsxRuntimeExports.jsx("em", { className: "warn", "aria-label": "unsaved changes", children: " •" })
          ]
        },
        t.id
      );
    }) }),
    loading && !config && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "drawer-body", children: /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "loading…" }) }),
    error && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "drawer-body", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", children: [
      "⚠ ",
      error
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "drawer-body", children: [
      tab === "connection" && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Backend" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Currently talking to ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: backendLabel() }),
          ". The dashboard API can run on any machine — leave this empty to use the origin that served this page."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "API base URL" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              placeholder: `${location.host} (this origin)`,
              value: base,
              onChange: (e) => setBase(e.target.value),
              onBlur: (e) => setBase(normalize(e.target.value) || e.target.value)
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Auth token (this browser)" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "password",
              placeholder: "only if the server requires one",
              value: token,
              onChange: (e) => setToken(e.target.value)
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: applyConnection, children: "connect & reload" }),
          (base || token) && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => {
            setBase("");
            setToken("");
            setConnVerdict(null);
          }, children: "clear" })
        ] }),
        connVerdict && connVerdict.kind !== "ok" && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", role: "alert", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: connVerdict.kind === "unparseable" ? "That address cannot be dialled" : "Send this token there?" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: connVerdict.detail }),
          needsConfirm(connVerdict) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost danger", onClick: () => {
              setConnVerdict(null);
              goConnect(token);
            }, children: "send it anyway" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: () => {
              setToken("");
              setConnVerdict(null);
              goConnect("");
            }, children: "alternative" in connVerdict ? connVerdict.alternative : "connect without a token" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setConnVerdict(null), children: "cancel" })
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Tip: ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "?backend=https://robot.lan:8080&token=…" }),
          " in the URL sets both, so a bookmark or QR code points a phone straight at one robot."
        ] })
      ] }),
      tab === "agent" && config && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Fleet agent" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          config.agent.built ? "built" : "not built yet",
          " ·",
          " ",
          config.agent.busy ? "busy" : "idle",
          " · ",
          config.agent.messages ?? 0,
          " messages · tools: ",
          (config.agent.tools ?? []).join(", ") || "fleet",
          config.agent.bridge_online === false && " · ⚠ mesh bridge not attached"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Model id" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              list: "known-models",
              value: modelId,
              placeholder: "(provider default)",
              onChange: (e) => setModelId(e.target.value)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("datalist", { id: "known-models", children: config.agent.known_models.map((m) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: m }, m)) }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "agent.model_id", raw: modelId })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Temperature" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                step: "0.1",
                min: "0",
                max: "2",
                value: temperature,
                placeholder: "default",
                onChange: (e) => setTemperature(e.target.value)
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "agent.temperature", raw: temperature })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Max tokens" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                min: "1",
                value: maxTokens,
                placeholder: "default",
                onChange: (e) => setMaxTokens(e.target.value)
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "agent.max_tokens", raw: maxTokens })
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
            "System prompt ",
            config.agent.is_default_prompt && /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "(default)" })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("textarea", { rows: 10, value: prompt, onChange: (e) => setPrompt(e.target.value) })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn go",
              disabled: saving || !agentValid,
              title: agentValid ? void 0 : "fix the highlighted fields first",
              onClick: () => apply({
                agent: {
                  model_id: modelId || null,
                  system_prompt: prompt,
                  temperature: temperature === "" ? null : Number(temperature),
                  max_tokens: maxTokens === "" ? null : Number(maxTokens)
                }
              }),
              children: "save"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost",
              disabled: saving,
              onClick: () => apply({ reset_prompt: true }),
              children: "reset prompt"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost",
              disabled: saving,
              onClick: () => apply({ reset_agent: true, clear_history: true }),
              children: "clear conversation"
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Model and prompt changes take effect on the next turn. Sampling knobs are applied to the resolved model, so a provider that ignores one will say so in the log." })
      ] }),
      tab === "voice" && config && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Voice" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Provider" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("select", { value: voiceProvider, onChange: (e) => setVoiceProvider(e.target.value), children: config.voice.providers.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: p, children: p }, p)) }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "voice.provider", raw: voiceProvider })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Voice name" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: voiceName,
              placeholder: "provider default",
              onChange: (e) => setVoiceName(e.target.value)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "voice.voice_name", raw: voiceName })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", disabled: saving, onClick: () => apply({
          voice: { provider: voiceProvider, voice_name: voiceName || null }
        }), children: "save" }) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Each provider needs its own credential in the Env tab (",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "OPENAI_API_KEY" }),
          ", ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "GOOGLE_API_KEY" }),
          ", or AWS for Nova Sonic)."
        ] })
      ] }),
      tab === "mesh" && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Mesh" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("dl", { className: "kv", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "status" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { children: mesh.online === false ? "offline" : mesh.online ? `online as ${mesh.peer_id}` : "not reported yet" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "peers" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("dd", { children: [
            mesh.live_peers ?? 0,
            " live / ",
            mesh.peers ?? 0,
            " known"
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "wire security" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: mesh.local_dev ? "bad" : "ok", children: mesh.wire_security ?? "unknown" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "backend" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { children: mesh.backend ?? "zenoh" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "cmd cap" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("dd", { children: [
            mesh.max_cmd_bytes ?? 0,
            " B"
          ] }),
          ((_a = mesh.policy_allow) == null ? void 0 : _a.length) ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "policy allowlist" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "mono", children: mesh.policy_allow.join(", ") })
          ] }) : null
        ] }),
        mesh.local_dev && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "explain", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: 'What "wire security off" means:' }),
          " robot commands and camera frames travel the mesh unencrypted and unauthenticated (",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "STRANDS_MESH_LOCAL_DEV=1" }),
          "). That is fine on a trusted home LAN. Before this network is shared or bridged, restart the dashboard ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "without" }),
          " that env var so the mesh requires mTLS — then only ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "tls/" }),
          " and ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "quic/" }),
          " endpoints are accepted. This is separate from dashboard login, which already guards the web UI."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", disabled: saving, title: "Fills the fields below - nothing is saved until you click save & re-point", onClick: () => {
          setMeshPort("");
          setConnect("");
          setListen("");
          setMeshBackend("");
          setCameraHz("15");
          setStatus('SO-101 preset filled in below - review, then "save & re-point"');
        }, children: "✦ recommended for SO-101 desk setup" }) }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Preset: multicast discovery (no endpoints), default port 7447, camera 15 Hz — smooth preview for two USB arm cams without flooding the LAN. It only fills the form; nothing applies until you save." }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Connect endpoints" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              placeholder: "tls/robot.lan:7447, tls/10.0.0.5:7447",
              value: connect,
              onChange: (e) => setConnect(e.target.value)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "mesh.connect", raw: connect })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Listen endpoints" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              placeholder: "tls/0.0.0.0:7447",
              value: listen,
              onChange: (e) => setListen(e.target.value)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "mesh.listen", raw: listen })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Port" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                value: meshPort,
                placeholder: "7447",
                onChange: (e) => setMeshPort(e.target.value)
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "mesh.port", raw: meshPort })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Transport" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: meshBackend, onChange: (e) => setMeshBackend(e.target.value), children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "zenoh (default)" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "iot", children: "AWS IoT Core" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "bridge", children: "bridge" })
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Camera Hz" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                step: "1",
                value: cameraHz,
                placeholder: "default",
                onChange: (e) => setCameraHz(e.target.value)
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(FieldMeta, { k: "mesh.camera_hz", raw: cameraHz })
          ] })
        ] }),
        !mesh.local_dev && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "With mTLS wire security, only ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "tls/" }),
          " and ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "quic/" }),
          " endpoints are accepted — a ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "tcp/" }),
          " endpoint is refused at session open."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn go",
              disabled: saving || !meshValid,
              title: meshValid ? void 0 : "fix the highlighted fields first",
              onClick: () => apply({
                mesh: {
                  connect,
                  listen,
                  port: meshPort === "" ? null : Number(meshPort),
                  backend: meshBackend || null,
                  camera_hz: cameraHz === "" ? null : Number(cameraHz)
                },
                restart_mesh: true
              }),
              children: "save & re-point"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", disabled: saving, onClick: () => restartMesh(false), children: "restart mesh" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Re-pointing re-opens the shared mesh session. Locally spawned robots hold their own reference to the old one, so the server refuses unless they are despawned — or you force it and accept that they stay on the old endpoints." }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost danger", disabled: saving, onClick: () => restartMesh(true), children: "force restart (orphan local robots)" }) })
      ] }),
      tab === "env" && config && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Environment" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Written to ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: config.env_file }),
          " (chmod 600). Secrets show masked; leaving a mask untouched leaves the stored value alone.",
          " ",
          "Clearing a value stores an ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "empty" }),
          " value; use ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "unset" }),
          " to remove the variable entirely."
        ] }),
        config.env.some((r) => r.shadowed) && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint warn", children: [
          config.env.filter((r) => r.shadowed).map((r) => r.key).join(", "),
          " ",
          config.env.filter((r) => r.shadowed).length > 1 ? "were" : "was",
          " exported into this process before it started, and that value wins over .env for as long as it runs — saving here updates the file, not this run."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "envlist", children: config.env.map((row) => /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field env", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
            row.key,
            row.secret && /* @__PURE__ */ jsxRuntimeExports.jsx("em", { title: "masked on read", children: " 🔒" }),
            !row.in_file && row.set && /* @__PURE__ */ jsxRuntimeExports.jsx("em", { title: "from the process environment", children: " (env)" }),
            row.shadowed && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "warn", title: "this process was launched with a different value, which wins over .env until it is restarted without it", children: [
              " ",
              "(shell overrides .env)"
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: envDraft[row.key] ?? row.value,
              placeholder: row.set ? "" : "not set",
              disabled: envUnset.includes(row.key),
              onChange: (e) => setEnvDraft((d) => ({ ...d, [row.key]: e.target.value }))
            }
          ),
          row.in_file && (envUnset.includes(row.key) ? /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "warn", children: [
            "will be removed on save —",
            " ",
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn ghost tiny",
                onClick: () => setEnvUnset((u) => u.filter((k) => k !== row.key)),
                children: "keep it"
              }
            )
          ] }) : /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost tiny",
              title: "remove this variable from the file",
              onClick: () => setEnvUnset((u) => [...u, row.key]),
              children: "unset"
            }
          ))
        ] }, row.key)) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "New key" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                value: newKey,
                placeholder: "MY_API_KEY",
                onChange: (e) => setNewKey(e.target.value.toUpperCase())
              }
            ),
            envKeyError(newKey) && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "field-err", role: "alert", children: [
              "⚠ ",
              envKeyError(newKey)
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Value" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("input", { value: newValue, onChange: (e) => setNewValue(e.target.value) }),
            envValueError(newValue) && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "field-err", role: "alert", children: [
              "⚠ ",
              envValueError(newValue)
            ] })
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field check", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("input", { type: "checkbox", checked: trustRemote, onChange: (e) => setTrustRemote(e.target.checked) }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
            "Allow HuggingFace ",
            /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "trust_remote_code" }),
            " (lerobot_local, kimodo) — executes code from the model repo"
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "btn go",
            disabled: saving || !envValid,
            title: envValid ? void 0 : "fix the highlighted fields first",
            onClick: () => {
              const env = { ...envDraft };
              for (const k of envUnset) env[k] = null;
              if (newKey.trim()) env[newKey.trim()] = newValue;
              void apply({ env, runtime: { trust_remote_code: trustRemote } });
              setNewKey("");
              setNewValue("");
              setEnvUnset([]);
            },
            children: "save"
          }
        ) })
      ] }),
      tab === "security" && config && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Security" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: config.security.auth_enabled ? "result ok" : "result bad", children: config.security.auth_enabled ? "✓ a token is required on /api and /ws" : "⚠ no auth: anyone who can reach this port can move motors" }),
        ((_b = config.security.notice) == null ? void 0 : _b.text) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result warn", "data-notice": config.security.notice.kind, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
            "⚠ ",
            config.security.notice.text
          ] }),
          config.security.notice.remedy && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", children: config.security.notice.remedy })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Server auth token" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "password",
              value: serverToken,
              placeholder: config.security.auth_enabled ? "•••••• (set)" : "not set",
              onChange: (e) => setServerToken(e.target.value)
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "CORS origins" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: corsOrigins,
              placeholder: "* (any origin)",
              onChange: (e) => setCorsOrigins(e.target.value)
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", disabled: saving, onClick: async () => {
            await apply({ security: { auth_token: serverToken || null, cors_origins: corsOrigins } });
            if (serverToken) {
              setAuthToken(serverToken);
              setToken(serverToken);
            }
          }, children: "save" }),
          config.security.auth_enabled && !removeArmed && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost danger",
              disabled: saving,
              onClick: () => setRemoveArmed(true),
              children: "remove token"
            }
          )
        ] }),
        config.security.auth_enabled && removeArmed && (() => {
          const w = authRemovalWarning({
            host: typeof location !== "undefined" ? location.hostname : "",
            corsOrigins,
            // the mesh panel's own number: live peers, not merely remembered ones
            peerCount: mesh.live_peers ?? mesh.peers ?? 0
          });
          return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: w.severity === "exposed" ? "result bad" : "result", role: "alert", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "Remove the token?" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { children: w.lines.map((l) => /* @__PURE__ */ jsxRuntimeExports.jsx("li", { children: l }, l)) }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost danger", disabled: saving, onClick: () => {
                setRemoveArmed(false);
                void apply({ security: { auth_token: null } });
              }, children: w.confirmLabel }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "button",
                {
                  className: "btn ghost",
                  disabled: saving,
                  onClick: () => setRemoveArmed(false),
                  children: "keep it"
                }
              )
            ] })
          ] });
        })(),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "CORS origins apply at startup; the token applies immediately (and is saved into this browser so you are not locked out). ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "STRANDS_MESH_LOCAL_DEV=1" }),
          " is separate and disables mesh ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "wire" }),
          " security — see the Mesh tab."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(ConsentSettings, {}),
        /* @__PURE__ */ jsxRuntimeExports.jsx(PasskeyList, { authRequired: Boolean((_c = config == null ? void 0 : config.security) == null ? void 0 : _c.auth_enabled) })
      ] })
    ] }),
    status && /* @__PURE__ */ jsxRuntimeExports.jsx("footer", { className: "drawer-foot", children: status })
  ] }) });
}
const str = (v) => typeof v === "string" ? v : "";
function detailOf(row) {
  const d = row.detail;
  return d && typeof d === "object" && !Array.isArray(d) ? d : {};
}
function noAnswer(row) {
  const d = detailOf(row);
  if (str(d.state) === "no_answer" || d.answered === false) return true;
  return /"?state"?\s*[:=]\s*"?no_answer/.test(str(row.result) + str(row.detail));
}
function activityLine(row) {
  const d = detailOf(row);
  const isEstop = row.action === "estop" || row.action === "emergency_stop";
  const target = row.target && row.target.trim() ? row.target : isEstop ? "all peers" : "—";
  if (row.ok === false) {
    return { tone: "bad", glyph: "✗", title: "the call failed", note: "", target };
  }
  if (isEstop) {
    const acks = typeof d.responses_received === "number" ? d.responses_received : null;
    const notStopped = Array.isArray(d.peers_not_stopped) ? d.peers_not_stopped : [];
    const lockout = d.lockout_engaged === true;
    const by = str(d.peer_id);
    const bits = [];
    if (by) bits.push(`issued by ${by}`);
    if (lockout) bits.push("lockout engaged");
    if (acks === 0) bits.push("no peer acknowledged");
    else if (acks != null) bits.push(`${acks} peer${acks === 1 ? "" : "s"} acknowledged`);
    if (notStopped.length) bits.push(`${notStopped.length} did NOT stop`);
    const unproven = acks === 0 || notStopped.length > 0;
    return {
      tone: unproven ? "warn" : "bad",
      glyph: unproven ? "⚠" : "■",
      title: unproven ? "emergency stop was broadcast but no peer confirmed stopping - the stop is unproven" : "emergency stop: the fleet was stopped",
      note: bits.join(" · "),
      target
    };
  }
  if (noAnswer(row)) {
    return {
      tone: "warn",
      glyph: "⚠",
      title: "the command was sent but the robot never answered - the effect is unknown",
      note: "robot did not answer",
      target
    };
  }
  if (row.ok === true) {
    return { tone: "ok", glyph: "✓", title: "the call completed", note: "", target };
  }
  return { tone: "pending", glyph: "…", title: "no verdict recorded yet", note: "", target };
}
function activityAnnouncement(latest, sinceT) {
  if (!latest || !(latest.t > sinceT)) return "";
  const v = activityLine(latest);
  const lead = v.tone === "bad" ? "failed — " : v.tone === "warn" ? "warning — " : "";
  const where = v.target && v.target !== "—" ? ` on ${v.target}` : "";
  const note = v.note ? ` ${v.note}` : "";
  return `${lead}${latest.source} ${latest.action}${where}: ${v.title}${note}`.replace(/\s+/g, " ").trim();
}
const SOURCE_ICON = {
  api: "🖥",
  agent: "🤖",
  estop: "🛑",
  safety: "🛑",
  mesh: "🔗",
  voice: "🎙",
  training: "🎓",
  record: "🎬",
  resume: "🟢"
};
function ago(t, now) {
  const s = Math.max(0, now - t);
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(0)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}
function ActivityLog({ live, open, onClose }) {
  const [history2, setHistory] = reactExports.useState([]);
  const sheetRef = reactExports.useRef(null);
  useDialogFocus(sheetRef, open);
  const [filter, setFilter] = reactExports.useState("all");
  const [now, setNow] = reactExports.useState(() => Date.now() / 1e3);
  const openedAt = reactExports.useRef(0);
  reactExports.useEffect(() => {
    if (open) openedAt.current = Date.now() / 1e3;
  }, [open]);
  const newest = live.reduce(
    (best, e) => best && best.t >= e.t ? best : e,
    void 0
  );
  reactExports.useEffect(() => {
    if (!open) return;
    void api("/api/activity?limit=200").then((r) => setHistory(r.activity ?? [])).catch(() => {
    });
  }, [open]);
  reactExports.useEffect(() => {
    if (!open) return;
    const id = setInterval(() => setNow(Date.now() / 1e3), 1e3);
    return () => clearInterval(id);
  }, [open]);
  const entries = reactExports.useMemo(() => {
    const seen = /* @__PURE__ */ new Set();
    const merged = [];
    for (const e of [...live, ...history2]) {
      const key = `${e.t}|${e.source}|${e.action}|${e.target}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(e);
    }
    merged.sort((a, b) => b.t - a.t);
    return filter === "all" ? merged : merged.filter((e) => e.source === filter);
  }, [live, history2, filter]);
  if (!open) return null;
  const sources = Array.from(new Set(entries.map((e) => e.source)));
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "drawer-backdrop", onClick: onClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("aside", { ref: sheetRef, className: "drawer wide", onClick: (e) => e.stopPropagation(), children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("header", { className: "drawer-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "Activity" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, "aria-label": "close the activity log", title: "Escape", children: "✕" })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("nav", { className: "tabs", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: filter === "all" ? "tab on" : "tab",
          "aria-pressed": filter === "all",
          onClick: () => setFilter("all"),
          children: [
            "all (",
            entries.length,
            ")"
          ]
        }
      ),
      sources.map((s) => /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "button",
        {
          className: filter === s ? "tab on" : "tab",
          "aria-pressed": filter === s,
          onClick: () => setFilter(s),
          children: [
            SOURCE_ICON[s] ?? "•",
            " ",
            s
          ]
        },
        s
      ))
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "drawer-body", children: [
      entries.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Nothing yet. Every task, stop, e-stop, spawn, recording session and training job — from this UI, the agent, or voice — lands here with what the robot answered." }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sr-only", role: "status", "aria-live": "polite", "aria-atomic": "true", children: activityAnnouncement(newest, openedAt.current) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "activity", role: "log", "aria-label": "activity — every command that left this dashboard", "aria-live": "off", children: entries.map((e, i) => /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { className: activityLine(e).tone, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "when", title: new Date(e.t * 1e3).toLocaleString(), children: ago(e.t, now) }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: SOURCE_ICON[e.source] ?? "•" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "what", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: e.action }),
          " → ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: activityLine(e).target }),
          e.elapsed != null && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { children: [
            " ",
            e.elapsed.toFixed(1),
            "s"
          ] }),
          activityLine(e).note && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "actnote", children: [
            " ",
            activityLine(e).note
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "verdict", title: activityLine(e).title, children: activityLine(e).glyph }),
        (e.detail != null || e.result) && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "what the robot answered" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: [
            e.detail == null ? null : typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail, null, 2),
            // `result` is already a truncated JSON string from the server.
            e.result || null
          ].filter(Boolean).join("\n") })
        ] })
      ] }, `${e.t}-${i}`)) })
    ] })
  ] }) });
}
function boardListEmptyLine(opts) {
  const err = (opts.error ?? "").trim();
  if (!opts.scanned) {
    if (err) {
      return {
        kind: "unscanned",
        // Never "no boards": nothing was asked successfully, so nothing is known.
        message: `the device scan failed (${err}) — this list is empty because nothing answered, not because nothing is plugged in`
      };
    }
    return { kind: "scanning", message: "scanning USB for servo boards…" };
  }
  const suffix = err ? ` (the last refresh also reported: ${err})` : "";
  return {
    kind: "detected",
    message: `no servo board detected — nothing on USB enumerated as a serial bus${suffix}`
  };
}
function unanswered(what, error) {
  const err = (error ?? "").trim();
  if (err) {
    return {
      kind: "unscanned",
      message: `the device scan failed (${err}) — this list is empty because nothing answered, not because there ${what}`
    };
  }
  return { kind: "scanning", message: "scanning this machine…" };
}
function managedListEmptyLine(opts) {
  if (!opts.scanned) return unanswered("are none", opts.error);
  return {
    kind: "detected",
    message: "None. Spawning one starts a child process that joins the mesh as its own peer — use it for a MuJoCo sim, or to drive a real arm from this machine."
  };
}
function cameraGridEmptyLine(opts) {
  if (!opts.scanned) return unanswered("are no cameras", opts.error);
  return {
    kind: "detected",
    message: "No camera index answered a probe — plug one in, or rescan if you just did."
  };
}
function hardwareSummaryValue(opts) {
  const items = opts.items.filter(Boolean);
  if (items.length > 0) return items.join(", ");
  const err = (opts.error ?? "").trim();
  if (!opts.scanned) return err ? `unknown — the scan failed (${err})` : "unknown — still scanning";
  return `none ${opts.emptyNote}`.trim();
}
const NAME_SHAPE = /^[A-Za-z][A-Za-z0-9_]*$/;
const RESERVED_NAMES = /* @__PURE__ */ new Set(["device_name"]);
function camerasField(rows, shared = {}) {
  const filled = (rows ?? []).filter((r) => (r.index ?? "").trim() !== "");
  if (filled.length === 0) return { value: null, problem: null, note: null };
  const seenNames = /* @__PURE__ */ new Set();
  const seenIndices = /* @__PURE__ */ new Map();
  const value = {};
  let note = null;
  for (const row of filled) {
    const name = (row.name ?? "").trim();
    if (!name) {
      return { value: null, problem: "every selected camera needs a name — main, wrist, top…", note: null };
    }
    if (!NAME_SHAPE.test(name)) {
      return {
        value: null,
        problem: `“${name}” cannot be a camera name — letters, digits and _ only, starting with a letter`,
        note: null
      };
    }
    const key = name.toLowerCase();
    if (RESERVED_NAMES.has(key)) {
      return {
        value: null,
        problem: `“${name}” is the dashboard's own bookkeeping key, not a camera name — pick main, wrist, top…`,
        note: null
      };
    }
    if (seenNames.has(key)) {
      return { value: null, problem: `two cameras named “${name}” — the second would overwrite the first`, note: null };
    }
    seenNames.add(key);
    const index = Number(row.index);
    if (!Number.isInteger(index) || index < 0) {
      return { value: null, problem: `“${row.index}” is not a camera index`, note: null };
    }
    const claimant = seenIndices.get(index);
    if (claimant !== void 0) {
      note = `index ${index} is claimed by both “${claimant}” and “${name}” — the second open usually fails at spawn`;
    }
    seenIndices.set(index, name);
    value[name] = {
      index_or_path: index,
      ...shared.fps ? { fps: shared.fps } : {},
      ...shared.width ? { width: shared.width } : {},
      ...shared.height ? { height: shared.height } : {}
    };
  }
  return { value, problem: null, note };
}
function spawnNotice(body) {
  if (!body || typeof body !== "object") return null;
  const raw = body.calibration_warning;
  if (typeof raw !== "string") return null;
  const text = raw.trim();
  if (!text) return null;
  return { text, tone: "warn" };
}
function deviceActionFailure(input) {
  const why2 = String(input.message ?? "").trim() || "no detail";
  const spawn = input.kind === "spawn";
  if (refusedBeforeActing(input.status)) {
    return {
      text: spawn ? `✗ refused (${input.status}): ${why2} — no process was started, nothing new is holding the serial port.` : `✗ refused (${input.status}): ${why2} — the robot was NOT stopped; it is still running (and still recording, if it was).`,
      ambiguous: false
    };
  }
  const head = Number(input.status ?? 0) ? `⚠ unknown — the server failed mid-request (${input.status}: ${why2})` : `⚠ unknown — no answer came back (${why2})`;
  return {
    text: spawn ? `${head}: the robot MAY have started and a process MAY already hold that port. Refreshing the device list — if it appears there, it started. Spawning again could put a second process on the same bus, which is the "Port is in use!" collision.` : `${head}: the robot MAY already have been killed — if it was mid-episode, that take is gone. Refreshing the device list — if it disappears from it, the despawn landed.`,
    ambiguous: true
  };
}
function plain(s) {
  return s.replace(/[*`]/g, "").trim();
}
function idProblem(id) {
  const bare = id.trim();
  if (!bare) return "this calibration has no id — the file name is empty";
  if (/^(none|null|undefined|nan)$/i.test(bare)) {
    return `"${bare}" is a missing value that reached a file name — something was spawned without a robot id, so these joint limits belong to that accident, not to an arm. Recalibrate under a real id and delete this file.`;
  }
  return void 0;
}
function parseCalibrationList(text) {
  const out = { entries: [] };
  if (!text) return out;
  let deviceType = "";
  let model = "";
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const loc = /^Location:\s*(.+)$/.exec(line.trim());
    if (loc) {
      out.location = plain(loc[1]);
      continue;
    }
    const type = /^##\s+(.+)$/.exec(line);
    if (type && !line.startsWith("###")) {
      deviceType = plain(type[1]).toLowerCase();
      model = "";
      continue;
    }
    const mod = /^###\s+(.+?)(?:\s+\(\d+\s+calibrations?\))?\s*$/.exec(line);
    if (mod) {
      model = plain(mod[1]);
      continue;
    }
    const item = /^\s*-\s+`([^`]+)`\s*(?:\*\((.*)\)\*)?\s*$/.exec(line);
    if (item) {
      const id = item[1];
      const meta = (item[2] ?? "").trim();
      const entry = {
        deviceType,
        model,
        id,
        unreadable: /error reading file/i.test(meta)
      };
      const bad = idProblem(id);
      if (bad) entry.problem = bad;
      if (meta && !entry.unreadable) {
        const when = /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})/.exec(meta);
        if (when) entry.modified = when[1];
        const size = /([\d.]+)\s*KB/i.exec(meta);
        if (size) entry.sizeKb = Number(size[1]);
        const motors = /(\d+)\s*motors?/i.exec(meta);
        if (motors) entry.motors = Number(motors[1]);
      }
      out.entries.push(entry);
    }
  }
  return out;
}
function parseCalibrationDetail(text) {
  const out = { motors: [] };
  if (!text) return out;
  let current = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const title = /^\*\*Calibration Details:\s*(.+?)\*\*$/.exec(line);
    if (title) {
      out.title = plain(title[1]);
      continue;
    }
    const path = /^\*\*Path:\*\*\s*(.+)$/.exec(line);
    if (path) {
      out.path = plain(path[1]);
      continue;
    }
    const modified = /^\*\*Modified:\*\*\s*(.+)$/.exec(line);
    if (modified) {
      out.modified = plain(modified[1]);
      continue;
    }
    const size = /^\*\*Size:\*\*\s*(.+)$/.exec(line);
    if (size) {
      out.size = plain(size[1]);
      continue;
    }
    const motor = /^###\s+(.+)$/.exec(line);
    if (motor) {
      current = { name: plain(motor[1]) };
      out.motors.push(current);
      continue;
    }
    if (!current) continue;
    const field = /^-\s+\*\*([^:]+):\*\*\s*(.*)$/.exec(line);
    if (!field) continue;
    const key = field[1].trim().toLowerCase();
    const value = plain(field[2]);
    if (key === "id") current.id = value;
    else if (key === "drive mode") current.driveMode = value;
    else if (key === "homing offset") current.homingOffset = value;
    else if (key === "range") {
      const range = /^(.*?)\s+to\s+(.*)$/.exec(value);
      if (range) {
        current.rangeMin = range[1].trim();
        current.rangeMax = range[2].trim();
      } else current.rangeMin = value;
    }
  }
  return out;
}
const show = (v) => v === null || v === void 0 ? void 0 : String(v);
const label = (e) => `${e.deviceType}/${e.model}/${e.id}`;
function CalibrationSection() {
  const [entries, setEntries] = reactExports.useState(null);
  const [location2, setLocation] = reactExports.useState(null);
  const [error, setError] = reactExports.useState(null);
  const [sel, setSel] = reactExports.useState(null);
  const load2 = reactExports.useCallback(async () => {
    try {
      const r = await api("/api/calibration");
      const parsed = parseCalibrationList(r.text ?? "");
      if (r.status && r.status !== "success" && parsed.entries.length === 0) {
        setError(r.text || "calibration list failed");
        setEntries([]);
        return;
      }
      setEntries(parsed.entries);
      setLocation(parsed.location ?? null);
      setError(null);
    } catch (e) {
      setError((e == null ? void 0 : e.message) ?? String(e));
      setEntries([]);
    }
  }, []);
  reactExports.useEffect(() => {
    void load2();
  }, [load2]);
  const select = async (entry) => {
    var _a, _b;
    if (sel && label(sel.entry) === label(entry)) {
      setSel(null);
      return;
    }
    setSel({ entry, loading: true });
    const path = `/api/calibration/${encodeURIComponent(entry.id)}?device_type=${encodeURIComponent(entry.deviceType)}&device_model=${encodeURIComponent(entry.model)}`;
    try {
      const r = await api(path);
      const detail = ((_a = r.motors) == null ? void 0 : _a.length) ? {
        title: (_b = r.text) == null ? void 0 : _b.split("\n")[0],
        path: r.path,
        modified: r.modified,
        motors: r.motors.map((m) => ({
          name: m.name,
          id: show(m.id),
          driveMode: show(m.drive_mode),
          homingOffset: show(m.homing_offset),
          rangeMin: show(m.range_min),
          rangeMax: show(m.range_max)
        }))
      } : parseCalibrationDetail(r.text ?? "");
      if (r.status === "success" && detail.motors.length > 0) {
        setSel({ entry, loading: false, detail });
      } else {
        setSel({ entry, loading: false, detail, problem: r.text || "no detail returned" });
      }
    } catch (e) {
      setSel({ entry, loading: false, problem: (e == null ? void 0 : e.message) ?? String(e) });
    }
  };
  const groups = [];
  for (const e of entries ?? []) {
    const key = `${e.deviceType}/${e.model}`;
    const last = groups[groups.length - 1];
    if (last && last.key === key) last.rows.push(e);
    else groups.push({ key, deviceType: e.deviceType, model: e.model, rows: [e] });
  }
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
      "Calibration ",
      entries ? /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: entries.length }) : null,
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void load2(), children: "reload" })
    ] }),
    error && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", children: [
      "⚠ ",
      error
    ] }),
    entries === null && !error && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "reading calibration files…" }),
    entries !== null && entries.length === 0 && !error && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
      "No calibration files on this machine",
      location2 ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        " under ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: location2 })
      ] }) : null,
      ". An arm with no calibration reports raw servo counts, so its joint limits will be wrong — run ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "lerobot-calibrate" }),
      " in a terminal to create one."
    ] }),
    groups.map((g) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("h4", { className: "calibgroup", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "mono", children: g.model }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "meta", children: g.deviceType })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "devlist", children: g.rows.map((e) => {
        const open = !!sel && label(sel.entry) === label(e);
        return /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { className: e.unreadable || e.problem ? "dead" : "", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
            e.problem ? "⚠ " : "",
            e.id
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "meta", children: e.problem ? e.problem : e.unreadable ? "file unreadable" : [
            e.motors !== void 0 ? `${e.motors} motors` : null,
            e.modified ?? null,
            e.sizeKb !== void 0 ? `${e.sizeKb.toFixed(1)}KB` : null
          ].filter(Boolean).join(" · ") }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "devactions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void select(e), children: open ? "hide" : "view" }) }),
          open && sel && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "calibdetail", children: [
            sel.loading && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
              "loading ",
              label(e),
              "…"
            ] }),
            sel.detail && sel.detail.motors.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs("table", { className: "jointtable", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("thead", { children: /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "motor" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "id" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "drive" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "homing offset" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "range" })
                ] }) }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("tbody", { children: sel.detail.motors.map((m) => /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { children: m.name }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.id ?? "—" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.driveMode ?? "—" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.homingOffset ?? "—" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.rangeMin !== void 0 ? `${m.rangeMin}${m.rangeMax !== void 0 ? ` … ${m.rangeMax}` : ""}` : "—" })
                ] }, m.name)) })
              ] }),
              sel.detail.path && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint mono small", children: sel.detail.path })
            ] }),
            !sel.loading && sel.problem && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "⚠ the backend could not return per-joint values" }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "details" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: sel.problem })
              ] })
            ] })
          ] })
        ] }, label(e));
      }) })
    ] }, g.key)),
    entries !== null && entries.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
      "Read-only. Values come from ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "lerobot-calibrate" }),
      ", which moves the arm through its range, so it is a terminal job and not a dashboard button — this view only shows what it wrote",
      location2 ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        " under ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: location2 })
      ] }) : null,
      ". The ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "id" }),
      " above is what the spawn form's ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("i", { children: "Calibration id" }),
      " expects."
    ] })
  ] });
}
const STATE_LABEL = {
  ready: "ready",
  in_use: "in use",
  assigned: "assigned, no frames",
  blocked: "blocked by macOS",
  unreadable: "not responding",
  absent: "nothing here",
  /** Not 'nothing here': a camera WAS here. */
  vanished: "gone since we saw it",
  unknown: "not probed"
};
function CameraGallery({ cameras: cameras2, names, problem, scanned = true, error = null }) {
  const [previews, setPreviews] = reactExports.useState({});
  const [errors, setErrors] = reactExports.useState({});
  const [loading, setLoading] = reactExports.useState({});
  const urlsRef = reactExports.useRef({});
  reactExports.useEffect(() => () => {
    for (const url of Object.values(urlsRef.current)) URL.revokeObjectURL(url);
  }, []);
  const snap = async (index) => {
    setLoading((l) => ({ ...l, [index]: true }));
    setErrors(({ [index]: _gone, ...rest }) => rest);
    try {
      const url = await apiBlob(`/api/devices/camera/${index}/preview`);
      if (urlsRef.current[index]) URL.revokeObjectURL(urlsRef.current[index]);
      urlsRef.current[index] = url;
      setPreviews((p) => ({ ...p, [index]: url }));
    } catch (e) {
      setErrors((er) => ({ ...er, [index]: (e == null ? void 0 : e.message) ?? String(e) }));
    } finally {
      setLoading((l) => ({ ...l, [index]: false }));
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
    problem && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad camproblem", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
        "⚠ ",
        problem.message
      ] }),
      problem.remedy ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: problem.remedy }) : null
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camgrid", children: [
      cameras2.length === 0 && (() => {
        const line = cameraGridEmptyLine({ scanned, error });
        return /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", role: "status", children: line.message });
      })(),
      cameras2.map((c) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `camcard cam-${c.state ?? "ready"}${c.claimed_by ? " claimed" : ""}`, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camcard-head", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
            "index ",
            c.index
          ] }),
          c.state && c.state !== "ready" && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `campill campill-${c.state}`, children: STATE_LABEL[c.state] ?? c.state }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "meta", children: [
            c.width ? `${c.width}×${c.height}` : "",
            c.fps ? ` @ ${c.fps}fps` : "",
            c.geometry_from === "remembered" && c.width ? " (last seen)" : ""
          ] })
        ] }),
        c.name_hint && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camname-hint", title: "from the OS listing — the order is not OpenCV's, so treat it as a hint", children: [
          "probably ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: c.name_hint }),
          c.name_is_guess ? " · snap a preview to be sure" : ""
        ] }),
        c.claimed_by && c.state !== "assigned" ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camcard-body claimed-note", children: [
          "streaming for ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: c.claimed_by }),
          " — watch it on that robot's card",
          c.remedy ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: c.remedy }) : null
        ] }) : c.state && c.state !== "ready" ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camcard-body cam-why", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: c.reason }),
          c.remedy ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "hint", children: [
            "→ ",
            c.remedy
          ] }) : null,
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost campreview-btn",
              disabled: !!loading[c.index],
              onClick: () => void snap(c.index),
              title: "opens the camera once, right now — the server refuses only if a robot is really streaming it",
              children: loading[c.index] ? "trying…" : c.state === "assigned" ? "📷 identify it anyway" : c.state === "vanished" ? "📷 see which camera is here now" : "try anyway"
            }
          ),
          previews[c.index] && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "img",
            {
              src: previews[c.index],
              alt: `camera index ${c.index} snapshot`,
              onClick: () => void snap(c.index),
              title: "click to re-snap"
            }
          ),
          errors[c.index] && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camerr", children: [
            "⚠ ",
            errors[c.index]
          ] })
        ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camcard-body", children: [
          previews[c.index] ? /* @__PURE__ */ jsxRuntimeExports.jsx(
            "img",
            {
              src: previews[c.index],
              alt: `camera index ${c.index} snapshot`,
              onClick: () => void snap(c.index),
              title: "click to re-snap"
            }
          ) : /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost campreview-btn",
              disabled: !!loading[c.index],
              onClick: () => void snap(c.index),
              children: loading[c.index] ? "snapping…" : "📷 snap a preview"
            }
          ),
          errors[c.index] && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "camerr", children: [
            "⚠ ",
            errors[c.index]
          ] })
        ] })
      ] }, c.index))
    ] }),
    names.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "camnames", children: names.map((n) => /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "chip", title: "position in the OS device listing", children: n.name }, n.listing_index)) }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        "Attached cameras by name, in OS listing order — which is ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "not" }),
        " OpenCV index order. The snapshot is the identity: if you're unsure which index is which, look."
      ] })
    ] })
  ] });
}
function entryToRobot(value, keyName) {
  if (typeof value === "string") {
    const text = value.trim();
    if (keyName) return { name: keyName, label: text ? `${keyName} — ${text}` : keyName };
    return text ? { name: text, label: text } : null;
  }
  if (!value || typeof value !== "object") {
    return keyName ? { name: keyName, label: keyName } : null;
  }
  const o = value;
  const inner = typeof o.name === "string" && o.name.trim() ? o.name.trim() : void 0;
  const name = keyName ?? inner;
  if (!name) return null;
  const bits = [];
  if (keyName && inner && inner !== keyName) bits.push(inner);
  if (typeof o.category === "string" && o.category) bits.push(o.category);
  if (typeof o.joints === "number" && Number.isFinite(o.joints)) bits.push(`${o.joints} joints`);
  if (o.has_real === false && o.has_sim === true) bits.push("sim only");
  return { name, label: bits.length ? `${name} — ${bits.join(", ")}` : name };
}
function dedupe(rows) {
  const seen = /* @__PURE__ */ new Set();
  return rows.filter((r) => seen.has(r.name) ? false : (seen.add(r.name), true));
}
function normalizeRegistry(robots) {
  if (Array.isArray(robots)) {
    return dedupe(robots.map((r) => entryToRobot(r)).filter((r) => r !== null));
  }
  if (robots && typeof robots === "object") {
    return dedupe(Object.entries(robots).map(([k, v]) => entryToRobot(v, k)).filter((r) => r !== null));
  }
  return [];
}
function typeForRole(role) {
  return role === "follower" ? "robots" : "teleoperators";
}
function deviceModel(family, role) {
  const base = family.trim().toLowerCase();
  if (base.endsWith("_follower") || base.endsWith("_leader")) return base;
  return `${base}_${role}`;
}
function deviceId(facts, role) {
  const known = (facts.robot_id ?? "").trim();
  if (known) return known;
  const serial = (facts.serial_number ?? "").trim();
  return serial ? `${role}_${serial}` : role;
}
function idNameContradictsRole(id, role) {
  const other = role === "follower" ? "leader" : "follower";
  return id.trim().toLowerCase().includes(other);
}
function idNote(facts, role) {
  const known = (facts.robot_id ?? "").trim();
  if (!known) {
    const serial = (facts.serial_number ?? "").trim();
    return serial ? `this port has no spawn profile yet, so the id is built from its serial (${serial}) — spawn the arm with this same id afterwards, or lerobot will not find the calibration` : "this port has no spawn profile and reports no serial, so the id is just the role — two arms of the same role on one machine would overwrite each other";
  }
  const contradicts = idNameContradictsRole(known, role);
  const base = `this is the id the arm already runs with (${known}), so the calibration lands where it will be read`;
  if (!contradicts) return base;
  const volts = facts.role_volts != null ? `${facts.role_volts}V` : "its measured voltage";
  return `${base} — note the id is NAMED "${known}" while this bus measures ${volts} = ${role}. The id is only a file name and is still the right one to pass; the name is what is wrong, so do not let it convince you this is the other arm`;
}
function shellArg(v) {
  return /^[A-Za-z0-9_./:@=-]+$/.test(v) ? v : `'${v.replace(/'/g, `'\\''`)}'`;
}
function calibratePlan(facts, family) {
  const role = (facts.role ?? "").trim().toLowerCase();
  if (!role) {
    return {
      command: null,
      needsMeasurement: true,
      reason: "nobody has measured this bus yet, and the role decides both the model name and which directory the calibration is written to — measure the role first, then this command fills itself in"
    };
  }
  if (role === "unpowered") {
    return {
      command: null,
      needsMeasurement: true,
      reason: `this bus reads ${facts.role_volts ?? "~5.5"}V, which is the USB logic rail — the arm's power supply is off. Calibration has to move the arm, and an unpowered arm cannot hold position, so switch its supply on and measure again`
    };
  }
  if (role === "mixed") {
    return {
      command: null,
      reason: "the servos on this bus disagree about their voltage, which is a fault rather than a role — calibrating would record limits from a bus that is not answering consistently"
    };
  }
  if (role !== "follower" && role !== "leader") {
    return {
      command: null,
      needsMeasurement: true,
      reason: `the measurement came back "${role}", so the role is not established yet`
    };
  }
  const fam = (family ?? "").trim();
  if (!fam) {
    return {
      command: null,
      reason: "the arm family is unknown here (so101, so100, …) and it is half of the model name — spawn the arm or pick its type first, so the command names the real model"
    };
  }
  const deviceType = typeForRole(role);
  const model = deviceModel(fam, role);
  const id = deviceId(facts, role);
  const prefix = role === "follower" ? "robot" : "teleop";
  const command = `lerobot-calibrate --${prefix}.type=${model} --${prefix}.id=${shellArg(id)} --${prefix}.port=${shellArg(facts.device)}`;
  const measured2 = facts.role_volts != null ? `measured ${facts.role_volts}V on the servo bus` : `role recorded as ${role}`;
  return {
    command,
    deviceType,
    deviceModel: model,
    deviceId: id,
    idNote: idNote(facts, role),
    idWarn: idNameContradictsRole(id, role),
    reason: `${measured2}, so this arm is the ${role} — a ${role} is a lerobot "${deviceType}" device. Run this in a terminal: it will ask you to move the arm through its range by hand.`
  };
}
function knownCalibrationId(profiles, facts) {
  if (!profiles) return void 0;
  const serial = (facts.serial_number ?? "").trim();
  const bySerial = serial ? profiles[serial] : void 0;
  const id = ((bySerial == null ? void 0 : bySerial.robot_id) ?? "").trim();
  if (id) return id;
  for (const entry of Object.values(profiles)) {
    if (((entry == null ? void 0 : entry.port) ?? "").trim() === facts.device) {
      const pid = ((entry == null ? void 0 : entry.robot_id) ?? "").trim();
      if (pid) return pid;
    }
  }
  return void 0;
}
const FULL_TURN = "wrist_roll";
function wizardView(s) {
  const cancel = { key: "cancel", label: "cancel — nothing is saved", danger: true };
  switch (s.step) {
    case "starting":
      return {
        title: "starting…",
        body: "opening the arm — torque switches OFF as calibration begins, so the arm will go limp. Keep a hand near it.",
        buttons: [cancel],
        motors: null,
        unmoved: [],
        tone: "info",
        finished: false,
        detail: null
      };
    case "reuse":
      return {
        title: "a calibration for this id already exists",
        body: "keep the file that is already on disk, or redo the measurement from scratch. Keeping it writes the existing values to the motors and ends the wizard.",
        buttons: [
          { key: "c", label: "recalibrate from scratch", primary: true },
          { key: "enter", label: "keep the existing file" },
          cancel
        ],
        motors: null,
        unmoved: [],
        tone: "info",
        finished: false,
        detail: null
      };
    case "middle":
      return {
        title: "hold the arm at the middle of its range",
        body: "torque is off — the arm is limp and nothing here will move it. With your hand, put every joint near the MIDDLE of its travel (upright, elbow half bent, gripper half open), hold it there, and continue.",
        buttons: [{ key: "enter", label: "it's at the middle — continue", primary: true }, cancel],
        motors: null,
        unmoved: [],
        tone: "info",
        finished: false,
        detail: null
      };
    case "recording": {
      const motors = s.motors ?? [];
      const unmoved = motors.filter((m) => m.name !== FULL_TURN && m.min === m.max).map((m) => m.name);
      return {
        title: "recording — move every joint through its FULL range",
        body: "move each joint by hand to both of its limits, one at a time (wrist_roll is a full turn — lerobot handles it). The table is live: a row whose min equals its max has not moved yet, and lerobot refuses to save a joint it never saw move.",
        buttons: [{ key: "enter", label: "every joint has been to both limits — stop & save", primary: true }, cancel],
        motors,
        unmoved,
        tone: "info",
        finished: false,
        detail: null
      };
    }
    case "saved":
      return {
        title: "calibration saved ✓",
        body: s.path ? `written to ${s.path} — this is the file the arm loads at spawn, under the id you calibrated.` : "written — the arm loads it at spawn under the id you calibrated.",
        buttons: [{ key: "close", label: "done", primary: true }],
        motors: null,
        unmoved: [],
        tone: "ok",
        finished: true,
        detail: null
      };
    case "failed":
      return {
        title: "calibration did not finish",
        body: s.reason || `the run exited (${s.returncode ?? "unknown"}) before saving`,
        buttons: [{ key: "close", label: "close", primary: true }],
        motors: null,
        unmoved: [],
        tone: "bad",
        finished: true,
        detail: s.tail && s.tail.length ? s.tail.join("\n") : null
      };
  }
}
function confirmSheet(args) {
  return {
    title: `calibrate ${args.deviceId}`,
    body: `This runs lerobot-calibrate on ${args.port} (${args.model}, saved under the id "${args.deviceId}"). The moment it starts, torque switches OFF and the arm goes LIMP — hold it or let it rest safely. Nothing here commands motion: your hand does all the moving.`
  };
}
function CalibrateWizard({ port, role, model, deviceId: deviceId2, onSaved, onClose }) {
  const [phase, setPhase] = reactExports.useState("confirm");
  const [status, setStatus] = reactExports.useState(null);
  const [error, setError] = reactExports.useState(null);
  const [busy, setBusy] = reactExports.useState(false);
  const sid = reactExports.useRef(null);
  const savedTold = reactExports.useRef(false);
  reactExports.useEffect(() => {
    if (phase !== "running" || !sid.current) return;
    let stop = false;
    const tick = async () => {
      try {
        const s = await api(`/api/calibration/run/${sid.current}`);
        if (stop) return;
        setStatus(s);
        if (s.step === "saved" && !savedTold.current) {
          savedTold.current = true;
          onSaved == null ? void 0 : onSaved();
        }
      } catch (e) {
        if (!stop) setError((e == null ? void 0 : e.message) ?? String(e));
      }
    };
    void tick();
    const t = setInterval(() => {
      void tick();
    }, 600);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, [phase, onSaved]);
  const start = reactExports.useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await post("/api/calibration/run", { role, model, device_id: deviceId2, port });
      sid.current = s.id;
      savedTold.current = false;
      setStatus(s);
      setPhase("running");
    } catch (e) {
      setError((e == null ? void 0 : e.message) ?? String(e));
    } finally {
      setBusy(false);
    }
  }, [role, model, deviceId2, port]);
  const press = reactExports.useCallback(async (key) => {
    if (key === "close") {
      setPhase("closed");
      onClose == null ? void 0 : onClose();
      return;
    }
    if (!sid.current) return;
    setBusy(true);
    setError(null);
    try {
      const path = key === "cancel" ? `/api/calibration/run/${sid.current}/cancel` : `/api/calibration/run/${sid.current}/key`;
      const s = await post(path, key === "cancel" ? {} : { key });
      setStatus(s);
    } catch (e) {
      setError((e == null ? void 0 : e.message) ?? String(e));
    } finally {
      setBusy(false);
    }
  }, [onClose]);
  if (phase === "closed") return null;
  if (phase === "confirm") {
    const c = confirmSheet({ port, deviceId: deviceId2, model });
    return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "calibwizard", role: "dialog", "aria-label": c.title, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h4", { children: c.title }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: c.body }),
      error && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", children: [
        "⚠ ",
        error
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn", disabled: busy, onClick: () => void start(), children: busy ? "starting…" : "start — the arm goes limp now" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => {
          setPhase("closed");
          onClose == null ? void 0 : onClose();
        }, children: "not now" })
      ] })
    ] });
  }
  if (!status) return /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "starting the calibration session…" });
  const v = wizardView(status);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "calibwizard", role: "dialog", "aria-label": v.title, "aria-live": "polite", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("h4", { className: v.tone === "bad" ? "warn" : void 0, children: v.title }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: v.body }),
    v.motors && v.motors.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("table", { className: "jointtable", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("thead", { children: /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "joint" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "min" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "now" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("th", { children: "max" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("th", {})
      ] }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("tbody", { children: v.motors.map((m) => /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { className: v.unmoved.includes(m.name) ? "dead" : void 0, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("td", { children: m.name }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.min }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.pos }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("td", { className: "mono", children: m.max }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("td", { children: v.unmoved.includes(m.name) ? "has not moved yet" : "✓" })
      ] }, m.name)) })
    ] }),
    v.motors && v.motors.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "waiting for the first position read…" }),
    error && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", children: [
      "⚠ ",
      error
    ] }),
    v.detail && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "raw output" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { className: "logtail", children: v.detail })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row", children: v.buttons.map((b) => /* @__PURE__ */ jsxRuntimeExports.jsx(
      "button",
      {
        className: b.primary ? "btn" : b.danger ? "btn ghost danger" : "btn ghost",
        disabled: busy,
        onClick: () => void press(b.key),
        children: b.label
      },
      b.key
    )) })
  ] });
}
const norm = (s) => s.trim().toLowerCase();
const loose = (s) => norm(s).replace(/[\s_-]+/g, "");
const isRobotSide = (e) => {
  const t = norm(e.deviceType || "");
  return !t || t.startsWith("robot");
};
function familyMatches(entry, family) {
  const f = loose(family);
  const m = loose(entry.model || "");
  if (!f || !m) return true;
  return m.includes(f) || f.includes(m);
}
function calibrationVerdict(typed, entries, family = "") {
  const id = (typed ?? "").trim();
  if (entries === null || entries === void 0) {
    return {
      kind: "unchecked",
      warn: false,
      note: id ? "the calibration files could not be read, so this id was not checked" : ""
    };
  }
  const known = entries.filter((e) => (e.id || "").trim() !== "");
  if (!id) {
    return {
      kind: "none",
      warn: true,
      note: "no id: the arm starts uncalibrated and reports raw servo counts, so its joint limits will be wrong" + (known.length ? ` — calibrations on this machine: ${known.map((e) => e.id).join(", ")}` : "")
    };
  }
  const pick = (from) => from.find((e) => e.id === id) ?? from.find((e) => norm(e.id) === norm(id));
  const robotSide = known.filter(isRobotSide);
  const exact = pick(robotSide) ?? pick(known);
  if (exact && !isRobotSide(exact)) {
    const usable = robotSide.filter((e) => familyMatches(e, family)).map((e) => e.id);
    return {
      kind: "match",
      warn: true,
      // No one-tap suggestion here on purpose: choosing WHICH of the robot-side files this arm
      // should load is a decision about a physical arm's limits, not a typo fix.
      note: `${exact.id} was calibrated as a teleoperator (${exact.deviceType}/${exact.model}), and a robot in real mode loads robots/{type}/{id}.json — lerobot will refuse with "has no calibration registered" and the arm will report presence with no joints` + (usable.length ? ` — ids calibrated as robots here: ${usable.join(", ")}` : " — nothing on this machine is calibrated as a robot for this family yet")
    };
  }
  if (exact) {
    if (exact.unreadable) {
      return {
        kind: "match",
        warn: true,
        note: `${exact.id} exists but its file could not be read — lerobot will fail to load it`
      };
    }
    const wrongFamily = family && !familyMatches(exact, family);
    if (wrongFamily) {
      return {
        kind: "match",
        warn: true,
        // A real mismatch: same name, different robot. Wrong limits, silently.
        note: `${exact.id} was calibrated for ${exact.model}, not ${family} — the joint limits would come from the wrong arm`
      };
    }
    const detail = [exact.model, exact.motors ? `${exact.motors} motors` : ""].filter(Boolean).join(", ");
    return {
      kind: "match",
      warn: false,
      note: detail ? `matches ${exact.id} (${detail})` : `matches ${exact.id}`
    };
  }
  const near = known.find((e) => loose(e.id) === loose(id)) ?? known.find((e) => loose(e.id).startsWith(loose(id)) || loose(id).startsWith(loose(e.id)));
  if (near) {
    return {
      kind: "suggest",
      warn: true,
      suggestion: near.id,
      note: `no calibration named "${id}" — did you mean ${near.id}?`
    };
  }
  if (!known.length) {
    return {
      kind: "unknown",
      warn: true,
      note: `no calibration files exist on this machine, so "${id}" will not load — run lerobot-calibrate first, or spawn now and calibrate after`
    };
  }
  return {
    kind: "unknown",
    warn: true,
    note: `no calibration named "${id}" on this machine (found: ${known.map((e) => e.id).join(", ")}) — the arm would report raw servo counts`
  };
}
function nameClaimsOtherRole(name, role) {
  const n = (name ?? "").trim().toLowerCase();
  const r = (role ?? "").trim().toLowerCase();
  if (!n || r !== "follower" && r !== "leader") return false;
  const other = r === "follower" ? "leader" : "follower";
  return n.includes(other) && !n.includes(r);
}
function rememberedLine(r, facts = {}) {
  if (!r || !r.peer_id) return null;
  const role = (facts.role ?? "").trim().toLowerCase();
  const bits = [];
  if (r.robot_name) bits.push(String(r.robot_name));
  if (r.mode) bits.push(String(r.mode));
  bits.push(r.cameras.length ? `cameras ${r.cameras.join(" + ")}` : "no cameras");
  const line = {
    summary: `${r.peer_id}${bits.length ? " — " + bits.join(", ") : ""}`
  };
  if (r.robot_id) line.calibrationId = r.robot_id;
  const badId = nameClaimsOtherRole(r.robot_id, role);
  const badPeer = nameClaimsOtherRole(r.peer_id, role);
  if (badId || badPeer) {
    const volts = facts.role_volts != null ? `${facts.role_volts}V` : "its measured voltage";
    const named = badId ? `the calibration id is named "${r.robot_id}"` : `this peer is named "${r.peer_id}"`;
    line.warning = `${named} while this bus measures ${volts} = ${role}. The measurement is the fact and the name is what is wrong — reuse the memory anyway (the calibration file lives under that id), but do not let the name convince you this is the other arm`;
  }
  if (r.robot_id && facts.calibrations != null) {
    const v = calibrationVerdict(r.robot_id, facts.calibrations, r.robot_name ?? "");
    if (v.warn) {
      line.idProblem = `${v.note} — spawning this memory as it stands repeats that failure`;
    }
  }
  return line;
}
const MAX_NAMED = 3;
function ports(s) {
  const list = Array.isArray(s == null ? void 0 : s.serial_ports) ? s.serial_ports : [];
  return list.map((p) => String((p == null ? void 0 : p.device) ?? "").trim()).filter(Boolean);
}
function cameras(s) {
  const list = Array.isArray(s == null ? void 0 : s.cameras) ? s.cameras : [];
  return list.map((c) => (c == null ? void 0 : c.index) == null ? "" : `index ${c.index}`).filter(Boolean);
}
function plural(n, one, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}
function delta(kind, added, removed) {
  const out = [];
  for (const [sign, list] of [["+", added], ["−", removed]]) {
    if (!list.length) continue;
    const shown = list.slice(0, MAX_NAMED).join(", ");
    const more = list.length > MAX_NAMED ? ` +${list.length - MAX_NAMED} more` : "";
    out.push(`${sign}${plural(list.length, kind)} (${shown}${more})`);
  }
  return out;
}
function ageWords(beforeAtMs, nowMs) {
  if (!beforeAtMs || !nowMs || nowMs < beforeAtMs) return "";
  const secs = Math.round((nowMs - beforeAtMs) / 1e3);
  if (secs < 5) return "";
  if (secs < 90) return ` (${secs}s old)`;
  return ` (${Math.round(secs / 60)}min old)`;
}
function hardwareKey(s) {
  return `${ports(s).slice().sort().join("|")}#${cameras(s).slice().sort().join("|")}`;
}
function rescanReport(before, outcome, opts) {
  var _a;
  if (!outcome.ok) {
    const why2 = String(outcome.error ?? "").trim() || "the request failed";
    if (!before) {
      return { tone: "bad", stale: true, text: `⚠ rescan failed: ${why2} — nothing has been scanned yet, so the lists below are empty for that reason, not because this machine has no hardware.` };
    }
    const age = ageWords(opts == null ? void 0 : opts.beforeAtMs, opts == null ? void 0 : opts.nowMs);
    return {
      tone: "bad",
      stale: true,
      text: `⚠ rescan failed: ${why2} — the lists below are the PREVIOUS scan${age}, not what is plugged in now.`
    };
  }
  const after = outcome.after;
  const pAfter = ports(after);
  const cAfter = cameras(after);
  if (!pAfter.length && !cAfter.length) {
    const blocked = ((_a = after == null ? void 0 : after.camera_problem) == null ? void 0 : _a.kind) ? " Cameras are blocked on this machine, so the camera count says nothing about what is connected — see the camera notice below." : "";
    return {
      tone: "warn",
      stale: false,
      text: `scan completed: this machine reports no serial ports and no cameras.${blocked} If an arm is plugged in, check the cable and its power, then rescan.`
    };
  }
  if (before == null) {
    return {
      tone: "ok",
      stale: false,
      text: `scan completed — found: ${plural(pAfter.length, "serial port")}, ${plural(cAfter.length, "camera")}.`
    };
  }
  const pBefore = ports(before);
  const cBefore = cameras(before);
  const parts = [
    ...delta("serial port", pAfter.filter((d) => !pBefore.includes(d)), pBefore.filter((d) => !pAfter.includes(d))),
    ...delta("camera", cAfter.filter((d) => !cBefore.includes(d)), cBefore.filter((d) => !cAfter.includes(d)))
  ];
  if (!parts.length) {
    return {
      tone: "ok",
      stale: false,
      text: `scan completed — unchanged: ${plural(pAfter.length, "serial port")}, ${plural(cAfter.length, "camera")}.`
    };
  }
  return {
    tone: "ok",
    stale: false,
    text: `scan completed: ${parts.join(", ")} — now ${plural(pAfter.length, "serial port")}, ${plural(cAfter.length, "camera")}.`
  };
}
const PEER_NAME_RE = /^[A-Za-z0-9._:-]{1,64}$/;
function sanitizePeerName(raw) {
  const s = (raw ?? "").trim().replace(/\s+/g, "-").replace(/[^A-Za-z0-9._:-]/g, "").replace(/-{2,}/g, "-").replace(/^-+|-+$/g, "").slice(0, 64);
  return s === "" ? null : s;
}
function freeVariant$1(name, taken) {
  const m = name.match(/^(.*?)-(\d+)$/);
  const stem = m ? m[1] : name;
  let n = m ? Number(m[2]) : 1;
  for (let i = 0; i < 100; i += 1) {
    n += 1;
    const candidate = `${stem}-${n}`.slice(0, 64);
    if (!taken.has(candidate)) return candidate;
  }
  return null;
}
function peerNameField(raw, opts = {}) {
  const taken = new Set((opts.existing ?? []).filter(Boolean));
  const typed = (raw ?? "").trim();
  if (typed === "") {
    const family = (opts.robotName ?? "").trim() || "robot";
    const mode = (opts.mode ?? "").trim() || "sim";
    return {
      value: null,
      problem: null,
      note: `unnamed: the server will call it ${family}-${mode}-{clock} — name it now if you want to recognise it later, a peer cannot be renamed while it runs`,
      suggestion: null
    };
  }
  if (typed.length > 64) {
    return {
      value: null,
      problem: `that name is ${typed.length} characters; a peer id must be 1-64`,
      note: null,
      suggestion: sanitizePeerName(typed)
    };
  }
  if (!PEER_NAME_RE.test(typed)) {
    const clean = sanitizePeerName(typed);
    return {
      value: null,
      problem: "a name becomes a zenoh key segment, so only letters, digits and . _ : - are allowed" + (/[*/?#]/.test(typed) ? " — '*' and '/' there rewrite the fleet's key space instead of naming a peer" : ""),
      note: null,
      suggestion: clean && clean !== typed ? clean : null
    };
  }
  if (taken.has(typed)) {
    return {
      value: null,
      problem: `${typed} already exists — the server refuses a duplicate peer id rather than shadowing the one that is running`,
      note: null,
      suggestion: freeVariant$1(typed, taken)
    };
  }
  return { value: typed, problem: null, note: null, suggestion: null };
}
function portChoice(input) {
  const chosen = (input.chosen ?? "").trim();
  if (!chosen) return { kind: "empty" };
  const scanned = input.scanned !== false;
  const known = input.known ?? [];
  const claimed = input.claimed ?? [];
  if (claimed.includes(chosen)) {
    return {
      kind: "claimed",
      port: chosen,
      detail: `${chosen} is now held by a robot that is already running — two owners on one servo bus is the "Port is in use!" collision, and it usually shows up as an arm that starts and immediately dies`,
      remedy: "pick another bus, or despawn the robot holding this one"
    };
  }
  if (!scanned || known.length === 0) {
    return {
      kind: "unknown",
      detail: `${chosen} has not been confirmed by a scan yet`
    };
  }
  if (!known.includes(chosen)) {
    return {
      kind: "vanished",
      port: chosen,
      detail: `${chosen} is no longer on this machine — the board was unplugged, or it came back under a different /dev path, which these arms do on every reconnect. The picker shows blank because nothing matches, but this path is still what would be opened.`,
      remedy: "rescan, then pick the bus again"
    };
  }
  return { kind: "ok", port: chosen };
}
function blocksSpawn(c) {
  return c.kind === "vanished" || c.kind === "claimed";
}
function safeFilename(name, peerId) {
  const raw = (name ?? "").trim() || `${(peerId ?? "robot").trim() || "robot"}.py`;
  const base = raw.replace(/[/\\:*?"<>|\s]+/g, "-").replace(/^[.-]+/, "");
  const stem = base.endsWith(".py") ? base.slice(0, -3) : base;
  return `${stem || "robot"}.py`;
}
function snippetRefusal(status, detail) {
  const text = (detail || "").trim();
  if (status === 404 || /no profile remembered/i.test(text)) {
    return "nothing is remembered for this board yet — spawn it once and the dashboard can write the file";
  }
  if (status === 422) return text || "the saved payload is not enough to write a snippet";
  if (status === 401 || status === 403) return "your session expired — reload and sign in again";
  return text || "could not write the snippet";
}
function hubAddressMissing(code) {
  if (!code) return null;
  if (/^\s*os\.environ\.setdefault\("ZENOH_CONNECT"/m.test(code)) return null;
  const note = code.match(/^#\s*NOTE:\s*(.+?)\.?\s*$/m);
  return note ? note[1] : "this file carries no hub address, so the peer it starts will connect to nothing";
}
function cleanHubHost(input) {
  const raw = (input ?? "").trim();
  if (!raw) return { host: null, why: "type the address this dashboard is reachable at from the other machine" };
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) {
    return { host: null, why: "paste a host, not a URL — no http:// and no path" };
  }
  const host = raw.replace(/\/+$/, "");
  if (/\s/.test(host)) return { host: null, why: "an address has no spaces in it" };
  const m = host.match(/^([^:]+)(?::([^:]*))?$/);
  if (!m || !m[1]) return { host: null, why: "that is not a host name or IP address" };
  if (m[2] !== void 0 && !/^\d{1,5}$/.test(m[2])) {
    return { host: null, why: 'the part after ":" must be a port number' };
  }
  return { host, why: "" };
}
function DevicePanel({ open, onClose }) {
  var _a;
  const [doc, setDoc] = reactExports.useState(null);
  const sheetRef = reactExports.useRef(null);
  useDialogFocus(sheetRef, open);
  const [robots, setRobots] = reactExports.useState([]);
  const [error, setError] = reactExports.useState(null);
  const [busy, setBusy] = reactExports.useState(false);
  const [status, setStatus] = reactExports.useState(null);
  const [snip, setSnip] = reactExports.useState(null);
  const [logs, setLogs] = reactExports.useState(null);
  const [consent, setConsent] = reactExports.useState(null);
  const [notice, setNotice] = reactExports.useState(null);
  const retry = reactExports.useRef(null);
  const [robotName, setRobotName] = reactExports.useState("");
  const [mode, setMode] = reactExports.useState("sim");
  const [port, setPort] = reactExports.useState("");
  const [camRows, setCamRows] = reactExports.useState([{ name: "main", index: "" }]);
  const [camFps, setCamFps] = reactExports.useState("");
  const [camW, setCamW] = reactExports.useState("");
  const [camH, setCamH] = reactExports.useState("");
  const [robotId, setRobotId] = reactExports.useState("");
  const [peerName, setPeerName] = reactExports.useState("");
  const [calibIds, setCalibIds] = reactExports.useState(null);
  const [profiles, setProfiles] = reactExports.useState(null);
  const [roles, setRoles] = reactExports.useState({});
  const [measuring, setMeasuring] = reactExports.useState(null);
  const [hubDraft, setHubDraft] = reactExports.useState("");
  const [calibFor, setCalibFor] = reactExports.useState(null);
  const [calibFamily, setCalibFamily] = reactExports.useState({});
  const [copied, setCopied] = reactExports.useState(null);
  const [scan, setScan] = reactExports.useState(null);
  const [scanning, setScanning] = reactExports.useState(false);
  const scannedAt = reactExports.useRef(null);
  const scanKey = reactExports.useRef(null);
  const loadSeq = reactExports.useRef(0);
  const inFlight = reactExports.useRef(0);
  const load2 = reactExports.useCallback(async (refresh = false) => {
    const mine = ++loadSeq.current;
    inFlight.current += 1;
    try {
      const next = await api(`/api/devices${refresh ? "?refresh=1" : ""}`);
      if (!isLatestRequest(mine, loadSeq.current)) {
        return { ok: true, after: next, superseded: true };
      }
      setDoc(next);
      scannedAt.current = Date.now();
      setError(null);
      if (!refresh && scanKey.current !== null && hardwareKey(next) !== scanKey.current) {
        scanKey.current = null;
        setScan(null);
      }
      return { ok: true, after: next };
    } catch (e) {
      const msg = (e == null ? void 0 : e.message) ?? String(e);
      if (!refresh) setError(msg);
      if (!isLatestRequest(mine, loadSeq.current)) {
        return { ok: false, error: msg, superseded: true };
      }
      return { ok: false, error: msg };
    } finally {
      inFlight.current -= 1;
    }
  }, []);
  const rescan = reactExports.useCallback(async () => {
    if (scanning) return;
    setScanning(true);
    const before = doc;
    const beforeAtMs = scannedAt.current;
    const outcome = await load2(true);
    if (outcome.superseded) {
      setScanning(false);
      return;
    }
    const verdict2 = rescanReport(before, outcome, { beforeAtMs, nowMs: Date.now() });
    scanKey.current = outcome.ok ? hardwareKey(outcome.after) : null;
    setScan(verdict2);
    setError(null);
    setScanning(false);
  }, [doc, load2, scanning]);
  reactExports.useEffect(() => {
    if (!open) return;
    void load2();
    void api("/api/robots/registry").then((r) => setRobots(normalizeRegistry(r.robots))).catch(() => setRobots([]));
    const id = setInterval(() => {
      if (inFlight.current === 0) void load2();
    }, 5e3);
    return () => clearInterval(id);
  }, [open, load2]);
  reactExports.useEffect(() => {
    if (!open) return;
    let alive = true;
    api("/api/calibration").then((r) => {
      if (alive && (r == null ? void 0 : r.text)) setCalibIds(parseCalibrationList(r.text).entries);
    }).catch(() => {
    });
    api("/api/devices/profiles").then((r) => {
      if (alive) setProfiles((r == null ? void 0 : r.profiles) ?? {});
    }).catch(() => {
    });
    return () => {
      alive = false;
    };
  }, [open]);
  if (!open) return null;
  const act = async (fn, label2, kind = "spawn") => {
    setBusy(true);
    setStatus(null);
    setConsent(null);
    setNotice(null);
    retry.current = { fn, label: label2, kind };
    try {
      const r = await fn();
      setStatus((r == null ? void 0 : r.error) ? `⚠ ${r.error}` : `${label2}: ${(r == null ? void 0 : r.peer_id) ?? "ok"}`);
      setConsent(findConsent(r));
      setNotice(spawnNotice(r));
      await load2();
    } catch (e) {
      const v = deviceActionFailure({
        kind,
        status: e instanceof HttpError ? e.status : 0,
        message: (e == null ? void 0 : e.message) ?? String(e)
      });
      setStatus(v.text);
      setConsent(findConsent(e == null ? void 0 : e.body));
      setNotice(spawnNotice(e == null ? void 0 : e.body));
      if (v.ambiguous) await load2();
    } finally {
      setBusy(false);
    }
  };
  const camNums = {
    fps: camFps === "" ? null : numField(camFps, { what: "fps", min: 1, max: 240 }),
    width: camW === "" ? null : numField(camW, { what: "pixels wide", min: 64, max: 7680 }),
    height: camH === "" ? null : numField(camH, { what: "pixels high", min: 64, max: 4320 })
  };
  const camField = camerasField(camRows, {
    fps: camNums.fps && !camNums.fps.problem ? camNums.fps.value : null,
    width: camNums.width && !camNums.width.problem ? camNums.width.value : null,
    height: camNums.height && !camNums.height.problem ? camNums.height.value : null
  });
  const camProblem = ((_a = [camNums.fps, camNums.width, camNums.height].find((v) => v == null ? void 0 : v.problem)) == null ? void 0 : _a.problem) ?? camField.problem;
  const camNote = [
    ...[camNums.fps, camNums.width, camNums.height].map((v) => v == null ? void 0 : v.note),
    camField.note
    // shared-index warning: sendable, but said out loud
  ].filter(Boolean).join(" · ") || null;
  const anyCam = camRows.some((r) => r.index !== "");
  const nameVerdict2 = peerNameField(peerName, {
    existing: [
      ...Object.keys((doc == null ? void 0 : doc.managed) ?? {}),
      ...Object.values(profiles ?? {}).map((p) => p == null ? void 0 : p.peer_id).filter(Boolean)
    ],
    robotName,
    mode
  });
  const spawn = () => act(() => (forgetJointFailure(nameVerdict2.value), post("/api/devices/spawn", {
    robot_name: robotName,
    peer_id: nameVerdict2.value,
    mode,
    port: mode === "real" ? port || null : null,
    // The camera config must be a MAPPING per entry ({index_or_path: N, ...}); a bare int here is
    // the exact ValueError an operator once hit live: "Camera 'main' config must be a mapping ...
    // got int: 3". camerasField owns that shape (and every name/index collision) in one place.
    cameras: camField.problem ? null : camField.value,
    robot_id: robotId || null
  })), "spawned");
  const showLogs = async (peer) => {
    try {
      const r = await api(`/api/devices/logs/${encodeURIComponent(peer)}`);
      setLogs({ peer, lines: r.lines ?? [] });
    } catch (e) {
      setLogs({ peer, lines: [`⚠ ${(e == null ? void 0 : e.message) ?? String(e)}`] });
    }
  };
  const writeSnippet = async (p, hubHost) => {
    setSnip(null);
    setNotice(null);
    try {
      const r = await post(
        "/api/deploy/snippet",
        { serial: p.serial_number, ...hubHost ? { hub_host: hubHost } : {} }
      );
      setSnip({ device: p.device, code: r.snippet, filename: safeFilename(r.filename, r.peer_id) });
    } catch (e) {
      const status2 = e instanceof HttpError ? e.status : 0;
      setNotice({ text: `✗ ${snippetRefusal(status2, e instanceof Error ? e.message : String(e))}` });
    }
  };
  const respawnRemembered = (p) => {
    var _a2;
    return act(
      () => post("/api/devices/spawn-remembered", { port: p.device }),
      `spawned ${((_a2 = p.remembered) == null ? void 0 : _a2.peer_id) ?? "it"} from its saved profile`
    );
  };
  const measureRole = async (port2) => {
    setMeasuring(port2);
    try {
      const v = await api(`/api/devices/arm-role?port=${encodeURIComponent(port2)}`);
      setRoles((r) => ({ ...r, [port2]: v }));
      if (v.remembered) void load2(false);
    } catch (e) {
      setRoles((r) => ({
        ...r,
        [port2]: { role: "unknown", reason: (e == null ? void 0 : e.message) ?? String(e) }
      }));
    } finally {
      setMeasuring(null);
    }
  };
  const managed = Object.values((doc == null ? void 0 : doc.managed) ?? {});
  const freePorts = (doc == null ? void 0 : doc.serial_ports) ?? [];
  const claimedPorts = new Set(managed.filter((m) => m.alive && m.port).map((m) => m.port));
  const portVerdict = portChoice({
    chosen: port,
    known: freePorts.map((p) => p.device),
    claimed: [...claimedPorts],
    scanned: doc !== null
  });
  const acting = busy;
  const familyFor = (p) => {
    const running = managed.find((m) => m.alive && m.port === p.device);
    if (running == null ? void 0 : running.robot_name) return { family: running.robot_name, source: "the arm running on this port" };
    const picked = (calibFamily[p.device] ?? "").trim();
    if (picked) return { family: picked, source: "your pick" };
    if (robotName.trim()) return { family: robotName.trim(), source: "the robot selected in the spawn form above" };
    if (p.likely_robot) return { family: p.likely_robot, source: "a guess from the USB id — confirm it" };
    return { family: "", source: "" };
  };
  const copyCommand = async (port2, command) => {
    try {
      if (!navigator.clipboard) throw new Error("this page is not a secure origin, so the browser blocks copying");
      await navigator.clipboard.writeText(command);
      setCopied(`${port2}\0ok`);
    } catch (e) {
      setCopied(`${port2}\0${(e == null ? void 0 : e.message) ?? String(e)}`);
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "drawer-backdrop", onClick: onClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("aside", { ref: sheetRef, className: "drawer wide", onClick: (e) => e.stopPropagation(), children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("header", { className: "drawer-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "Devices" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void rescan(), disabled: busy || scanning, children: scanning ? "scanning…" : "rescan" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, "aria-label": "close devices", title: "Escape", children: "✕" })
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "drawer-body", children: [
      error && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", children: [
        "⚠ ",
        error
      ] }),
      scan && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: scan.tone === "bad" ? "result bad" : scan.tone === "warn" ? "result warn" : "result ok", children: scan.text }),
      notice && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result warn", role: "status", children: [
        "⚠ ",
        notice.text,
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", style: { marginLeft: 8 }, onClick: () => setNotice(null), children: "dismiss" })
      ] }),
      consent && /* @__PURE__ */ jsxRuntimeExports.jsx(
        ConsentSheet,
        {
          need: consent,
          target: "spawn",
          onCancel: () => setConsent(null),
          onRetry: () => {
            const again = retry.current;
            setConsent(null);
            if (again) void act(again.fn, again.label, again.kind);
          }
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
          "Managed robots",
          doc !== null ? ` (${managed.length})` : ""
        ] }),
        managed.length === 0 && (() => {
          const line = managedListEmptyLine({ scanned: doc !== null, error });
          return /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", role: "status", children: line.message });
        })(),
        /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "devlist", children: managed.map((m) => {
          var _a2;
          return /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { className: m.alive ? "" : "dead", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "span",
              {
                className: m.alive ? "dot on" : "dot off",
                role: "img",
                "aria-label": m.alive ? "process running" : "process exited",
                title: m.alive ? "process running" : "process exited"
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: m.peer_id }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "meta", children: [
              m.robot_name,
              " · ",
              m.mode,
              m.port ? ` · ${m.port}` : ""
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "devactions", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void showLogs(m.peer_id), children: "logs" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "button",
                {
                  className: "btn ghost danger",
                  disabled: busy,
                  onClick: () => void act(() => post("/api/devices/despawn", { peer_id: m.peer_id }), "despawned", "despawn"),
                  children: m.alive ? "despawn" : "remove"
                }
              )
            ] }),
            !m.alive && (() => {
              const v = deathVerdict(m.returncode);
              return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: v.unexplained ? "deathnote unexplained" : "deathnote", children: v.phrase });
            })(),
            !m.alive && ((_a2 = m.log_tail) == null ? void 0 : _a2.length) ? (() => {
              const startup = retainedOutputIsStartup({ lines: m.log_tail, startedAt: m.started_at });
              return /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                startup === true && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "meta", children: "its startup output — it printed nothing after that, so these are not its last words" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { className: "logtail", children: m.log_tail.slice(-6).join("\n") })
              ] });
            })() : null
          ] }, m.peer_id);
        }) })
      ] }),
      logs && /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
          logs.peer,
          " log",
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setLogs(null), children: "close" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { className: "logtail tall", children: logs.lines.join("\n") || "(no output yet)" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Spawn" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Name" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: peerName,
              placeholder: "left-arm (optional)",
              "aria-invalid": nameVerdict2.problem ? true : void 0,
              onChange: (e) => setPeerName(e.target.value)
            }
          )
        ] }),
        nameVerdict2.problem ? /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint bad", role: "alert", children: [
          "⚠ ",
          nameVerdict2.problem,
          nameVerdict2.suggestion && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            " ",
            /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                type: "button",
                className: "btn ghost tiny",
                onClick: () => setPeerName(nameVerdict2.suggestion),
                children: [
                  "use ",
                  nameVerdict2.suggestion
                ]
              }
            )
          ] })
        ] }) : nameVerdict2.note ? /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: nameVerdict2.note }) : null,
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Robot" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: robotName, onChange: (e) => setRobotName(e.target.value), children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "select…" }),
              robots.map((r) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: r.name, children: r.label }, r.name))
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Mode" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: mode, onChange: (e) => setMode(e.target.value), children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "sim", children: "sim (MuJoCo)" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "real", children: "real hardware" })
            ] })
          ] })
        ] }),
        mode === "real" && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Servo bus" }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: port, onChange: (e) => setPort(e.target.value), children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "select a port…" }),
                freePorts.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: p.device, disabled: claimedPorts.has(p.device), children: [
                  p.device,
                  p.likely_robot ? ` (${p.likely_robot})` : "",
                  claimedPorts.has(p.device) ? " — in use" : ""
                ] }, p.device))
              ] }),
              (portVerdict.kind === "vanished" || portVerdict.kind === "claimed") && /* @__PURE__ */ jsxRuntimeExports.jsxs("em", { className: "field-err", role: "alert", children: [
                "⚠ ",
                portVerdict.detail,
                " — ",
                portVerdict.remedy
              ] })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Calibration id" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  value: robotId,
                  placeholder: "lerobot id (optional)",
                  list: "calib-ids",
                  onChange: (e) => setRobotId(e.target.value)
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("datalist", { id: "calib-ids", children: (calibIds ?? []).filter((c) => c.id && !c.problem).map((c) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: c.id, children: c.model }, `${c.deviceType}/${c.model}/${c.id}`)) })
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "hint", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "why the calibration id matters" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
              "A real robot moves as soon as a task runs. The calibration id must match the one used by ",
              /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "lerobot-calibrate" }),
              ", or the joint limits will be wrong."
            ] })
          ] }),
          (() => {
            const v = calibrationVerdict(robotId, calibIds, robotName);
            if (!v.note) return null;
            return /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: v.warn ? "hint warn" : "hint ok", children: [
              v.warn ? "⚠ " : "✓ ",
              v.note,
              v.suggestion && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                " ",
                /* @__PURE__ */ jsxRuntimeExports.jsxs(
                  "button",
                  {
                    type: "button",
                    className: "btn ghost tiny",
                    onClick: () => setRobotId(v.suggestion),
                    children: [
                      "use ",
                      v.suggestion
                    ]
                  }
                )
              ] })
            ] });
          })()
        ] }),
        camRows.map((row, i) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: i === 0 ? "Camera name" : `Camera ${i + 1} name` }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "text",
                value: row.name,
                placeholder: i === 0 ? "main" : "wrist",
                onChange: (e) => setCamRows((rows) => rows.map((r, j) => j === i ? { ...r, name: e.target.value } : r))
              }
            )
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "index" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "select",
              {
                value: row.index,
                onChange: (e) => setCamRows((rows) => rows.map((r, j) => j === i ? { ...r, index: e.target.value } : r)),
                children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "none" }),
                  ((doc == null ? void 0 : doc.cameras) ?? []).map((c) => (
                    // claimed-by-a-robot is disabled; an index another ROW picked stays
                    // selectable — that case warns below instead of blocking (the operator
                    // may know something we don't).
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: c.index, disabled: !!c.claimed_by, children: [
                      "index ",
                      c.index,
                      c.width ? ` — ${c.width}×${c.height}` : "",
                      c.fps ? ` @ ${c.fps}fps` : "",
                      c.claimed_by ? ` — claimed by ${c.claimed_by}` : "",
                      camRows.some((o, j) => j !== i && o.index === String(c.index)) ? " — also picked above" : ""
                    ] }, c.index)
                  ))
                ]
              }
            )
          ] }),
          camRows.length > 1 && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              type: "button",
              className: "btn ghost tiny",
              "aria-label": `remove camera ${i + 1}`,
              onClick: () => setCamRows((rows) => rows.filter((_, j) => j !== i)),
              children: "✕"
            }
          )
        ] }, i)),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            type: "button",
            className: "btn ghost tiny",
            onClick: () => setCamRows((rows) => [...rows, {
              name: ["main", "wrist", "top"].find((n) => !rows.some((r) => r.name.trim().toLowerCase() === n)) ?? "",
              index: ""
            }]),
            children: "+ add camera"
          }
        ),
        anyCam && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "FPS" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "number",
                  inputMode: "numeric",
                  min: 1,
                  max: 120,
                  value: camFps,
                  placeholder: "30",
                  onChange: (e) => setCamFps(e.target.value)
                }
              )
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Width" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "number",
                  inputMode: "numeric",
                  min: 64,
                  step: 2,
                  value: camW,
                  placeholder: "640",
                  onChange: (e) => setCamW(e.target.value)
                }
              )
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Height" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "number",
                  inputMode: "numeric",
                  min: 64,
                  step: 2,
                  value: camH,
                  placeholder: "480",
                  onChange: (e) => setCamH(e.target.value)
                }
              )
            ] })
          ] }),
          camProblem && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint bad", role: "alert", children: [
            "⚠ ",
            camProblem
          ] }),
          !camProblem && camNote && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: camNote }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Blank = the driver's defaults (640×480 @ 30). A setting the camera can't do fails loudly at spawn — check the log tail, not the stream." })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "btn go",
            disabled: busy || !robotName || mode === "real" && !port || !!camProblem || !!nameVerdict2.problem || mode === "real" && blocksSpawn(portVerdict),
            onClick: spawn,
            children: "spawn"
          }
        ) })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Cameras" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          CameraGallery,
          {
            cameras: (doc == null ? void 0 : doc.cameras) ?? [],
            names: (doc == null ? void 0 : doc.camera_names) ?? [],
            problem: (doc == null ? void 0 : doc.camera_problem) ?? null,
            scanned: doc !== null,
            error
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "hint", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "why some cameras are not previewed" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Camera indices owned by a running robot are never re-probed — opening one steals frames from its capture thread mid-episode." })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(CalibrationSection, {}),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Servo boards" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "leader ↔ follower is measured off the bus voltage (7.4V vs 12V), never guessed from a name." }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "hint", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "how the measurement works" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
            "A follower arm runs a 12V servo bus, a leader 7.4V — so the role can be read off the hardware instead of inherited from a name. The read touches one register (",
            /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "Present_Voltage" }),
            ") and cannot move the arm. A servo bus has a single owner, so an arm that is running must be despawned before it can be measured."
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("ul", { className: "boardlist", children: [
          freePorts.length === 0 && (() => {
            const line = boardListEmptyLine({ scanned: doc !== null, error });
            return /* @__PURE__ */ jsxRuntimeExports.jsx("li", { className: "muted", role: "status", children: line.message });
          })(),
          freePorts.map((p) => {
            const v = roles[p.device];
            const busy2 = claimedPorts.has(p.device);
            return /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row between", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "mono", children: p.device }),
                p.role ? /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "rolebadge " + p.role, children: [
                  p.role,
                  p.role_volts ? " · " + p.role_volts + "V" : ""
                ] }) : /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "rolebadge unmeasured", children: "role not measured" })
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row between", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "muted small", children: p.serial_number ? "serial " + p.serial_number : "no serial number" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "button",
                  {
                    className: "btn ghost",
                    disabled: busy2 || measuring === p.device,
                    title: busy2 ? "this arm is running and owns its bus — despawn it first" : "reads one register; cannot move the arm",
                    onClick: () => void measureRole(p.device),
                    children: measuring === p.device ? "reading…" : busy2 ? "running — despawn to measure" : "measure role"
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "button",
                  {
                    className: "btn ghost",
                    "aria-expanded": calibFor === p.device,
                    title: "show the exact lerobot-calibrate command for this arm — the dashboard runs nothing",
                    onClick: () => {
                      setCopied(null);
                      setCalibFor(calibFor === p.device ? null : p.device);
                    },
                    children: calibFor === p.device ? "hide calibrate command" : "calibrate…"
                  }
                )
              ] }),
              p.remembered && (() => {
                var _a2, _b;
                const mem = rememberedLine(p.remembered, { ...p, calibrations: calibIds });
                return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row between remembered", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "muted small", children: [
                    "last spawned as ",
                    /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: mem.summary }),
                    mem.calibrationId ? ` · calibration id ${mem.calibrationId}` : "",
                    mem.warning && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "warn small", children: [
                      " ⚠ ",
                      mem.warning
                    ] }),
                    mem.idProblem && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "warn small", role: "alert", children: [
                      " ⚠ ",
                      mem.idProblem
                    ] }),
                    ((_a2 = p.remembered.camera_health) == null ? void 0 : _a2.text) && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "warn small", children: [
                      " ⚠ ",
                      p.remembered.camera_health.text
                    ] }),
                    (((_b = p.remembered.camera_health) == null ? void 0 : _b.cameras) ?? []).filter((c) => c.remedy && c.state !== "ready" && c.state !== "unchecked").map((c) => /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint small", children: [
                      " ",
                      c.name,
                      ": ",
                      c.remedy
                    ] }, c.name))
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "button",
                    {
                      className: "btn ghost",
                      title: "write the Python file that recreates this rig on another machine",
                      onClick: () => void writeSnippet(p),
                      children: "deploy .py"
                    }
                  ),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "button",
                    {
                      className: "btn ghost",
                      disabled: acting || claimedPorts.has(p.device),
                      title: claimedPorts.has(p.device) ? "something is already running on this bus — despawn it first" : "starts a child process with the saved payload; a real arm will be energised",
                      onClick: () => void respawnRemembered(p),
                      children: claimedPorts.has(p.device) ? "already running" : `spawn ${p.remembered.peer_id} again`
                    }
                  )
                ] });
              })(),
              (snip == null ? void 0 : snip.device) === p.device && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dev-snippet", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row between", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "muted small", children: [
                    snip.filename,
                    " — runs this rig on another machine"
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost tiny", onClick: () => {
                      var _a2;
                      void ((_a2 = navigator.clipboard) == null ? void 0 : _a2.writeText(snip.code));
                      setCopied(`${p.device}\0copied ${snip.filename}`);
                    }, children: "copy" }),
                    " ",
                    /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost tiny", onClick: () => {
                      const url = URL.createObjectURL(new Blob([snip.code], { type: "text/x-python" }));
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = snip.filename;
                      a.click();
                      setTimeout(() => URL.revokeObjectURL(url), 4e3);
                    }, children: "download" }),
                    " ",
                    /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost tiny", onClick: () => setSnip(null), children: "close" })
                  ] })
                ] }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { className: "snippet", children: snip.code }),
                (() => {
                  const gap = hubAddressMissing(snip.code);
                  if (!gap) return null;
                  const draft = cleanHubHost(hubDraft);
                  const typed = hubDraft.trim() !== "";
                  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "muted small", children: [
                      "No hub address — ",
                      gap,
                      ". The machine that runs this file needs an address it can reach THIS dashboard at; type the one you know."
                    ] }),
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
                      /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
                        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Hub address" }),
                        /* @__PURE__ */ jsxRuntimeExports.jsx(
                          "input",
                          {
                            value: hubDraft,
                            placeholder: "gpu.lan:7447",
                            "aria-invalid": typed && !draft.host ? true : void 0,
                            onChange: (e) => setHubDraft(e.target.value)
                          }
                        )
                      ] }),
                      /* @__PURE__ */ jsxRuntimeExports.jsx(
                        "button",
                        {
                          className: "btn ghost tiny",
                          disabled: !draft.host,
                          onClick: () => {
                            if (draft.host) void writeSnippet(p, draft.host);
                          },
                          children: "rebuild with this address"
                        }
                      )
                    ] }),
                    typed && draft.why && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "muted small", children: draft.why })
                  ] });
                })()
              ] }),
              calibFor === p.device && (() => {
                const { family, source } = familyFor(p);
                const plan = calibratePlan({ ...p, robot_id: knownCalibrationId(profiles, p) }, family);
                const verdict2 = (copied == null ? void 0 : copied.startsWith(p.device + "\0")) ? copied.split("\0")[1] : null;
                return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "calibcmd", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "muted small", children: plan.reason }),
                  plan.command ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "muted small", children: [
                      "model ",
                      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "mono", children: plan.deviceModel }),
                      " — family from ",
                      source
                    ] }),
                    plan.idNote && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: plan.idWarn ? "hint warn" : "hint", children: [
                      plan.idWarn ? "⚠ " : "",
                      plan.idNote
                    ] }),
                    /* @__PURE__ */ jsxRuntimeExports.jsx(
                      CalibrateWizard,
                      {
                        port: p.device,
                        role: p.role === "leader" ? "leader" : "follower",
                        model: plan.deviceModel,
                        deviceId: plan.deviceId,
                        onSaved: () => {
                          api("/api/calibration").then((r) => {
                            if (r == null ? void 0 : r.text) setCalibIds(parseCalibrationList(r.text).entries);
                          }).catch(() => {
                          });
                        }
                      }
                    ),
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
                      /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "run it in a terminal instead" }),
                      /* @__PURE__ */ jsxRuntimeExports.jsx("code", { className: "cmdline", children: plan.command }),
                      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row", children: [
                        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void copyCommand(p.device, plan.command), children: "copy" }),
                        verdict2 === "ok" && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "muted small", children: "copied — paste it in a terminal" }),
                        verdict2 && verdict2 !== "ok" && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "warn small", children: [
                          "⚠ could not copy: ",
                          verdict2,
                          " — select the line above instead"
                        ] })
                      ] }),
                      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
                        "Same procedure as the wizard: it will ask you to move the arm through its range BY HAND. When the file lands, press ",
                        /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "reload" }),
                        " in Calibration above to see it."
                      ] })
                    ] })
                  ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                    plan.needsMeasurement && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
                      "use ",
                      /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "measure role" }),
                      " on this row first — one register read, no motion."
                    ] }),
                    !family && robots.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "row", children: [
                      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "muted small", children: "which arm is this?" }),
                      /* @__PURE__ */ jsxRuntimeExports.jsxs(
                        "select",
                        {
                          value: calibFamily[p.device] ?? "",
                          onChange: (e) => setCalibFamily({ ...calibFamily, [p.device]: e.target.value }),
                          children: [
                            /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "— pick the robot type —" }),
                            robots.map((r) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: r.name, children: r.label }, r.name))
                          ]
                        }
                      )
                    ] })
                  ] })
                ] });
              })(),
              v && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: v.mismatch ? "warn small" : "muted small", children: [
                v.reason,
                v.remedy ? " — " + v.remedy : "",
                v.mismatch ? " ⚠ " + v.mismatch.message + " — " + v.mismatch.remedy : "",
                v.remembered ? " · remembered for this board" : "",
                v.remember_problem ? " · " + v.remember_problem : ""
              ] })
            ] }, p.device);
          })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Detected hardware" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("dl", { className: "kv", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "serial" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "mono", children: hardwareSummaryValue({
            scanned: doc !== null,
            error,
            items: freePorts.map((p) => p.device),
            emptyNote: "(a servo bus shows up as /dev/tty.usbmodem* or /dev/ttyACM*)"
          }) }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dt", { children: "cameras" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("dd", { className: "mono", children: hardwareSummaryValue({
            scanned: doc !== null,
            error,
            items: ((doc == null ? void 0 : doc.cameras) ?? []).map((c) => `#${c.index}${c.claimed_by ? `→${c.claimed_by}` : ""}`),
            emptyNote: "answered a probe"
          }) })
        ] })
      ] })
    ] }),
    status && /* @__PURE__ */ jsxRuntimeExports.jsx("footer", { className: "drawer-foot", children: status })
  ] }) });
}
const ABSENCE = "no arms are on the mesh";
function verdict(joiner, route) {
  return { text: `${ABSENCE}${joiner}${route}`, route, offerDevices: true };
}
function noArmsVerdict(peerCount, remembered) {
  if (peerCount > 0) return null;
  if (remembered == null) {
    return verdict(
      ", and ",
      "the devices screen could not be reached to say whether any are configured — open it to check"
    );
  }
  const bringable = remembered.filter((b) => b.peer_id && !b.claimed);
  if (bringable.length) {
    const names = bringable.map((b) => b.peer_id).join(" and ");
    return verdict(", but ", `the devices screen remembers ${names} — one click there brings a board back up with the config it last ran with`);
  }
  if (remembered.length) {
    return verdict(", though ", "a process already holds every configured board — check those children's logs on the devices screen rather than spawning again");
  }
  return verdict(" and ", "no board is configured yet — plug an arm in and spawn it once from the devices screen; it is remembered by its USB serial afterwards");
}
const SIM = 'Robot("so101").run()   # sim, no hardware needed';
function startSnippet(boards) {
  const real = (boards ?? []).find((b) => b.device);
  if (!real) {
    return {
      code: `from strands_robots import Robot
${SIM}
Robot("so101", mode="real", port="/dev/ttyACM0").run()`,
      // Said out loud: no board was detected here, so that path is an EXAMPLE. The old snippet made
      // the same claim silently and it was wrong on this very machine.
      provenance: "no servo board is plugged into this machine, so the port above is an example — a real one appears here once a board is detected",
      real: false
    };
  }
  const family = (real.robot_name || "so101").trim() || "so101";
  return {
    code: `from strands_robots import Robot
Robot("${family}").run()   # sim, no hardware needed
Robot("${family}", mode="real", port="${real.device}").run()`,
    // Naming the port's origin keeps it falsifiable: a port that moved is a visible claim, not a
    // mysterious failure inside lerobot.
    provenance: `the port is the board detected on this machine right now (${real.device})` + (real.robot_name ? ` — last spawned as "${family}"` : ""),
    real: true
  };
}
function estopNothingTargeted(opts) {
  const stale = (opts.staleSkipped ?? []).filter(Boolean);
  if (stale.length > 0) {
    return {
      headline: "nothing was stopped — every peer was unreachable",
      detail: `no stop was delivered: ${stale.join(", ")} had no heartbeat, so ${stale.length === 1 ? "it was" : "they were"} skipped rather than stopped. A peer with no heartbeat can still be moving — losing telemetry is not stopping. If anything is moving, cut power at the supply.`,
      cutPower: true
    };
  }
  return {
    headline: "nothing was stopped — this view had no live peer to target",
    detail: "no stop was delivered, because this dashboard sees no live peer. That is a statement about this fleet view, not about the room: a robot started outside this dashboard, one whose presence has not arrived yet, or one already dropped as stale is invisible here and can still be moving. If anything is moving, cut power at the supply.",
    cutPower: true
  };
}
function signedRailClaim(opts) {
  if (opts.lockoutEngaged !== true) return null;
  const by = typeof opts.issuer === "string" && opts.issuer.trim() ? ` (signed by ${opts.issuer.trim()})` : "";
  const latched = `fleet LOCKOUT engaged${by}`;
  const refused = "A peer that received it refuses all commands until you resume with the override code.";
  const acks = typeof opts.responsesReceived === "number" ? opts.responsesReceived : null;
  const notStopped = (opts.peersNotStopped ?? []).map((v) => v == null ? "" : String(v).trim()).filter(Boolean);
  if (notStopped.length > 0) {
    return {
      headline: `${latched} — but ${notStopped.length} peer${notStopped.length === 1 ? "" : "s"} reported NOT stopping`,
      detail: `${notStopped.join(", ")} answered that ${notStopped.length === 1 ? "it did" : "they did"} not stop, so ${notStopped.length === 1 ? "that robot" : "those robots"} may still be executing. The lockout stops the NEXT command; it does not halt motion already underway. If anything is moving, cut power at the supply.`,
      cutPower: true
    };
  }
  if (acks === 0) {
    return {
      headline: `${latched} — but NO peer acknowledged`,
      detail: "the lockout is latched on this rail, and no peer replied to say it received the stop. Nothing here shows a robot was halted: a peer that never answered is not a peer that stopped. If anything is moving, cut power at the supply.",
      cutPower: true
    };
  }
  if (acks === null) {
    return {
      headline: `${latched} — peer acknowledgements unknown`,
      detail: `this server does not report which peers replied, so how far the stop reached cannot be shown here. ${refused}`,
      cutPower: false
    };
  }
  return {
    headline: `${latched} — ${acks} peer${acks === 1 ? "" : "s"} acknowledged, none reported a failure to stop`,
    detail: `${refused} A peer that never replied is not counted here, so it is not covered by that count.`,
    cutPower: false
  };
}
function estopPaths(meshBacked) {
  return meshBacked ? { estop: ["/api/mesh/safety/estop", "/api/safety/estop"], resume: ["/api/mesh/safety/resume", "/api/safety/resume"] } : { estop: ["/api/safety/estop"], resume: ["/api/safety/resume"] };
}
function EstopSheet({
  open,
  onClose,
  linkWarning,
  meshBacked = false
}) {
  const [firing, setFiring] = reactExports.useState(false);
  const [result, setResult] = reactExports.useState(null);
  const [error, setError] = reactExports.useState(null);
  const [code, setCode] = reactExports.useState("");
  const [resuming, setResuming] = reactExports.useState(false);
  const [resumeMsg, setResumeMsg] = reactExports.useState(null);
  const [simLockout, setSimLockout] = reactExports.useState(null);
  const paths = estopPaths(meshBacked);
  const resume = async () => {
    if (meshBacked && !code.trim()) return;
    setResuming(true);
    setResumeMsg(null);
    try {
      if (meshBacked) {
        const r = await post(paths.resume[0], { override_code: code });
        if (r.status === "ok") {
          setResumeMsg("✓ lockout cleared - fleet accepting commands again");
          setCode("");
        } else setResumeMsg(`✗ ${r.error ?? "resume rejected"} (wrong code? brute-force cooldown?)`);
      }
      const sim = await post("/api/safety/resume");
      setSimLockout(sim.lockout);
      if (!meshBacked) setResumeMsg(`✓ sim lockout lifted - ${sim.lockout.state}: ${sim.lockout.reason ?? ""}`);
    } catch (e) {
      setResumeMsg(resumeFailureVerdict({
        status: e instanceof HttpError ? e.status : 0,
        message: (e == null ? void 0 : e.message) ?? String(e)
      }).text);
    } finally {
      setResuming(false);
    }
  };
  const fire = async () => {
    setFiring(true);
    setError(null);
    try {
      const sim = await post("/api/safety/estop");
      setSimLockout(sim.lockout);
      if (meshBacked) setResult(await post(paths.estop[0]));
      else setResult(simOnlyResult(sim.lockout));
    } catch (e) {
      setError(estopFailureVerdict({
        status: e instanceof HttpError ? e.status : 0,
        message: (e == null ? void 0 : e.message) ?? String(e)
      }));
    } finally {
      setFiring(false);
    }
  };
  if (!open) return null;
  const simOnly = (result == null ? void 0 : result.targeted.length) === 0 && !meshBacked;
  const unconfirmed = result ? result.counts.not_stopped + result.counts.no_answer : 0;
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-backdrop", onClick: result && unconfirmed > 0 ? void 0 : onClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `sheet estop-sheet${unconfirmed > 0 ? " danger" : ""}`, onClick: (e) => e.stopPropagation(), children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "🛑 Stop everything" }),
    !result && !error && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: meshBacked ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        "Sends ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: '{action: "stop"}' }),
        " to every peer with a live heartbeat and reports what each one answered."
      ] }) : /* @__PURE__ */ jsxRuntimeExports.jsx(jsxRuntimeExports.Fragment, { children: "Freezes every simulated robot this dashboard is running and latches its lockout; no mesh bridge is online, so no fleet peer is reached from here." }) }),
      meshBacked && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        "Fires BOTH rails: per-peer stop commands (answered individually below) and the signed ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "strands/safety/estop" }),
        " envelope, which engages a fleet-wide LOCKOUT - every listening peer refuses further commands until a resume with the operator override code. A peer that is wedged or fully off the mesh still needs the hardware e-stop. The simulated robots of this dashboard are frozen too."
      ] }),
      linkWarning && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint warn", children: [
        "⚠ ",
        linkWarning,
        " Pressing STOP ALL is still worth it — it is sent the moment the link returns — but do not wait for it: the arms’ power switch is the only brake that does not go through this page."
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        "tip: ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "." }),
        " opens this sheet from anywhere — it works even when a drawer or dialog is covering the button."
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn danger big", onClick: fire, disabled: firing, children: firing ? "stopping…" : "STOP ALL ROBOTS" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, disabled: firing, children: "cancel" })
      ] })
    ] }),
    error && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "result bad", role: "alert", children: error.headline }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint warn", children: error.advice }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn danger", onClick: fire, children: error.retryRepeats ? "send the stop again" : "retry" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, children: "close" })
      ] })
    ] }),
    result && simOnly && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: (simLockout == null ? void 0 : simLockout.state) === "locked" ? "result ok" : "result bad", role: "status", children: (simLockout == null ? void 0 : simLockout.state) === "locked" ? `✓ sim lockout engaged${simLockout.by ? ` by ${simLockout.by}` : ""} - every simulated robot is frozen` : `lockout ${(simLockout == null ? void 0 : simLockout.state) ?? "unknown"}: ${(simLockout == null ? void 0 : simLockout.reason) ?? ""}` }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Nothing outside this process was reached: the mesh bridge is offline, so a real arm on the desk still needs its power switch." }),
      (simLockout == null ? void 0 : simLockout.state) === "locked" && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "resume-box", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "resume-row", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: resume, disabled: resuming, children: resuming ? "…" : "resume simulation" }) }),
        resumeMsg && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: resumeMsg })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, children: "close" }) })
    ] }),
    result && !simOnly && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "div",
        {
          className: result.all_stopped ? "result ok" : "result bad",
          role: result.all_stopped ? "status" : "alert",
          children: result.all_stopped ? `✓ all ${result.counts.stopped} live peer(s) confirmed stopped` : `⚠ ${unconfirmed} of ${result.targeted.length} peer(s) NOT confirmed stopped`
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "estop-list", children: Object.entries(result.stopped).map(([peer, info]) => /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { className: info.state, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: peer }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: info.state === "stopped" ? "stopped" : info.state === "no_answer" ? "no answer — may still be moving" : `refused: ${typeof info.detail === "string" ? info.detail : JSON.stringify(info.detail)}` })
      ] }, peer)) }),
      result.stale_skipped.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        "skipped (no heartbeat, cannot be reached): ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: result.stale_skipped.join(", ") })
      ] }),
      result.targeted.length === 0 && (() => {
        const reach = estopNothingTargeted({ staleSkipped: result.stale_skipped });
        return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", role: "alert", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: reach.headline }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: reach.detail })
        ] });
      })(),
      result.lockout_engaged && (() => {
        var _a, _b, _c;
        const claim = signedRailClaim({
          lockoutEngaged: result.lockout_engaged,
          issuer: (_a = result.signed_rail) == null ? void 0 : _a.issuer,
          responsesReceived: (_b = result.signed_rail) == null ? void 0 : _b.responses_received,
          peersNotStopped: (_c = result.signed_rail) == null ? void 0 : _c.peers_not_stopped
        });
        return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "resume-box", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "result bad", role: "alert", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
              "🔒 ",
              claim == null ? void 0 : claim.headline
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: claim == null ? void 0 : claim.detail })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "resume-row", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "password",
                placeholder: "operator override code",
                "aria-label": "operator override code",
                value: code,
                onChange: (e) => setCode(e.target.value),
                onKeyDown: (e) => e.key === "Enter" && resume(),
                disabled: resuming
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: resume, disabled: resuming || !code.trim(), children: resuming ? "…" : "resume fleet" })
          ] }),
          resumeMsg && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint", children: resumeMsg }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
            "The code is verified locally and an HMAC proof is broadcast — the code itself never crosses the wire. Set ",
            /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "STRANDS_MESH_OVERRIDE_CODE" }),
            " identically on every peer."
          ] })
        ] });
      })(),
      result.signed_rail && !result.signed_rail.signed && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint warn", children: [
        "⚠ signed rail unavailable (",
        result.signed_rail.error,
        ") — only per-peer stops were sent; no fleet lockout is in place."
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sheet-actions", children: [
        !result.all_stopped && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn danger", onClick: fire, children: "send again" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: onClose, children: "close" })
      ] })
    ] })
  ] }) });
}
function simOnlyResult(_lockout) {
  return {
    targeted: [],
    stale_skipped: [],
    counts: { stopped: 0, not_stopped: 0, no_answer: 0 },
    all_stopped: true,
    stopped: {},
    lockout_engaged: false
  };
}
const DOCS_ORIGIN = "https://strands-labs.github.io/robots";
const DOC_LINKS = [
  {
    label: "Strands Robots docs",
    url: `${DOCS_ORIGIN}/`,
    note: "the whole library: robots, mesh, policies, training"
  },
  {
    label: "Quickstart",
    url: `${DOCS_ORIGIN}/getting-started/quickstart/`,
    note: "install, then a robot in three lines of Python"
  }
];
const REPO_DOC_PATHS = [
  "docs/dashboard/quickstart.md — this screen, with an SO-101",
  "docs/dashboard/collect-train-deploy.md — the full loop",
  "docs/dashboard/troubleshooting.md — when a camera or an arm stays dark",
  "docs/dashboard/remote-access.md — reaching this dashboard from outside"
];
const HELP_TOPICS = [
  {
    title: "What this page is",
    lines: [
      "A live console for every robot on your mesh: each card is one peer, its joint strips are real positions arriving over the network, and any command you send goes to real hardware unless that peer is a simulation.",
      "The dashboard does not own the robots. It joins the same mesh they do, so a peer can appear or vanish without anything here being broken."
    ]
  },
  {
    title: "Stopping things — read this first",
    lines: [
      'STOP ALL is in the top-right corner of every screen and is the first thing the Tab key reaches. The "." key opens it from anywhere, including over an open drawer, and Cmd+. (Ctrl+. on Windows and Linux) works even while you are typing in a field.',
      "It broadcasts an emergency stop and then LOCKS OUT commands until you resume; it does not power anything down.",
      "If this page cannot reach the server it says so and marks the button degraded — the arm's own power switch is then the only brake that does not go through this page."
    ]
  },
  {
    title: "A safe first action",
    lines: [
      'Open a robot card, pick the "mock" policy (a built-in sine test that needs no model), type any task sentence and press ▶ on a SIM peer. Nothing physical moves.',
      "On a real arm, ▶ asks for confirmation first and names the robot it is about to move. That confirmation is the last step before metal moves."
    ]
  },
  {
    title: "Collect → train → deploy",
    lines: [
      "1. ⚙ devices — see the USB arms and cameras this machine can find, name them, and spawn one as a peer.",
      "2. ⏺ record — teleoperate a leader arm and capture episodes into a LeRobot dataset.",
      "3. 🎓 train — point a trainer at that dataset (local folder or a Hugging Face repo) and watch the loss.",
      "4. Deploy the checkpoint from the training screen: it prefills the run form on a robot card, and you press ▶."
    ]
  },
  {
    title: "When something looks wrong",
    lines: [
      "A card greys out when its peer stops announcing itself — that is the mesh going quiet, not a crash.",
      "A camera tile that never shows a frame is usually an OS permission: on macOS a dashboard started by a background daemon can never be granted camera access, and no prompt will appear.",
      "The ☰ activity log records every command this dashboard sent, with what came back."
    ]
  }
];
function HelpSheet({ open, onClose }) {
  const closeRef = reactExports.useRef(null);
  reactExports.useEffect(() => {
    var _a;
    if (open) (_a = closeRef.current) == null ? void 0 : _a.focus();
  }, [open]);
  if (!open) return null;
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sheet-backdrop", onClick: onClose, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "div",
    {
      className: "sheet help-sheet",
      role: "dialog",
      "aria-modal": "true",
      "aria-label": "Help",
      onClick: (e) => e.stopPropagation(),
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "help-head", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "Help" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { ref: closeRef, className: "btn ghost", onClick: onClose, "aria-label": "Close help", children: "✕" })
        ] }),
        HELP_TOPICS.map((t) => /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { className: "help-topic", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: t.title }),
          t.lines.map((line, i) => /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: line }, i))
        ] }, t.title)),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("section", { className: "help-topic", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Deeper reading" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "help-links", children: DOC_LINKS.map((l) => /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("a", { href: l.url, target: "_blank", rel: "noreferrer noopener", children: [
              l.label,
              " ↗"
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint", children: [
              " — ",
              l.note
            ] })
          ] }, l.url)) }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "In this repository (not published yet):" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "help-paths", children: REPO_DOC_PATHS.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: p.split(" — ")[0] }),
            " — ",
            p.split(" — ")[1]
          ] }, p)) })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Press ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "?" }),
          " for this sheet, ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "." }),
          " for STOP ALL, ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "Esc" }),
          " to close."
        ] })
      ]
    }
  ) });
}
const TYPING_TAGS = /^(INPUT|TEXTAREA|SELECT)$/i;
function isTyping(e) {
  return !!e.editable || TYPING_TAGS.test(e.targetTag ?? "");
}
function hotkeyVerdict(e) {
  if (e.key === "Escape") return "close";
  const stopChord = (e.metaKey || e.ctrlKey) && !e.altKey && e.key === ".";
  if (stopChord) return "estop";
  if (isTyping(e)) return null;
  if (e.altKey || e.metaKey || e.ctrlKey) return null;
  if (e.key === ".") return "estop";
  if (e.key === "?") return "help";
  return null;
}
const ESTOP_KEYSHORTCUTS = ". Meta+. Control+.";
function EstopButton({
  onClick,
  posture
}) {
  const degraded = !!(posture == null ? void 0 : posture.degraded);
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "estop-layer", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
    "button",
    {
      className: `estop${degraded ? " unreachable" : ""}`,
      onClick,
      title: (posture == null ? void 0 : posture.title) ?? "Stop every robot on the mesh - keyboard: . anywhere, or Cmd/Ctrl+. even while typing",
      "aria-label": degraded ? "Emergency stop: the link is down, so this may not reach the robots" : "Emergency stop: stop every robot on the mesh",
      "aria-keyshortcuts": ESTOP_KEYSHORTCUTS,
      children: degraded ? "🛑 STOP ALL ⚠" : "🛑 STOP ALL"
    }
  ) });
}
class ErrorBoundary extends reactExports.Component {
  constructor() {
    super(...arguments);
    __publicField(this, "state", { error: null });
    /** Try this screen again without losing the rest of the session. */
    __publicField(this, "retry", () => this.setState({ error: null }));
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    console.error(`[dashboard] ${this.props.label} crashed:`, error, info.componentStack);
  }
  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "crashcard", role: "alert", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { children: [
        this.props.label,
        " stopped working"
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { children: [
        "The rest of the dashboard is still live — the fleet, the robot cards and",
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: " STOP ALL" }),
        " all still work."
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { children: error.message || String(error) }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "crashcard-actions", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: this.retry, children: "try again" }),
        this.props.onDismiss && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: this.props.onDismiss, children: "close" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => location.reload(), children: "reload the page" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Details are in the browser console. If it crashes again on the same screen, that screen's data is the problem, not your fleet." })
    ] });
  }
}
const TRAINING_POLL_S = 5;
const TRAINING_STALE_POLLS = 3;
function trainingFreshness(input) {
  const { polledAtS, nowS } = input;
  const failures = input.failures ?? 0;
  const state = (input.state ?? "").toLowerCase();
  const staleAfterS = input.staleAfterS ?? TRAINING_POLL_S * TRAINING_STALE_POLLS;
  const ageS2 = polledAtS != null && Number.isFinite(polledAtS) && polledAtS > 0 ? Math.max(0, nowS - polledAtS) : null;
  const settled = state === "success" || state === "failed" || state === "cancelled";
  if (ageS2 === null) {
    const why2 = failures > 0 ? `${failures} status poll${failures === 1 ? "" : "s"} failed${input.error ? `: ${input.error}` : ""}` : "no status read yet";
    return {
      stale: !settled && failures > 0,
      ageS: null,
      note: failures > 0 ? `⚠ never read this job's status — ${why2}` : "",
      title: `no status has been read for this job (${why2})`
    };
  }
  const stale = !settled && ageS2 > staleAfterS;
  const agoStr = ageS2 < 90 ? `${Math.round(ageS2)}s` : `${Math.round(ageS2 / 60)}m`;
  const failPart = failures > 0 ? ` (${failures} failed poll${failures === 1 ? "" : "s"}${input.error ? `: ${input.error}` : ""})` : "";
  return {
    stale,
    ageS: ageS2,
    // The numbers are the claim, so the sentence names them rather than the poll.
    note: stale ? `⚠ these numbers are ${agoStr} old — the status feed stopped${failPart}, so the run may have died, finished, or moved on` : "",
    title: settled ? `final status, read ${agoStr} ago` : `status read ${agoStr} ago${failPart}`
  };
}
const EMBODIMENT = {
  key: "embodiment",
  label: "embodiment",
  placeholder: "new_embodiment",
  say: "GR00T needs an embodiment tag (--embodiment_tag): which robot body the data came from",
  required: true
};
function extraFields(provider) {
  return provider === "groot" ? [EMBODIMENT] : [];
}
function missingForProvider(provider, values) {
  const missing = extraFields(provider).filter((f) => f.required && !(values[f.key] || "").trim());
  if (!missing.length) return "";
  return missing.map((f) => f.say).join(" · ");
}
function datasetName(sel) {
  const src = (sel.dataset_root ?? "").trim() || (sel.dataset_repo_id ?? "").trim();
  if (!src) return null;
  const seg = src.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
  const clean = seg.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[_.]+|[_.]+$/g, "");
  return clean || null;
}
function suggestOutputDir(sel, currentValue) {
  if ((currentValue ?? "").trim()) return null;
  const name = datasetName(sel);
  if (!name) return null;
  return `/tmp/train_${name}`;
}
const NO_SPLIT = "empty: trains on every episode and reports training loss only — a validation split is what tells learning from memorising";
function holdout(raw, episodeCount) {
  const text = (raw ?? "").trim();
  if (!text) return { send: null, problem: null, say: NO_SPLIT };
  const n = Number(text);
  if (!Number.isFinite(n)) {
    return { send: null, problem: `“${text}” is not a number`, say: NO_SPLIT };
  }
  if (n <= 0) {
    return {
      send: null,
      problem: `a holdout of ${n} reserves no episodes at all — leave it empty to train on every episode`,
      say: NO_SPLIT
    };
  }
  if (!Number.isInteger(n)) {
    return {
      send: null,
      problem: `${text} would reserve ${Math.ceil(n)} episodes, not ${Math.floor(n)} — enter a whole number`,
      say: NO_SPLIT
    };
  }
  const known = typeof episodeCount === "number" && Number.isFinite(episodeCount) && episodeCount > 0;
  if (known && n >= episodeCount) {
    return {
      send: null,
      problem: `this dataset has ${episodeCount} episodes — holding out ${n} leaves ${Math.max(0, episodeCount - n)} to train on`,
      say: NO_SPLIT
    };
  }
  const rest = known ? `${episodeCount - n} to train on, ` : "";
  return {
    send: n,
    problem: null,
    say: `holds out the last ${n} episode${n === 1 ? "" : "s"} for validation (${rest}so an eval loss is logged alongside the training loss)`
  };
}
function labelsGate(row) {
  if (!row) return { ok: false, reason: "no dataset selected" };
  if (!row.root) {
    return { ok: false, reason: "labels live in a sidecar next to the dataset on disk — download this Hub dataset first" };
  }
  if (row.recording) {
    return { ok: false, reason: "this dataset is being recorded right now — episodes are judged after the session" };
  }
  return { ok: true, reason: "show the deterministic verdicts and judge annotations for each episode" };
}
function labelSummary(view, error) {
  if (error) return { text: `Labels could not be read — ${error}`, tone: "warn" };
  if (!view) return { text: "Reading labels…", tone: "plain" };
  if (view.sidecar_error) return { text: `Labels may be damaged — ${view.why}`, tone: "warn" };
  if (!view.episodes.length || !view.can_annotate) {
    return { text: view.why, tone: "plain" };
  }
  const parts = [
    `${view.labelled}/${view.with_verdict} judged`,
    view.total_episodes != null ? `${view.total_episodes} episodes recorded` : null,
    view.benchmark ? `benchmark ${view.benchmark}` : null,
    view.disputed ? `${view.disputed} disputing the verdict` : null
  ].filter(Boolean);
  return { text: parts.join(" · "), tone: view.disputed ? "warn" : "plain" };
}
function labelRowLine(row) {
  if (!row.annotatable) {
    return {
      badge: "—",
      detail: "no deterministic verdict, so it cannot be annotated",
      muted: true
    };
  }
  const badge = row.verdict === "success" ? "✓" : row.verdict === "failure" ? "✗" : "—";
  const bits = [
    row.quality ? `quality ${row.quality}` : "awaiting a quality grade",
    row.failure_mode || null,
    row.disputes_verdict ? "judge disputes this verdict" : null,
    row.note || null,
    row.model ? `by ${row.model}` : null
  ].filter(Boolean);
  return { badge, detail: bits.join(" · "), muted: !row.quality };
}
function fieldSupport(fields, key, loaded) {
  if (!loaded) return { ok: true, why: "" };
  if (Array.isArray(fields)) {
    if (fields.includes(key)) return { ok: true, why: "" };
    return {
      ok: false,
      why: `this dashboard's server does not accept ${key} — it is running code from before the field existed. Restart the dashboard to pick it up.`
    };
  }
  return {
    ok: false,
    why: `this dashboard's server is older than ${key} (it does not publish the field list yet). Restart the dashboard to pick it up.`
  };
}
const NOUN = {
  training: "the training job",
  collect: "the collection run",
  replay: "the replay",
  export: "the export"
};
const DUPLICATE = {
  training: "Pressing train again could start a SECOND run on the same GPU, writing into the same output_dir — check the job list below (it refreshes now) before you do.",
  collect: "Pressing collect again could spawn a SECOND recorder appending to the same dataset — look for a new peer in the fleet grid first.",
  replay: "Pressing replay again could spawn a SECOND peer driving the same arm — look for a new peer in the fleet grid first.",
  export: "It is safe to retry an export, but check the artifact path before assuming nothing was written."
};
function sideEffectVerdict(input) {
  const noun = NOUN[input.kind] ?? "the request";
  const why2 = String(input.message ?? "").trim() || "no detail";
  if (refusedBeforeActing(input.status)) {
    return {
      text: `✗ refused (${input.status}: ${why2}) — ${noun} was NOT started, nothing is running.`,
      delivered: "no",
      doubleRunRisk: false
    };
  }
  const transport = !Number(input.status ?? 0);
  const head = transport ? `⚠ no answer came back (${why2}) — ${noun} MAY have started; this page cannot tell.` : `⚠ the server failed mid-request (${input.status}: ${why2}) — ${noun} MAY have started anyway.`;
  return {
    text: `${head} ${DUPLICATE[input.kind] ?? ""}`.trim(),
    delivered: "unknown",
    doubleRunRisk: input.kind !== "export"
  };
}
function pushLoss(trace, step, loss, cap = 240) {
  const s = typeof step === "number" && Number.isFinite(step) ? step : null;
  const l = typeof loss === "number" && Number.isFinite(loss) ? loss : null;
  if (s === null || l === null) return trace;
  const last = trace[trace.length - 1];
  if (last) {
    if (s === last.step) {
      if (l === last.loss) return trace;
      return [...trace.slice(0, -1), { step: s, loss: l }];
    }
    if (s < last.step) return [{ step: s, loss: l }];
  }
  let next = [...trace, { step: s, loss: l }];
  if (next.length > cap) {
    const half = Math.floor(next.length / 2);
    next = [...next.slice(0, half).filter((_, i) => i % 2 === 0), ...next.slice(half)];
  }
  return next;
}
function lossBand(points) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    if (p.loss < lo) lo = p.loss;
    if (p.loss > hi) hi = p.loss;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1, flat: true };
  const spread = hi - lo;
  const magnitude = Math.max(Math.abs(hi), Math.abs(lo));
  const floorSpan = magnitude > 0 ? magnitude * 0.02 : 1;
  if (spread >= floorSpan) return { lo, hi, flat: false };
  const mid = (lo + hi) / 2;
  return { lo: mid - floorSpan / 2, hi: mid + floorSpan / 2, flat: true };
}
function lossPath(points, width, height, pad = 2) {
  if (points.length < 2 || width <= 0 || height <= 0) return [];
  const s0 = points[0].step;
  const s1 = points[points.length - 1].step;
  const { lo, hi } = lossBand(points);
  const sSpan = Math.max(1e-9, s1 - s0);
  const lSpan = Math.max(1e-9, hi - lo);
  const w = width - pad * 2;
  const h = height - pad * 2;
  return points.map((p) => [
    pad + (p.step - s0) / sSpan * w,
    // loss falls downward on the chart: low loss = low y? No - low loss is
    // GOOD, so it sits at the BOTTOM (large y), matching every loss curve
    // a practitioner has ever seen.
    pad + (1 - (p.loss - lo) / lSpan) * h
  ]);
}
function fmtStep(step) {
  if (!Number.isFinite(step)) return "?";
  if (step >= 1e6) return `${(step / 1e6).toFixed(1)}M`;
  if (step >= 1e3) return `${(step / 1e3).toFixed(1)}k`;
  return String(step);
}
function LossSpark({ trace, height = 34 }) {
  const ref = reactExports.useRef(null);
  reactExports.useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const cssW = Math.max(1, (parent == null ? void 0 : parent.clientWidth) ?? 160);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, height);
    const pts = lossPath(trace, cssW, height, 3);
    if (!pts.length) return;
    ctx.strokeStyle = "rgba(124, 192, 255, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.stroke();
    const [lx, ly] = pts[pts.length - 1];
    ctx.fillStyle = "#7cc0ff";
    ctx.beginPath();
    ctx.arc(lx, ly, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }, [trace, height]);
  if (trace.length < 2) {
    return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "loss-spark empty", children: trace.length === 1 ? `first reading: loss ${trace[0].loss.toPrecision(3)} @ ${fmtStep(trace[0].step)} steps — curve appears as polling continues` : "no loss readings yet" });
  }
  const first = trace[0];
  const last = trace[trace.length - 1];
  const flat = lossBand(trace).flat;
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "loss-spark", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("canvas", { ref }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "loss-spark-label", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        "loss ",
        last.loss.toPrecision(3)
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "dim", children: [
        fmtStep(first.step),
        " → ",
        fmtStep(last.step),
        " steps (observed)",
        flat ? " · flat across this window" : ""
      ] })
    ] })
  ] });
}
const DONE_OK = /* @__PURE__ */ new Set(["succeeded", "success", "completed", "complete", "finished"]);
const DONE_BAD = /* @__PURE__ */ new Set(["failed", "error", "crashed", "cancelled", "canceled", "stopped", "killed"]);
function isTerminal(state) {
  const s = (state ?? "").toLowerCase();
  return DONE_OK.has(s) || DONE_BAD.has(s);
}
function shortJob(id) {
  const s = String(id ?? "");
  return s.length <= 12 ? s : `${s.slice(0, 8)}…`;
}
function jobTransitions(prev, now) {
  const ok = [];
  const bad = [];
  for (const [id, state] of Object.entries(now)) {
    const was = prev[id];
    if (was === void 0) continue;
    if (!isTerminal(state) || isTerminal(was)) continue;
    const s = (state ?? "").toLowerCase();
    (DONE_BAD.has(s) ? bad : ok).push(`${shortJob(id)} ${s}`);
  }
  if (!bad.length && !ok.length) return "";
  const parts = [...bad, ...ok];
  const head = bad.length ? "training job failed" : "training job finished";
  return parts.length === 1 ? `${head}: ${parts[0]}` : `${bad.length ? "training jobs ended, some badly" : "training jobs finished"}: ${parts.join(", ")}`;
}
const EMPTY = { dataset_root: "", dataset_repo_id: "", fromHub: false, label: "" };
function datasetKey(d) {
  return d.root ?? `hub:${d.repo_id}`;
}
function selectDataset(rows, key) {
  if (!key) return EMPTY;
  const d = rows.find((r) => datasetKey(r) === key);
  if (!d) return EMPTY;
  if (d.root) {
    return {
      dataset_root: d.root,
      dataset_repo_id: "",
      fromHub: false,
      label: `${d.repo_id}${d.total_episodes ? ` (${d.total_episodes} eps)` : ""}`
    };
  }
  return {
    dataset_root: "",
    dataset_repo_id: d.repo_id,
    fromHub: true,
    label: `${d.repo_id} — downloaded from the Hub at start`
  };
}
function selectionKey(sel) {
  if (sel.dataset_root) return sel.dataset_root;
  return sel.dataset_repo_id ? `hub:${sel.dataset_repo_id}` : "";
}
function noEpisodes(d) {
  if (d.usable !== void 0) return false;
  return d.total_episodes === 0 || d.total_frames === 0;
}
const EMPTY_ROW = "0 episodes. meta/info.json is written when a recording session OPENS, before the first episode is captured, so a directory like this is what an abandoned session leaves behind — not a dataset. Record into it, or delete it.";
function replayable(d) {
  if (!d.root) return { ok: false, reason: "on the Hub, not on this machine — training downloads it; replay needs it local" };
  if (d.usable === false) return { ok: false, reason: d.problem ?? "this dataset has no episodes to replay" };
  if (noEpisodes(d)) return { ok: false, reason: EMPTY_ROW };
  return { ok: true, reason: "Replay in a live mesh sim — appears in the fleet grid" };
}
function datasetMark(d) {
  if (d.recording) return { glyph: "⏺ ", kind: "recording" };
  if (d.usable === false) return { glyph: "⚠ ", kind: "problem" };
  if (noEpisodes(d)) return { glyph: "⚠ ", kind: "problem" };
  return { glyph: "", kind: "ok" };
}
function trainable(d) {
  if (!d) return { ok: true, reason: "" };
  if (d.usable === false) return { ok: false, reason: d.problem ?? "this dataset has no episodes to train on" };
  if (noEpisodes(d)) return { ok: false, reason: EMPTY_ROW };
  return { ok: true, reason: "" };
}
function selectedRow(rows, sel) {
  return rows.find((r) => datasetKey(r) === selectionKey(sel)) ?? null;
}
function episodeChoice(d, requested) {
  const total = typeof d.total_episodes === "number" && Number.isFinite(d.total_episodes) ? d.total_episodes : null;
  const countKnown = total !== null && total > 0;
  const raw = typeof requested === "string" ? requested.trim() : requested;
  if (raw === void 0 || raw === null || raw === "") {
    return { ok: true, episode: 0, reason: countKnown ? `episode 0 of ${total}` : "episode 0", countKnown };
  }
  const n = Number(raw);
  if (!Number.isInteger(n)) {
    return { ok: false, episode: 0, reason: `“${raw}” is not a whole episode number`, countKnown };
  }
  if (n < 0) return { ok: false, episode: 0, reason: "episode numbers start at 0", countKnown };
  if (total !== null && total <= 0) {
    return { ok: false, episode: 0, reason: "this dataset records no episodes yet", countKnown };
  }
  if (total !== null && n >= total) {
    return {
      ok: false,
      episode: 0,
      reason: `this dataset has ${total} episode${total === 1 ? "" : "s"}, numbered 0–${total - 1}`,
      countKnown
    };
  }
  return {
    ok: true,
    episode: n,
    reason: countKnown ? `episode ${n} of ${total}` : `episode ${n} (this server does not report a count)`,
    countKnown
  };
}
function datasetHint(input) {
  const q = input.query.trim();
  const shown = input.shownQuery === null ? null : input.shownQuery.trim();
  const stale = shown !== q;
  const auth = input.anonymous && q ? `Hub results are public only${input.authDetail ? ` (${input.authDetail})` : ""} — a private or gated dataset will look like “no match”.` : null;
  if (input.problem && !stale) {
    return { text: `⚠ ${input.problem}`, tone: "warn", auth };
  }
  if (stale) {
    return {
      text: q ? `searching for “${q}”…${input.count > 0 && shown ? ` — the rows below are still the results for “${shown}”` : ""}` : null,
      tone: "pending",
      auth
    };
  }
  if (input.count === 0 && q) {
    return {
      text: `nothing here or on the Hub matches “${q}” — the Hub answered, it simply has no match.`,
      tone: "info",
      auth
    };
  }
  return { text: null, tone: "info", auth };
}
function jobsLedgerNotice({ count, problem }) {
  const trimmed = (problem ?? "").trim();
  if (trimmed) {
    const lead = count > 0 ? "Some earlier runs may be missing from this list" : "This list is empty because the history could not be read, not because nothing ran";
    return { text: `⚠ ${lead} — ${trimmed}`, tone: "warn", partial: true };
  }
  if (count === 0) {
    return { text: "No training jobs yet.", tone: "info", partial: false };
  }
  return { text: null, tone: "info", partial: false };
}
function knownTime(v) {
  return typeof v === "number" && Number.isFinite(v) && v > 0;
}
function orderJobsNewestFirst(jobs) {
  const timed = [];
  const untimed = [];
  for (const j of jobs) (knownTime(j == null ? void 0 : j.submitted_at) ? timed : untimed).push(j);
  timed.sort((a, b) => b.submitted_at - a.submitted_at);
  untimed.reverse();
  return [...timed, ...untimed];
}
function outputDirSay(v) {
  const none = { text: null, tone: "info", confirmable: false, confirmLabel: null, blocked: false };
  if (!v || !v.state) return none;
  const detail = (v.detail ?? "").trim();
  switch (v.state) {
    case "free":
      return none;
    case "occupied":
      return {
        text: `⚠ ${detail}`,
        tone: "bad",
        confirmable: true,
        confirmLabel: `delete ${v.total ?? "the"} item(s) in ${v.path ?? "this directory"} and train here`,
        blocked: false
      };
    case "resumable":
      return { text: `⚠ ${detail}`, tone: "warn", confirmable: false, confirmLabel: null, blocked: true };
    case "not_a_dir":
      return { text: `✗ ${detail}`, tone: "bad", confirmable: false, confirmLabel: null, blocked: true };
    case "unknown":
      return { text: `⚠ ${detail}`, tone: "warn", confirmable: false, confirmLabel: null, blocked: true };
    default:
      return none;
  }
}
function trainGate(args) {
  var _a, _b;
  const path = (args.path ?? "").trim();
  if (!path) return { ok: false, confirmClear: false, why: "an output dir is required" };
  const say = outputDirSay(args.verdict);
  if (say.blocked) {
    return { ok: false, confirmClear: false, why: ((_a = args.verdict) == null ? void 0 : _a.detail) ?? "that output dir cannot be used" };
  }
  if (say.confirmable) {
    const armed = !!args.armedFor && args.armedFor === (((_b = args.verdict) == null ? void 0 : _b.path) ?? path);
    if (!armed) {
      return {
        ok: false,
        confirmClear: false,
        why: "that directory is not empty — tick the box to confirm what gets deleted"
      };
    }
    return { ok: true, confirmClear: true, why: null };
  }
  return { ok: true, confirmClear: false, why: null };
}
function guessPolicyType(baseModel) {
  const m = (baseModel ?? "").replace(/_/g, "-").match(/\b(smolvla|act|diffusion|pi0-fast|pi05|pi0|tdmpc|vqbet)\b/i);
  return m ? m[1].toLowerCase().replace("-", "_").replace("pi0fast", "pi0_fast") : null;
}
function TrainingTab({ onClose, prefill }) {
  var _a;
  const [trainers, setTrainers] = reactExports.useState([]);
  const [unsupported, setUnsupported] = reactExports.useState({});
  const [srvFields, setSrvFields] = reactExports.useState(null);
  const [srvHeard, setSrvHeard] = reactExports.useState(false);
  const sheetRef = reactExports.useRef(null);
  useDialogFocus(sheetRef);
  const [datasets, setDatasets] = reactExports.useState([]);
  const [jobs, setJobs] = reactExports.useState([]);
  const [statuses, setStatuses] = reactExports.useState({});
  const [jobSay, setJobSay] = reactExports.useState("");
  const seenStates = reactExports.useRef({});
  const [polledAt, setPolledAt] = reactExports.useState({});
  const [pollFail, setPollFail] = reactExports.useState({});
  const [nowS, setNowS] = reactExports.useState(() => Date.now() / 1e3);
  const [traces, setTraces] = reactExports.useState({});
  const [form, setForm] = reactExports.useState({ provider: "lerobot_local", dataset_root: (prefill == null ? void 0 : prefill.dataset_root) ?? "", dataset_repo_id: "", base_model: "lerobot/smolvla_base", output_dir: "", steps: "10000", method: "lora", embodiment: "", val_episodes: "" });
  const [dsQuery, setDsQuery] = reactExports.useState("");
  const [dsProblem, setDsProblem] = reactExports.useState(null);
  const [dsShownQuery, setDsShownQuery] = reactExports.useState(null);
  const dsSeq = reactExports.useRef(0);
  const tick = reactExports.useRef(0);
  const tickBusy = reactExports.useRef(false);
  const applied = reactExports.useRef({});
  const [dsAuth, setDsAuth] = reactExports.useState(null);
  const [jobsProblem, setJobsProblem] = reactExports.useState(null);
  const [busy, setBusy] = reactExports.useState(false);
  const [msg, setMsg] = reactExports.useState(null);
  const [episodeBox, setEpisodeBox] = reactExports.useState({});
  const refresh = async () => {
    const seq = ++dsSeq.current;
    try {
      const [t, d, j] = await Promise.all([
        api("/api/training/trainers"),
        api(`/api/training/datasets?q=${encodeURIComponent(dsQuery)}`),
        api("/api/training/jobs")
      ]);
      setTrainers(t.trainers ?? []);
      setUnsupported(t.unsupported ?? {});
      setSrvFields(Array.isArray(t.fields) ? t.fields : null);
      setSrvHeard(true);
      setJobs(orderJobsNewestFirst(j.jobs ?? []));
      setJobsProblem(j.problem ?? null);
      if (isLatestRequest(seq, dsSeq.current)) {
        setDatasets(d.datasets ?? []);
        setDsProblem(d.problem ?? null);
        setDsAuth(d.hf_auth ?? null);
        setDsShownQuery(dsQuery);
      }
    } catch (e) {
      setMsg(`⚠ ${(e == null ? void 0 : e.message) ?? e}`);
    }
  };
  reactExports.useEffect(() => {
    refresh();
  }, []);
  const outSeq = reactExports.useRef(0);
  reactExports.useEffect(() => {
    const path = form.output_dir.trim();
    setClearArmedFor(null);
    if (!path) {
      setOutDir(null);
      return;
    }
    const seq = ++outSeq.current;
    const t = setTimeout(async () => {
      try {
        const v = await api(`/api/training/output-dir?path=${encodeURIComponent(path)}`);
        if (!isLatestRequest(seq, outSeq.current)) return;
        setOutDir(v);
      } catch {
        if (!isLatestRequest(seq, outSeq.current)) return;
        setOutDir(null);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [form.output_dir]);
  reactExports.useEffect(() => {
    const t = setTimeout(async () => {
      const seq = ++dsSeq.current;
      const asked = dsQuery;
      try {
        const d = await api(`/api/training/datasets?q=${encodeURIComponent(asked)}`);
        if (!isLatestRequest(seq, dsSeq.current)) return;
        setDatasets(d.datasets ?? []);
        setDsProblem(d.problem ?? null);
        setDsAuth(d.hf_auth ?? null);
        setDsShownQuery(asked);
      } catch (e) {
        if (!isLatestRequest(seq, dsSeq.current)) return;
        setDsProblem(`search failed: ${(e == null ? void 0 : e.message) ?? e}`);
        setDsShownQuery(asked);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [dsQuery]);
  reactExports.useEffect(() => {
    var _a2;
    const now = {};
    for (const [id, st] of Object.entries(statuses)) {
      const state = (_a2 = st == null ? void 0 : st.data) == null ? void 0 : _a2.status;
      if (typeof state === "string" && state) now[id] = state;
    }
    const said = jobTransitions(seenStates.current, now);
    seenStates.current = { ...seenStates.current, ...now };
    if (said) setJobSay(said);
  }, [statuses]);
  reactExports.useEffect(() => {
    const id = setInterval(async () => {
      var _a2;
      if (tickBusy.current) return;
      tickBusy.current = true;
      const round = ++tick.current;
      try {
        for (const job of jobs.slice(0, 5)) {
          if (!job.job_id) continue;
          try {
            const s = await api(`/api/training/status?provider=${job.provider}&job_id=${encodeURIComponent(job.job_id)}`);
            if (!newerThanApplied(round, applied.current[job.job_id])) continue;
            applied.current[job.job_id] = round;
            setStatuses((prev) => ({ ...prev, [job.job_id]: s }));
            setPolledAt((prev) => ({ ...prev, [job.job_id]: Date.now() / 1e3 }));
            setPollFail((prev) => prev[job.job_id] ? { ...prev, [job.job_id]: { n: 0, msg: "" } } : prev);
            const m = (_a2 = s == null ? void 0 : s.data) == null ? void 0 : _a2.metrics;
            if (m) setTraces((prev) => ({
              ...prev,
              [job.job_id]: pushLoss(prev[job.job_id] ?? [], m.latest_step, m.latest_loss)
            }));
          } catch (e) {
            const msg2 = String((e == null ? void 0 : e.message) ?? e).slice(0, 120);
            setPollFail((prev) => {
              var _a3;
              return { ...prev, [job.job_id]: { n: (((_a3 = prev[job.job_id]) == null ? void 0 : _a3.n) ?? 0) + 1, msg: msg2 } };
            });
          }
        }
      } finally {
        tickBusy.current = false;
      }
    }, 5e3);
    return () => clearInterval(id);
  }, [jobs]);
  reactExports.useEffect(() => {
    const id = setInterval(() => setNowS(Date.now() / 1e3), 5e3);
    return () => clearInterval(id);
  }, []);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const failed = (kind, e) => {
    const v = sideEffectVerdict({
      kind,
      status: e instanceof HttpError ? e.status : 0,
      message: (e == null ? void 0 : e.message) ?? String(e)
    });
    setMsg(v.text);
    if (v.delivered === "unknown") refresh();
  };
  const submit = async (validateOnly) => {
    var _a2, _b;
    const picked = selectedRow(datasets, form);
    const can = trainable(picked);
    if (!can.ok) {
      if (dsOverride !== selectionKey(form)) {
        setDsWarn({ key: selectionKey(form), reason: can.reason, recording: (picked == null ? void 0 : picked.recording) === true });
        setMsg(null);
        return;
      }
      setDsOverride(null);
    }
    if (!validateOnly && !gate.ok) {
      setMsg(`✗ ${gate.why}`);
      return;
    }
    setBusy(true);
    setMsg(null);
    const body = {
      provider: form.provider,
      dataset_root: form.dataset_root || void 0,
      dataset_repo_id: form.dataset_repo_id || void 0,
      base_model: form.base_model || void 0,
      output_dir: form.output_dir || void 0,
      ...gate.confirmClear ? { confirm_clear: true } : {},
      steps: wantedSteps.value,
      method: form.method || void 0,
      ...extraFields(form.provider).some((f) => f.key === "embodiment") && form.embodiment.trim() ? { embodiment: form.embodiment.trim() } : {},
      // Held out only when the operator asked for it: `null` and an absent key both mean "train
      // on every episode", and sending 0 would show a split in the form that the backend drops.
      ...wantedHoldout.send !== null && holdoutSupport.ok ? { val_episodes: wantedHoldout.send } : {}
    };
    try {
      const j = await post(validateOnly ? "/api/training/validate" : "/api/training/submit", body);
      setMsg(j.status === "success" ? `✓ ${(_a2 = j.text) == null ? void 0 : _a2.slice(0, 200)}` : `✗ ${(_b = j.text) == null ? void 0 : _b.slice(0, 300)}`);
      if (!validateOnly && j.status === "success") refresh();
    } catch (e) {
      failed(validateOnly ? "export" : "training", e);
    }
    setBusy(false);
  };
  const [dsWarn, setDsWarn] = reactExports.useState(null);
  const [dsOverride, setDsOverride] = reactExports.useState(null);
  const [outDir, setOutDir] = reactExports.useState(null);
  const [clearArmedFor, setClearArmedFor] = reactExports.useState(null);
  const outSay = outputDirSay(outDir);
  const gate = trainGate({ path: form.output_dir, verdict: outDir, armedFor: clearArmedFor });
  const [collect, setCollect] = reactExports.useState({ dataset_root: "", instruction: "pick up the red cube", n_episodes: "5", duration: "10", robot_name: "so101" });
  const [showCollect, setShowCollect] = reactExports.useState(false);
  const STEP_RULES = { what: "steps", min: 1, max: 2e6, remedy: "submit a shorter run" };
  const wantedSteps = numField(form.steps, STEP_RULES);
  const [labelsFor, setLabelsFor] = reactExports.useState(null);
  const [labelData, setLabelData] = reactExports.useState(null);
  const [labelErr, setLabelErr] = reactExports.useState(null);
  async function openLabels(d) {
    const key = datasetKey(d);
    if (labelsFor === key) {
      setLabelsFor(null);
      return;
    }
    setLabelsFor(key);
    setLabelData(null);
    setLabelErr(null);
    const path = "/api/datasets/labels";
    try {
      setLabelData(await api(`${path}?root=${encodeURIComponent(d.root || "")}`));
    } catch (e) {
      setLabelErr(e instanceof HttpError ? e.message : String(e));
    }
  }
  const wantedHoldout = holdout(form.val_episodes, ((_a = selectedRow(datasets, form)) == null ? void 0 : _a.total_episodes) ?? null);
  const holdoutSupport = fieldSupport(srvFields, "val_episodes", srvHeard);
  const wantedEpisodes = numField(collect.n_episodes, { what: "episodes", min: 1, max: 500, remedy: "collect in batches" });
  const wantedSeconds = numField(collect.duration, { what: "seconds per episode", min: 1, max: 600 });
  const submitCollect = async () => {
    if (!collect.dataset_root.trim()) return;
    setBusy(true);
    setMsg(null);
    try {
      const j = await post("/api/collect", {
        dataset_root: collect.dataset_root,
        instruction: collect.instruction,
        n_episodes: wantedEpisodes.value,
        duration: wantedSeconds.value,
        robot_name: collect.robot_name
      });
      setMsg(j.peer_id ? `▶ collecting ${j.n_episodes} episodes as ${j.peer_id} — watch it in the fleet grid; dataset appears below when done` : `⚠ ${JSON.stringify(j).slice(0, 200)}`);
      if (j.peer_id) setTimeout(refresh, 15e3);
    } catch (e) {
      failed("collect", e);
    }
    setBusy(false);
  };
  const replay = async (d) => {
    if (!d.root) {
      setMsg(`⚠ ${d.repo_id} is on the Hub, not on this machine — train with it (the trainer downloads it) or clone it locally to replay`);
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      const choice = episodeChoice(d, episodeBox[datasetKey(d)]);
      if (!choice.ok) {
        setMsg(`⚠ ${choice.reason}`);
        return;
      }
      const j = await post("/api/replay", { repo_id: d.repo_id, root: d.root, episode: choice.episode });
      setMsg(j.peer_id ? `▶ replaying ${d.repo_id} ep${choice.episode} as ${j.peer_id} — watch it in the fleet grid` : `⚠ ${JSON.stringify(j).slice(0, 200)}`);
    } catch (e) {
      failed("replay", e);
    }
    setBusy(false);
  };
  const [stageAnyway, setStageAnyway] = reactExports.useState(null);
  const exportCkpt = async (job) => {
    var _a2, _b;
    setBusy(true);
    try {
      const j = await post("/api/training/export", { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model });
      const art = j == null ? void 0 : j.artifact;
      if (j.status !== "success") setMsg(`✗ ${(_a2 = j.text) == null ? void 0 : _a2.slice(0, 250)}`);
      else if (j.deployable === false) {
        setMsg(`⚠ the export succeeded but the artifact is not usable: ${(art == null ? void 0 : art.message) ?? "the checkpoint could not be confirmed on disk"}`);
      } else setMsg(`✓ ${(_b = j.text) == null ? void 0 : _b.slice(0, 250)}${(art == null ? void 0 : art.warning) ? ` — ⚠ ${art.warning}` : ""}`);
    } catch (e) {
      failed("export", e);
    }
    setBusy(false);
  };
  const deployCkpt = async (job) => {
    var _a2, _b;
    setBusy(true);
    try {
      const j = await post("/api/training/export", { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model });
      const ckpt = (_a2 = j == null ? void 0 : j.data) == null ? void 0 : _a2.exported_model;
      const art = j == null ? void 0 : j.artifact;
      if (j.status !== "success" || typeof ckpt !== "string" || !ckpt) {
        setMsg(`✗ nothing deployable: ${((_b = j.text) == null ? void 0 : _b.slice(0, 200)) ?? "export returned no artifact path"}`);
      } else if (j.deployable === false) {
        setStageAnyway({ job, ckpt, message: (art == null ? void 0 : art.message) ?? "the checkpoint could not be confirmed on disk" });
        setMsg(null);
      } else {
        setDeployIntent({
          checkpoint: ckpt,
          policy_type: guessPolicyType(job.base_model),
          source: `training job ${job.job_id} (${job.base_model || job.provider})`
        });
        setMsg("🚀 checkpoint staged — close this sheet and open a robot’s run form: it will be prefilled, and nothing runs until you press Run there");
      }
    } catch (e) {
      failed("export", e);
    }
    setBusy(false);
  };
  const stageRegardless = () => {
    if (!stageAnyway) return;
    const { job, ckpt } = stageAnyway;
    setDeployIntent({
      checkpoint: ckpt,
      policy_type: guessPolicyType(job.base_model),
      source: `training job ${job.job_id} (${job.base_model || job.provider}) — staged over an unconfirmed artifact`
    });
    setStageAnyway(null);
    setMsg("🚀 checkpoint staged over the warning — open a robot’s run form; nothing runs until you press Run there");
  };
  const datasetPicked = form.dataset_root || form.dataset_repo_id;
  const datasetLabel = selectDataset(datasets, selectionKey(form)).label || null;
  const stepsPhrase = wantedSteps.problem ? `an unset number of` : fmtStep(wantedSteps.value);
  const missingExtra = missingForProvider(form.provider, form);
  const story = datasetPicked ? `Fine-tune ${form.base_model || "lerobot/smolvla_base"} on ${datasetLabel ?? datasetPicked} for ${stepsPhrase} steps (${form.method}), saving to ${form.output_dir || "…pick an output dir"}.${wantedHoldout.send ? ` Holding out the last ${wantedHoldout.send} episodes to score it.` : ""}` + (missingExtra ? ` This will be refused until you fill it in: ${missingExtra}.` : "") : "Pick a dataset to begin — the plan reads back here before anything runs.";
  return (
    /**
     * role + label like RecordPanel's sheet: this is a full-bleed layer over the fleet, and a
     * screen reader that is not told it entered a dialog reads it as more of the page it just
     * left.
     */
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { ref: sheetRef, className: "train-sheet", role: "dialog", "aria-label": "Training", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-head", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "🎓 Training" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "dock-min", onClick: onClose, "aria-label": "close training", title: "Escape", children: "✕" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-form", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: `train-story${datasetPicked ? "" : " empty"}`, children: story }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "provider" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("select", { value: form.provider, onChange: (e) => set("provider", e.target.value), disabled: busy, children: trainers.map((t) => /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "option",
            {
              disabled: t in unsupported,
              title: unsupported[t] ?? void 0,
              children: [
                t,
                t in unsupported ? " — not from this form" : ""
              ]
            },
            t
          )) })
        ] }),
        Object.keys(unsupported).length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "hint", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("summary", { children: [
            Object.keys(unsupported).length,
            " provider",
            Object.keys(unsupported).length === 1 ? " is" : "s are",
            " not trainable from this form — why?"
          ] }),
          [...new Set(Object.values(unsupported))].map((reason2) => /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
            Object.keys(unsupported).filter((k) => unsupported[k] === reason2).sort().join(" and "),
            " cannot be trained from here: ",
            reason2,
            "."
          ] }, reason2))
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "dataset" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: dsQuery,
              onChange: (e) => setDsQuery(e.target.value),
              disabled: busy,
              placeholder: "search this machine and the Hub — e.g. pusht, so101, your org"
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              value: selectionKey(form),
              onChange: (e) => {
                const sel = selectDataset(datasets, e.target.value);
                setForm((f) => ({ ...f, dataset_root: sel.dataset_root, dataset_repo_id: sel.dataset_repo_id }));
              },
              disabled: busy,
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "— pick a dataset —" }),
                datasets.filter((d) => d.local !== false).length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("optgroup", { label: "on this machine", children: datasets.filter((d) => d.local !== false).map((d) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: datasetKey(d), children: [
                  datasetMark(d).glyph,
                  d.repo_id,
                  " (",
                  d.total_episodes ?? "?",
                  " eps",
                  d.robot_type && d.robot_type !== "unknown" ? `, ${d.robot_type}` : "",
                  ")"
                ] }, datasetKey(d))) }),
                datasets.filter((d) => d.local === false).length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("optgroup", { label: "HuggingFace Hub — downloaded when training starts", children: datasets.filter((d) => d.local === false).map((d) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: datasetKey(d), children: [
                  d.repo_id,
                  d.downloads ? ` · ${d.downloads.toLocaleString()} downloads` : ""
                ] }, datasetKey(d))) })
              ]
            }
          ),
          (() => {
            const h = datasetHint({
              query: dsQuery,
              shownQuery: dsShownQuery,
              count: datasets.length,
              problem: dsProblem,
              anonymous: (dsAuth == null ? void 0 : dsAuth.authenticated) === false,
              authDetail: (dsAuth == null ? void 0 : dsAuth.detail) ?? null
            });
            return /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              h.text && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `hint${h.tone === "warn" ? " warn" : ""}`, role: "status", "aria-live": "polite", children: h.text }),
              h.auth && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", children: h.auth })
            ] });
          })()
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "base model" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("input", { value: form.base_model, onChange: (e) => set("base_model", e.target.value), disabled: busy, placeholder: "lerobot/smolvla_base" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "output dir" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: form.output_dir,
              onChange: (e) => set("output_dir", e.target.value),
              disabled: busy,
              placeholder: "/tmp/my_policy_ckpt",
              "aria-describedby": "train-outdir-say",
              "aria-invalid": outSay.blocked || outSay.confirmable
            }
          ),
          (() => {
            const sug = suggestOutputDir(form, form.output_dir);
            return sug ? /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                type: "button",
                className: "btn ghost suggest",
                disabled: busy,
                onClick: () => set("output_dir", sug),
                children: [
                  "use ",
                  sug
                ]
              }
            ) : null;
          })(),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "train-outdir-say", className: `fieldsay${outSay.tone === "info" ? "" : " bad"}`, role: "status", "aria-live": "polite", children: outSay.text ?? "" })
        ] }),
        outSay.confirmable && /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field check consent-clear", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "checkbox",
              checked: clearArmedFor === ((outDir == null ? void 0 : outDir.path) ?? form.output_dir.trim()),
              onChange: (e) => setClearArmedFor(e.target.checked ? (outDir == null ? void 0 : outDir.path) ?? form.output_dir.trim() : null),
              disabled: busy
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: outSay.confirmLabel })
        ] }),
        extraFields(form.provider).map((f) => /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: f.label }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              value: form[f.key],
              onChange: (e) => set(f.key, e.target.value),
              disabled: busy,
              placeholder: f.placeholder,
              "aria-describedby": `train-${f.key}-say`,
              "aria-invalid": f.required && !form[f.key].trim()
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: `train-${f.key}-say`, className: `fieldsay${f.required && !form[f.key].trim() ? " bad" : ""}`, children: f.say })
        ] }, f.key)),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "steps" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                value: form.steps,
                onChange: (e) => set("steps", e.target.value),
                disabled: busy,
                "aria-invalid": !!wantedSteps.problem,
                "aria-describedby": "train-steps-say"
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "train-steps-say", className: `fieldsay${wantedSteps.problem ? " bad" : ""}`, children: wantedSteps.problem ?? wantedSteps.note ?? "" })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "val episodes" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                value: holdoutSupport.ok ? form.val_episodes : "",
                placeholder: "none",
                onChange: (e) => set("val_episodes", e.target.value),
                disabled: busy || !holdoutSupport.ok,
                "aria-invalid": !!wantedHoldout.problem,
                "aria-describedby": "train-val-say"
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "train-val-say", className: `fieldsay${wantedHoldout.problem || !holdoutSupport.ok ? " bad" : ""}`, children: holdoutSupport.why || wantedHoldout.problem || wantedHoldout.say })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "method" }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: form.method, onChange: (e) => set("method", e.target.value), disabled: busy, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "lora", children: "lora" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "full", children: "full" })
            ] })
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-actions", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => submit(true), disabled: busy || !!wantedSteps.problem, children: "✓ validate" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn go wide",
              onClick: () => submit(false),
              disabled: busy || !datasetPicked || !!wantedSteps.problem || holdoutSupport.ok && !!wantedHoldout.problem || !gate.ok,
              title: gate.why ?? void 0,
              children: outSay.confirmable && gate.ok ? "▶ delete and train" : "▶ train"
            }
          )
        ] }),
        msg && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg", children: msg }),
        dsWarn && dsWarn.key === selectionKey(form) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn artifact-hold", role: "alert", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            dsWarn.recording ? "⏺" : "⚠",
            " not started: ",
            dsWarn.reason
          ] }),
          dsWarn.recording && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "jstate", children: "the record screen shows this session's progress; training can start the moment it closes" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "artifact-hold-actions", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => {
              setDsWarn(null);
              setDsOverride(null);
            }, children: "pick another dataset" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn",
                onClick: () => {
                  setDsOverride(dsWarn.key);
                  setDsWarn(null);
                  setMsg("⚠ dataset warning overridden — press start training again");
                },
                title: dsWarn.recording ? "the trainer would read episodes as they are still being written - only useful if the session is about to close" : "the check reads metadata only - insist if you know the episodes are there",
                children: "train on it anyway"
              }
            )
          ] })
        ] }),
        stageAnyway && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn artifact-hold", role: "alert", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
            "⚠ not staged: ",
            stageAnyway.message
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "artifact-hold-actions", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => setStageAnyway(null), children: "keep it unstaged" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn",
                onClick: stageRegardless,
                title: "the disk check can be wrong - stage it and find out on the run form",
                children: "stage it anyway"
              }
            )
          ] })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-form", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn ghost", onClick: () => setShowCollect((s) => !s), children: [
          showCollect ? "▾" : "▸",
          " 📹 collect new dataset (sim rollouts)"
        ] }),
        showCollect && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "dataset root (path)" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                value: collect.dataset_root,
                placeholder: "/tmp/my_demos",
                onChange: (e) => setCollect((c) => ({ ...c, dataset_root: e.target.value })),
                disabled: busy
              }
            )
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "instruction" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                value: collect.instruction,
                onChange: (e) => setCollect((c) => ({ ...c, instruction: e.target.value })),
                disabled: busy
              }
            )
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-row", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "episodes" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "number",
                  value: collect.n_episodes,
                  "aria-invalid": !!wantedEpisodes.problem,
                  "aria-describedby": "collect-episodes-say",
                  onChange: (e) => setCollect((c) => ({ ...c, n_episodes: e.target.value })),
                  disabled: busy
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "collect-episodes-say", className: `fieldsay${wantedEpisodes.problem ? " bad" : ""}`, children: wantedEpisodes.problem ?? wantedEpisodes.note ?? "" })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "sec/episode" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "number",
                  value: collect.duration,
                  "aria-invalid": !!wantedSeconds.problem,
                  "aria-describedby": "collect-seconds-say",
                  onChange: (e) => setCollect((c) => ({ ...c, duration: e.target.value })),
                  disabled: busy
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "collect-seconds-say", className: `fieldsay${wantedSeconds.problem ? " bad" : ""}`, children: wantedSeconds.problem ?? wantedSeconds.note ?? "" })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "robot" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  value: collect.robot_name,
                  onChange: (e) => setCollect((c) => ({ ...c, robot_name: e.target.value })),
                  disabled: busy
                }
              )
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn go wide",
              onClick: submitCollect,
              disabled: busy || !collect.dataset_root.trim() || !!wantedEpisodes.problem || !!wantedSeconds.problem,
              children: "📹 collect"
            }
          ) })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-jobs", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Datasets" }),
        datasets.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "dock-hint", children: dsShownQuery !== null && dsShownQuery.trim() !== dsQuery.trim() ? `Searching for “${dsQuery.trim()}”…` : dsProblem ? `No local LeRobotDatasets, and the Hub could not be searched — ${dsProblem}` : dsQuery ? `Nothing matches “${dsQuery}” here or on the Hub.` : "No LeRobotDatasets on this machine — type above to search the Hub, or record one in Collect." }),
        datasets.map((d) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-job", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-job-head", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: d.repo_id }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "jstate", children: d.recording ? "⏺ recording now" : d.root ? `${d.total_episodes ?? "?"} eps · ${d.fps ?? "?"} fps` : `Hub${d.downloads ? ` · ${d.downloads.toLocaleString()} downloads` : ""}` })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-job-actions", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                className: "ep-box",
                type: "number",
                min: 0,
                inputMode: "numeric",
                value: episodeBox[datasetKey(d)] ?? "",
                placeholder: typeof d.total_episodes === "number" && d.total_episodes > 0 ? `0–${d.total_episodes - 1}` : "0",
                "aria-label": `episode to replay from ${d.repo_id}`,
                title: typeof d.total_episodes === "number" && d.total_episodes > 0 ? `This dataset has ${d.total_episodes} episode${d.total_episodes === 1 ? "" : "s"} — blank replays episode 0` : "Episode index — blank replays episode 0",
                disabled: busy || !replayable(d).ok,
                onChange: (e) => setEpisodeBox((prev) => ({ ...prev, [datasetKey(d)]: e.target.value }))
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn ghost",
                onClick: () => replay(d),
                disabled: busy || !replayable(d).ok,
                title: episodeChoice(d, episodeBox[datasetKey(d)]).ok ? `${replayable(d).reason} — ${episodeChoice(d, episodeBox[datasetKey(d)]).reason}` : episodeChoice(d, episodeBox[datasetKey(d)]).reason,
                children: "🎬 replay in sim"
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn ghost",
                onClick: () => openLabels(d),
                disabled: !labelsGate(d).ok,
                title: labelsGate(d).reason,
                "aria-expanded": labelsFor === datasetKey(d),
                children: "🏷 labels"
              }
            )
          ] }),
          labelsFor === datasetKey(d) && (() => {
            const sum = labelSummary(labelData, labelErr);
            return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "ds-labels", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: sum.tone === "warn" ? "dock-hint warn" : "dock-hint", children: sum.text }),
              ((labelData == null ? void 0 : labelData.episodes) ?? []).map((ep) => {
                const line = labelRowLine(ep);
                return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: line.muted ? "ds-label-row muted" : "ds-label-row", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "ds-label-badge", children: line.badge }),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
                    "episode ",
                    ep.episode_index
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "jstate", children: line.detail })
                ] }, ep.episode_index);
              })
            ] });
          })()
        ] }, datasetKey(d))),
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { children: "Jobs" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sr-only", role: "status", "aria-live": "polite", "aria-atomic": "true", children: jobSay }),
        (() => {
          const notice = jobsLedgerNotice({ count: jobs.length, problem: jobsProblem });
          return notice.text ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: notice.tone === "warn" ? "dock-hint warn" : "dock-hint", children: notice.text }) : null;
        })(),
        jobs.map((job) => {
          var _a2, _b, _c, _d, _e;
          const st = statuses[job.job_id];
          const state = ((_a2 = st == null ? void 0 : st.data) == null ? void 0 : _a2.status) ?? "…";
          const fresh = trainingFreshness({
            polledAtS: polledAt[job.job_id],
            nowS,
            failures: (_b = pollFail[job.job_id]) == null ? void 0 : _b.n,
            error: ((_c = pollFail[job.job_id]) == null ? void 0 : _c.msg) || null,
            state: ((_d = st == null ? void 0 : st.data) == null ? void 0 : _d.status) ?? null
          });
          return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `train-job${fresh.stale ? " stalefeed" : ""}`, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-job-head", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: job.provider }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `jstate ${state}`, title: fresh.title, children: state }),
              fresh.stale && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "jstale", title: fresh.title, children: [
                "as of ",
                fresh.ageS != null && fresh.ageS < 90 ? `${Math.round(fresh.ageS)}s` : `${Math.round((fresh.ageS ?? 0) / 60)}m`,
                " ago"
              ] })
            ] }),
            fresh.note && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg warn", children: fresh.note }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-job-meta", children: [
              (_e = job.dataset) == null ? void 0 : _e.split("/").slice(-2).join("/"),
              " → ",
              job.output_dir,
              " · ",
              job.steps,
              " steps"
            ] }),
            (() => {
              var _a3;
              const m = (_a3 = st == null ? void 0 : st.data) == null ? void 0 : _a3.metrics;
              if (!m || Object.keys(m).length === 0) return null;
              const step = typeof m.latest_step === "number" ? m.latest_step : null;
              const total = typeof job.steps === "number" ? job.steps : Number(job.steps) || null;
              return /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
                step !== null && total ? /* @__PURE__ */ jsxRuntimeExports.jsxs(
                  "div",
                  {
                    className: "train-progress",
                    role: "progressbar",
                    "aria-valuemin": 0,
                    "aria-valuemax": total,
                    "aria-valuenow": Math.min(step, total),
                    children: [
                      /* @__PURE__ */ jsxRuntimeExports.jsx(
                        "div",
                        {
                          className: "train-progress-fill",
                          style: { width: `${Math.min(100, step / total * 100)}%` }
                        }
                      ),
                      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "train-progress-label", children: [
                        fmtStep(step),
                        " / ",
                        fmtStep(total),
                        " steps"
                      ] })
                    ]
                  }
                ) : null,
                /* @__PURE__ */ jsxRuntimeExports.jsx(LossSpark, { trace: traces[job.job_id] ?? [] }),
                m.latest_loss !== void 0 && !Number.isFinite(m.latest_loss) && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg", children: "⚠ loss is NaN — the run is executing but NOT learning (check LR / data)" }),
                m.liveness_ok === false && state === "running" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg", children: "⚠ no step lines in the log yet — still warming up, or stalled" })
              ] });
            })(),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-job-actions", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => exportCkpt(job), disabled: busy, children: "📦 export checkpoint" }),
              state === "success" && /* @__PURE__ */ jsxRuntimeExports.jsx(
                "button",
                {
                  className: "btn ghost",
                  onClick: () => deployCkpt(job),
                  disabled: busy,
                  title: "stages this checkpoint into a robot's run form — never starts it",
                  children: "🚀 deploy…"
                }
              )
            ] })
          ] }, job.job_id ?? Math.random());
        })
      ] })
    ] })
  );
}
let _module = null;
const TWIN_URL = "/static/twin.js";
function loadTwinModule() {
  if (!_module) {
    _module = import(
      /* @vite-ignore */
      TWIN_URL
    ).catch((e) => {
      _module = null;
      throw e;
    });
  }
  return _module;
}
async function makeTwin(canvas, sessionId) {
  const { Twin } = await loadTwinModule();
  const token = authToken();
  const base = apiUrl("");
  return new Twin(canvas, sessionId, { base, headers: token ? { Authorization: `Bearer ${token}` } : {} });
}
async function simRobots() {
  try {
    const f = await api("/api/fleet?mode=sim");
    const rows = (f.robots ?? []).filter((r) => r && typeof r.name === "string" && (r.has_sim ?? true));
    if (rows.length) return rows.map((r) => ({ name: r.name, label: `${r.name}${r.joints ? ` · ${r.joints} dof` : ""}${r.model_local === false ? " · asset not downloaded" : ""}` }));
  } catch {
  }
  const reg = await api("/api/robots/registry");
  return (reg.robots ?? []).filter((r) => r && r.has_sim).map((r) => ({ name: r.name, label: `${r.name}${r.joints ? ` · ${r.joints} dof` : ""}` }));
}
function lockoutText(l) {
  if (!l) return "lockout not read yet";
  if (l.state === "locked") return `e-stop engaged · ${l.by || "dashboard"} · ${l.reason ?? ""}`;
  if (l.state === "unknown") return `lockout unknown · ${l.reason ?? ""}`;
  return `clear · ${l.reason ?? ""}`;
}
function SimTab({ onClose }) {
  const [robots, setRobots] = reactExports.useState([]);
  const [robot, setRobot] = reactExports.useState("so101");
  const [ports2, setPorts] = reactExports.useState([]);
  const [mirror, setMirror] = reactExports.useState("");
  const [sessions, setSessions] = reactExports.useState([]);
  const [lockout, setLockout] = reactExports.useState(null);
  const [msg, setMsg] = reactExports.useState(null);
  const [starting, setStarting] = reactExports.useState(false);
  const readLockout = reactExports.useCallback(async () => {
    try {
      setLockout((await api("/api/safety")).lockout);
    } catch {
    }
  }, []);
  const failed = reactExports.useCallback(async (e) => {
    setMsg(e instanceof HttpError && e.status === 423 ? `refused: ${e.message}` : (e == null ? void 0 : e.message) ?? String(e));
    await readLockout();
  }, [readLockout]);
  const refresh = reactExports.useCallback(async () => {
    try {
      const { sessions: sessions2 } = await api("/api/sim");
      setSessions(sessions2);
    } catch (e) {
      await failed(e);
    }
    await readLockout();
  }, [failed, readLockout]);
  reactExports.useEffect(() => {
    void refresh();
    simRobots().then((list) => {
      setRobots(list);
      if (list.length && !list.some((r) => r.name === "so101")) setRobot(list[0].name);
    }).catch((e) => setMsg(`registry: ${(e == null ? void 0 : e.message) ?? e}`));
    api("/api/sim/ports").then((r) => setPorts(r.ports ?? [])).catch(() => setPorts([]));
  }, [refresh]);
  const start = async (e) => {
    e.preventDefault();
    setStarting(true);
    setMsg(null);
    try {
      const body = { robot };
      if (mirror) body.mirror = { port: mirror };
      await post("/api/sim", body);
      await refresh();
    } catch (err) {
      await failed(err);
    } finally {
      setStarting(false);
    }
  };
  const toggleEstop = async () => {
    const action = (lockout == null ? void 0 : lockout.state) === "locked" ? "resume" : "estop";
    try {
      setLockout((await post(`/api/safety/${action}`)).lockout);
      setMsg(null);
    } catch (e) {
      await failed(e);
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-sheet sim-sheet", role: "dialog", "aria-label": "Simulation", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "🧊 Simulation" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "dock-min", onClick: onClose, "aria-label": "close simulation", title: "Escape", children: "✕" })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("form", { className: "train-form sim-new", onSubmit: start, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-row", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "robot" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: robot, onChange: (e) => setRobot(e.target.value), disabled: starting, children: [
            !robots.some((r) => r.name === robot) && /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: robot, children: robot }),
            robots.map((r) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: r.name, children: r.label }, r.name))
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "source" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("select", { value: mirror, onChange: (e) => setMirror(e.target.value), disabled: starting, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "physics (steps in this process)" }),
            ports2.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: p.port, children: [
              "mirror ",
              p.port.replace(/^\/dev\//, ""),
              p.likely_servo_bus ? " · servo bus" : ""
            ] }, p.port))
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-actions sim-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", type: "submit", disabled: starting || (lockout == null ? void 0 : lockout.state) === "locked", children: starting ? "starting…" : "start" }) })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `sim-lockout ${(lockout == null ? void 0 : lockout.state) ?? "unknown"}`, role: "status", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: lockoutText(lockout) }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", className: (lockout == null ? void 0 : lockout.state) === "locked" ? "btn go" : "btn danger", onClick: toggleEstop, children: (lockout == null ? void 0 : lockout.state) === "locked" ? "RESUME" : "E-STOP (sim)" })
      ] }),
      msg && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg warn", role: "alert", children: msg }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "A mirror session reads the servo bus at that port and never writes it: the arm decides the pose, the twin follows. A physics session accepts joint targets from the sliders below and from the agent (which asks first)." })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sim-sessions", children: [
      sessions.length === 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "No session yet. Pick a robot and press start - it steps in this process and streams here." }),
      sessions.map((s) => /* @__PURE__ */ jsxRuntimeExports.jsx(SessionCard, { initial: s, onLockout: setLockout, onGone: refresh, onFailed: failed }, s.id))
    ] })
  ] });
}
function SessionCard({ initial, onLockout, onGone, onFailed }) {
  const [snap, setSnap] = reactExports.useState(initial);
  const [view, setView] = reactExports.useState("twin");
  const [twinErr, setTwinErr] = reactExports.useState(null);
  const [targets, setTargets] = reactExports.useState({});
  const canvasRef = reactExports.useRef(null);
  const twinRef = reactExports.useRef(null);
  const sendTimer = reactExports.useRef(null);
  const pending = reactExports.useRef({});
  const mirror = !!snap.source && snap.source.startsWith("real:");
  const live = snap.state !== "stopped" && snap.state !== "error";
  reactExports.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let disposed = false;
    makeTwin(canvas, initial.id).then((t) => {
      if (disposed) {
        t.dispose();
        return;
      }
      twinRef.current = t;
      t.load().catch((e) => setTwinErr(`twin: ${(e == null ? void 0 : e.message) ?? e}`));
    }).catch((e) => setTwinErr(`twin unavailable: ${(e == null ? void 0 : e.message) ?? e}`));
    return () => {
      var _a;
      disposed = true;
      (_a = twinRef.current) == null ? void 0 : _a.dispose();
      twinRef.current = null;
    };
  }, [initial.id]);
  reactExports.useEffect(() => {
    const ws = new WebSocket(wsUrl(`/ws/telemetry/${initial.id}?poses=1`));
    ws.binaryType = "arraybuffer";
    ws.onmessage = (ev) => {
      var _a;
      if (ev.data instanceof ArrayBuffer) {
        (_a = twinRef.current) == null ? void 0 : _a.poses(ev.data);
        return;
      }
      let m;
      try {
        m = JSON.parse(ev.data);
      } catch {
        return;
      }
      setSnap(m);
      if (m.lockout) onLockout(m.lockout);
      if (m.state === "stopped" || m.state === "error") ws.close();
    };
    ws.onclose = (ev) => {
      if (ev.code === 4404) void onGone();
    };
    return () => ws.close();
  }, [initial.id, onLockout, onGone]);
  const setJoint = (name, value) => {
    setTargets((t) => ({ ...t, [name]: value }));
    pending.current[name] = value;
    if (sendTimer.current != null) return;
    sendTimer.current = window.setTimeout(async () => {
      sendTimer.current = null;
      const positions = pending.current;
      pending.current = {};
      try {
        await post(`/api/sim/${initial.id}/joints`, { positions });
      } catch (e) {
        setTargets({});
        await onFailed(e);
      }
    }, 80);
  };
  const reset = async () => {
    setTargets({});
    try {
      await post(`/api/sim/${initial.id}/reset`);
    } catch (e) {
      await onFailed(e);
    }
  };
  const stop = async () => {
    try {
      await del(`/api/sim/${initial.id}`);
    } catch (e) {
      await onFailed(e);
    }
    await onGone();
  };
  const camSrc = view === "cam" ? apiUrl(`/api/sim/${initial.id}/stream.mjpg`) : "";
  const busLine = snap.bus ? snap.error || snap.bus.error || (snap.bus.age_ms == null ? "waiting for the bus" : `read ${snap.bus.age_ms} ms ago · torque untouched`) : snap.error;
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("article", { className: `sim-session ${snap.state}`, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sim-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: snap.robot }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge mono", children: snap.id }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge ${snap.state === "frozen" || snap.state === "error" ? "danger" : snap.state === "running" ? "" : "warn"}`, children: snap.state }),
      mirror && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge warn", title: `Reads the servo bus at ${snap.source.slice(5)}; never writes it`, children: "mirror · read-only" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "spacer" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "viewsel", role: "tablist", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", role: "tab", className: `chip${view === "twin" ? " on" : ""}`, "aria-selected": view === "twin", onClick: () => setView("twin"), children: "Twin" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", role: "tab", className: `chip${view === "cam" ? " on" : ""}`, "aria-selected": view === "cam", onClick: () => setView("cam"), children: "Camera" })
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sim-view", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("canvas", { ref: canvasRef, className: "twin", hidden: view !== "twin" }),
      view === "cam" && /* @__PURE__ */ jsxRuntimeExports.jsx("img", { className: "cam", alt: `${snap.robot} camera`, src: camSrc }),
      twinErr && view === "twin" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint warn sim-overlay", children: twinErr })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "sim-joints", children: snap.joint_names.map((name, i) => {
      const q = snap.qpos[i] ?? 0;
      const v = targets[name] ?? q;
      return /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "sim-joint", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "label", children: name }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: "range",
            min: -Math.PI,
            max: Math.PI,
            step: 5e-3,
            value: v,
            disabled: mirror || !live,
            "aria-label": `${name} target, radians`,
            onChange: (e) => setJoint(name, Number(e.target.value))
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "val mono", children: q.toFixed(3) })
      ] }, name);
    }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sim-foot", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "mono", children: snap.bus ? `${snap.bus.hz ?? 0} Hz bus` : `t=${snap.sim_time.toFixed(2)}s` }),
      snap.fps ? /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "mono", children: [
        snap.fps,
        " fps"
      ] }) : null,
      busLine && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", title: busLine, children: busLine }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "spacer" }),
      !mirror && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", className: "btn ghost", onClick: reset, disabled: !live, children: "reset" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", className: "btn ghost danger", onClick: stop, children: "stop" })
    ] })
  ] });
}
function diskNoticeView(notice, opts = {}) {
  if (!notice) return null;
  const level = notice.level === "critical" ? "critical" : notice.level === "tight" ? "tight" : null;
  if (!level) return null;
  const headline = (notice.headline ?? "").trim();
  if (!headline) return null;
  const backendAdvice = (notice.advice ?? "").trim();
  const recording = !!opts.recording;
  if (level === "critical" && recording) {
    return {
      tone: "bad",
      urgent: true,
      headline,
      // Deliberately NOT the backend's "free space first": that is unreachable advice for someone
      // holding an arm over a live dataset.
      advice: "Stop after this episode. The episodes already written are complete and safe — it is the one that runs out mid-write that leaves a dataset whose meta promises more than its data holds. Close the session, free space, then open a new one and keep going.",
      testid: "disk-critical-recording"
    };
  }
  if (level === "critical") {
    return { tone: "bad", urgent: true, headline, advice: backendAdvice, testid: "disk-critical" };
  }
  return {
    tone: "warn",
    // A tight disk is a fact to notice, not an emergency, and this document is polled about once a
    // second — role="alert" on it would interrupt a screen reader repeatedly for unchanged news.
    urgent: false,
    headline,
    advice: backendAdvice,
    testid: recording ? "disk-tight-recording" : "disk-tight"
  };
}
const REAL_HINT = "Both arms leave the fleet while recording: their peers are despawned and the ports handed to the recorder, and the follower is energised to hold position. Nothing is written until you start an episode.";
const MOCK_HINT = "Rehearsal: this backend has no /api/record, so no arm is touched, no port is taken and no dataset is written. The buttons work so you can learn the flow.";
function openActionCopy(mock) {
  if (mock === true) {
    return {
      label: "open a rehearsal session",
      hint: MOCK_HINT,
      cls: "rehearsal",
      aria: "open a rehearsal session — no arm is touched and nothing is written"
    };
  }
  if (mock === false) {
    return {
      label: "open the arms for recording",
      hint: REAL_HINT,
      cls: "",
      aria: "open the arms for recording — despawns both peers and energises the follower"
    };
  }
  return {
    label: "open the arms for recording",
    hint: REAL_HINT,
    cls: "",
    aria: "open the arms for recording — despawns both peers and energises the follower"
  };
}
const STALE_AFTER_MS = 3200;
function sessionFreshness(input) {
  const { lastOkAtMs, nowMs, recording } = input;
  const staleAfter = input.staleAfterMs ?? STALE_AFTER_MS;
  const why2 = String(input.lastError ?? "").trim();
  if (lastOkAtMs == null || !Number.isFinite(lastOkAtMs)) {
    return { stale: false, ageS: 0, text: null, tone: "warn" };
  }
  const ageMs = Math.max(0, nowMs - lastOkAtMs);
  const ageS2 = Math.floor(ageMs / 1e3);
  if (ageMs < staleAfter) return { stale: false, ageS: ageS2, text: null, tone: "warn" };
  const reason2 = why2 ? ` (${why2})` : "";
  const text = recording ? `⚠ the frame counter is NOT updating — last session read ${ageS2}s ago${reason2}. It may still be recording, or the episode may already have ended: this page cannot tell. The counts below are that old, not live.` : `⚠ session state is ${ageS2}s old — polling is not landing${reason2}. Buttons still work, but what you see may already have changed.`;
  return { stale: true, ageS: ageS2, text, tone: recording ? "bad" : "warn" };
}
function staleSuffix(f) {
  return f.stale ? ` · ${f.ageS}s old` : "";
}
const INERT = {
  open: "no session was opened and nothing was recorded — and unless the message above says otherwise, the arms are back in the fleet.",
  start: "no episode was started — nothing is being recorded.",
  stop: "the episode was NOT stopped — if one was recording, it still is.",
  redo: "nothing was thrown away — the take is still there.",
  discard: "nothing was discarded — that episode is still there.",
  close: "the dataset was NOT finished — the session is still open, and nothing was uploaded."
};
const MAYBE = {
  open: "the session MAY be open: both arms despawned, ports handed to the recorder, follower energised and stiff.",
  start: "the episode MAY already be recording — do the demonstration only once you know, and do not walk away assuming it is idle.",
  stop: "the take MAY already be saved, or MAY still be recording — this panel cannot tell which.",
  redo: "the take MAY already have been thrown away, and that cannot be undone.",
  discard: "that episode MAY already have been discarded, and that cannot be undone.",
  close: "the dataset MAY already be finished — and if you ticked upload, MAY already be on the Hub."
};
const WATCH = "Re-reading the session now — the episode list and the recording pill are the truth, they refresh every second.";
function recordFailure(input) {
  const why2 = String(input.message ?? "").trim() || "no detail";
  if (refusedBeforeActing(input.status)) {
    return {
      text: `✗ refused (${input.status}): ${why2} — ${INERT[input.kind]}`,
      ambiguous: false,
      destructive: false
    };
  }
  const head = Number(input.status ?? 0) ? `⚠ unknown — the server failed mid-request (${input.status}: ${why2})` : `⚠ unknown — no answer came back (${why2})`;
  return {
    text: `${head}: ${MAYBE[input.kind]} ${WATCH}`,
    ambiguous: true,
    destructive: input.kind === "redo" || input.kind === "discard"
  };
}
const NAMED_LEADER = /(^|[^a-z])leader([^a-z]|$)/i;
const NAMED_FOLLOWER = /(^|[^a-z])follower([^a-z]|$)/i;
function measured(candidates, role) {
  return candidates.filter((c) => c.role === role);
}
function pairArms(candidates) {
  var _a, _b;
  const leaders = measured(candidates, "leader");
  const followers = measured(candidates, "follower");
  if (leaders.length === 1 && followers.length === 1) {
    return { leader: leaders[0].peer_id, follower: followers[0].peer_id, basis: "measured" };
  }
  if (leaders.length === 1 && followers.length !== 1) {
    return {
      leader: leaders[0].peer_id,
      follower: "",
      basis: "measured",
      note: followers.length === 0 ? `${leaders[0].peer_id} measured as the leader; no arm has measured as a 12V follower yet — pick the follower yourself, or measure it on the devices screen` : `${leaders[0].peer_id} measured as the leader, but ${followers.length} arms measured as followers — which one it drives is your call, so check the volts next to each name`
    };
  }
  if (followers.length === 1 && leaders.length !== 1) {
    return {
      leader: "",
      follower: followers[0].peer_id,
      basis: "measured",
      note: leaders.length === 0 ? `${followers[0].peer_id} measured as the follower; no arm has measured as a 7.4V leader yet (an unpowered arm reads 5.5V on the USB rail) — pick the leader yourself, or measure it on the devices screen` : `${followers[0].peer_id} measured as the follower, but ${leaders.length} arms measured as leaders — which one drives is your call, so check the volts next to each name`
    };
  }
  if (leaders.length > 1 || followers.length > 1) {
    const parts = [];
    if (leaders.length > 1) parts.push(`${leaders.length} arms measured as the leader`);
    if (followers.length > 1) {
      parts.push(parts.length ? `${followers.length} as the follower` : `${followers.length} arms measured as the follower`);
    }
    return {
      leader: "",
      follower: "",
      basis: "none",
      note: `${parts.join(" and ")}, so which one drives is your call — check the volts next to each name`
    };
  }
  const named = {
    leader: ((_a = candidates.find((c) => NAMED_LEADER.test(c.peer_id))) == null ? void 0 : _a.peer_id) ?? "",
    follower: ((_b = candidates.find((c) => NAMED_FOLLOWER.test(c.peer_id))) == null ? void 0 : _b.peer_id) ?? ""
  };
  if (named.leader && named.follower && named.leader !== named.follower) {
    return {
      ...named,
      basis: "named",
      note: "paired from the peer names — nobody has measured these buses, so the names are being taken at their word"
    };
  }
  if (named.leader !== named.follower && (named.leader || named.follower)) {
    const stated = named.leader ? "leader" : "follower";
    return {
      leader: named.leader,
      follower: named.follower,
      basis: "named",
      note: `only ${named.leader || named.follower} states a role in its name, so it fills the ${stated} slot — nobody has measured these buses, and the other arm is left to you (the leader is the lighter 7.4V arm)`
    };
  }
  return {
    leader: "",
    follower: "",
    basis: "none",
    note: candidates.length ? "no arm has been measured, and the names do not say which is which — the leader is the lighter 7.4V arm, or measure both on the devices screen" : "no arms on the mesh"
  };
}
function roleLabel(c) {
  if (!c.role) return `${c.peer_id} — role not measured`;
  const v = typeof c.role_volts === "number" ? ` · ${c.role_volts}V` : "";
  return `${c.peer_id} — ${c.role}${v}`;
}
function contradiction(candidates, slot, chosen) {
  if (!chosen) return null;
  const c = candidates.find((x) => x.peer_id === chosen);
  if (!(c == null ? void 0 : c.role)) return null;
  if (c.role === slot) return null;
  const v = typeof c.role_volts === "number" ? `${c.role_volts}V` : "its bus";
  if (c.role === "unpowered") {
    return `${chosen} read ${v} — that is the USB logic rail, so its power supply is off. It can report positions but it cannot hold or mirror anything.`;
  }
  if (c.role === "mixed") {
    return `${chosen} reported inconsistent voltages across its servos — that is a wiring fault, not a role. Fix it before recording an episode with it.`;
  }
  return `${chosen} measured ${v} — it is the ${c.role}, not the ${slot}. ` + (slot === "leader" ? "You would be hand-moving a torqued 12V arm while the 7.4V one tries to mirror it." : "The arm being recorded should be the 12V one that mirrors your hand.");
}
const MAX_AGE_S = 30;
function jointCount(peer) {
  var _a;
  const joints = (_a = peer == null ? void 0 : peer.state) == null ? void 0 : _a.joints;
  if ((peer == null ? void 0 : peer.state) == null) return null;
  if (joints == null) return 0;
  if (Array.isArray(joints)) return joints.length;
  if (typeof joints === "object") return Object.keys(joints).length;
  return null;
}
function ageS(peer, nowS) {
  const seen = peer == null ? void 0 : peer.last_seen;
  if (typeof seen !== "number" || !Number.isFinite(seen) || seen <= 0) return null;
  return Math.max(0, nowS - seen);
}
function armJointWarning(peer, { slot, nowS }) {
  const count = jointCount(peer);
  if (count === null || count > 0) return null;
  const age = ageS(peer, nowS);
  if (age === null || age > MAX_AGE_S) return null;
  const note = jointAbsence({
    state: peer == null ? void 0 : peer.state,
    presence: peer == null ? void 0 : peer.presence,
    problem: peer == null ? void 0 : peer.joint_problem,
    nowS
  });
  const why2 = [note.text, note.hint].filter(Boolean).join(" — ");
  return `${slot} ${(peer == null ? void 0 : peer.peer_id) ?? ""} reports no joint positions, so the episodes would carry no ${slot === "leader" ? "actions" : "observations"} to learn from` + (why2 ? `: ${why2}` : "") + ". The recording will be refused until this arm reads.";
}
const EPISODE_MAX = 500;
function episodeTarget(raw) {
  const text = (raw ?? "").trim();
  if (!text) return { value: 0, problem: "how many episodes? enter a number", note: null };
  const n = Number(text);
  if (!Number.isFinite(n)) return { value: 0, problem: `“${text}” is not a number`, note: null };
  if (n <= 0) {
    return { value: 0, problem: n === 0 ? "zero episodes would record nothing" : "that is a negative number of episodes", note: null };
  }
  if (n > EPISODE_MAX) {
    return { value: 0, problem: `${n} episodes is more than this screen will start (max ${EPISODE_MAX}) — record in batches`, note: null };
  }
  if (!Number.isInteger(n)) {
    const floored = Math.floor(n);
    return { value: floored, problem: null, note: `recording ${floored} episodes — ${text} is not a whole number` };
  }
  return { value: n, problem: null, note: null };
}
const DEFAULT_FPS = 30;
const MIN_FPS = 1;
const MAX_FPS = 60;
function fpsField(raw) {
  const text = (raw ?? "").trim();
  if (!text) return { value: DEFAULT_FPS, problem: null, note: null };
  const n = Number(text);
  if (!Number.isFinite(n)) {
    return { value: DEFAULT_FPS, problem: `"${text}" is not a number of frames per second`, note: null };
  }
  if (n < MIN_FPS || n > MAX_FPS) {
    return {
      value: DEFAULT_FPS,
      problem: `fps must be between ${MIN_FPS} and ${MAX_FPS} — ${n} is outside what a dataset can declare`,
      note: null
    };
  }
  const rounded = Math.round(n);
  return {
    value: rounded,
    problem: null,
    note: rounded !== n ? `recording at ${rounded} fps (a dataset's rate is a whole number)` : null
  };
}
function fpsSuggestion(notice) {
  if (!notice || !Number.isFinite(notice.measured_fps)) return null;
  const measured2 = Math.round(notice.measured_fps);
  if (measured2 < MIN_FPS || measured2 > MAX_FPS) return null;
  if (measured2 === Math.round(notice.declared_fps)) return null;
  return {
    fps: String(measured2),
    label: `use ${measured2} fps next session`,
    // Deliberately future-tense: the current session's episodes keep their declaration, and
    // pretending otherwise would be the third lie in this chain.
    why: notice.slower ? `this session is already stamped ${notice.declared_fps} fps and cannot be re-declared; recording the next one at ${measured2} makes its timestamps match real time` : `capture is running faster than declared; ${measured2} fps would describe it honestly`
  };
}
function AuthedImg({ path, alt, className }) {
  const [url, setUrl] = reactExports.useState("");
  const [failed, setFailed] = reactExports.useState(false);
  reactExports.useEffect(() => {
    let alive = true;
    let made = "";
    setFailed(false);
    setUrl("");
    void apiBlob(path).then((u) => {
      if (!alive) {
        URL.revokeObjectURL(u);
        return;
      }
      made = u;
      setUrl(u);
    }).catch(() => {
      if (alive) setFailed(true);
    });
    return () => {
      alive = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [path]);
  if (failed) return /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint small", title: path, children: [
    alt,
    " — unavailable"
  ] });
  if (!url) return /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "thumb-loading", "aria-label": `${alt} loading` });
  return /* @__PURE__ */ jsxRuntimeExports.jsx("img", { src: url, alt, className, loading: "lazy" });
}
function localNames(known) {
  const out = /* @__PURE__ */ new Map();
  for (const d of known) {
    if (d.local === false) continue;
    const id = (d.repo_id ?? "").trim();
    if (id) out.set(id, typeof d.total_episodes === "number" ? d.total_episodes : void 0);
  }
  return out;
}
function freeVariant(name, known) {
  const taken = localNames(known);
  const base = name.trim();
  if (!base) return null;
  const m = base.match(/^(.*?)-(\d+)$/);
  const stem = m ? m[1] : base;
  let n = m ? Number(m[2]) : 1;
  for (let i = 0; i < 200; i += 1) {
    n += 1;
    const candidate = `${stem}-${n}`;
    if (!taken.has(candidate)) return candidate;
  }
  return null;
}
function nameVerdict(name, known) {
  const id = (name ?? "").trim();
  if (!id || !known) return null;
  const taken = localNames(known);
  if (!taken.has(id)) return null;
  const episodes = taken.get(id);
  const what = typeof episodes === "number" && episodes > 0 ? `already exists with ${episodes} episode(s)` : "already exists on disk (no episodes recorded — probably an interrupted session)";
  return {
    message: `“${id}” ${what}. Recording refuses to reuse a dataset directory, so this will be turned away when you press start.`,
    suggestion: freeVariant(id, known)
  };
}
function slugFromTask(task) {
  const words2 = (task ?? "").toLowerCase().replace(/[^a-z0-9\s_-]+/g, " ").split(/[\s_]+/).filter(Boolean);
  if (words2.length === 0) return null;
  return words2.slice(0, 4).join("-");
}
function suggestDatasetName(task, currentValue, known) {
  if ((currentValue ?? "").trim()) return null;
  const slug = slugFromTask(task);
  if (!slug) return null;
  const taken = (known ?? []).some((d) => d.local !== false && (d.repo_id ?? "").trim() === slug);
  if (!taken) return slug;
  return freeVariant(slug, known ?? []);
}
const OVERRIDES = [
  {
    flag: "ignore_dead_cameras",
    label: "I know this camera is stale — record without waiting for it",
    cost: "the episodes may carry a frozen image, or none, for that view"
  },
  {
    flag: "ignore_missing_cameras",
    label: "record without the camera this machine cannot see",
    cost: "that view is simply absent from every episode"
  },
  {
    flag: "ignore_camera_identity",
    label: "my cameras really are at these indices — record with them as they stand",
    cost: "if the numbering did shift, every episode records the WRONG view while looking healthy"
  }
];
function overrideOffered(message) {
  if (typeof message !== "string" || !message) return null;
  const named = OVERRIDES.filter((o) => message.includes(o.flag));
  return named.length === 1 ? named[0] : null;
}
const FLAGS = OVERRIDES.map((o) => o.flag);
function nextAcknowledged(prev, offered, acknowledged) {
  const kept = FLAGS.filter((f) => prev.includes(f));
  if (!offered || !acknowledged || kept.includes(offered.flag)) return kept;
  return FLAGS.filter((f) => kept.includes(f) || f === offered.flag);
}
function overrideBodyFlags(flags) {
  const body = {};
  for (const f of FLAGS) if (flags.includes(f)) body[f] = true;
  return body;
}
function trainHandoff(r) {
  var _a, _b;
  if (!(r == null ? void 0 : r.ok)) return null;
  if (!r.episodes_kept || r.episodes_kept <= 0) return null;
  const where = (r.root ?? "").trim() || (r.dataset ?? "").trim();
  if (!where) return null;
  const caveat = r.camera_notice && (((_a = r.camera_notice.missing) == null ? void 0 : _a.length) ?? 0) > 0 ? (((_b = r.camera_notice.present) == null ? void 0 : _b.length) ?? 0) > 0 ? "some image channels are missing — a visual policy trains on less than you saw" : "no image channel at all — this dataset cannot train a visual policy" : null;
  return {
    prefill: { dataset_root: where },
    label: `train on it (${r.episodes_kept} episode${r.episodes_kept === 1 ? "" : "s"}) →`,
    caveat
  };
}
function RecordPanel({ peers, onClose, onDevices, onTrain }) {
  var _a;
  const peerIds = peers.map((p) => p.peer_id);
  const [api$1, setApi] = reactExports.useState(null);
  const sheetRef = reactExports.useRef(null);
  useDialogFocus(sheetRef);
  const [s, setS] = reactExports.useState(null);
  const [busy, setBusy] = reactExports.useState(false);
  const [err, setErr] = reactExports.useState(null);
  const [roles, setRoles] = reactExports.useState({});
  const hosts = armHosts(peers.map((p) => {
    var _a2;
    return {
      peer_id: p.peer_id,
      joints: ((_a2 = p.state) == null ? void 0 : _a2.joints) ? Array.isArray(p.state.joints) ? p.state.joints.length : typeof p.state.joints === "object" ? Object.keys(p.state.joints).length : null : 0
    };
  }));
  const candidates = peerIds.filter((id) => !hosts[id]).map((id) => roles[id] ?? { peer_id: id });
  const [boards, setBoards] = reactExports.useState(void 0);
  const suggestion = pairArms(candidates);
  const noArms = noArmsVerdict(peerIds.length, boards === void 0 ? null : boards);
  const [form, setForm] = reactExports.useState({
    dataset: "",
    task: "",
    leader: "",
    follower: "",
    target_episodes: "20",
    fps: ""
  });
  const [touched, setTouched] = reactExports.useState({ leader: false, follower: false });
  const wanted = episodeTarget(form.target_episodes);
  const rate = fpsField(form.fps);
  const [ack, setAck] = reactExports.useState(false);
  const problems = ["leader", "follower"].map((slot) => ({ slot, msg: contradiction(candidates, slot, form[slot]) })).filter((x) => !!x.msg);
  const [upload, setUpload] = reactExports.useState(false);
  const [pre, setPre] = reactExports.useState(null);
  const [preErr, setPreErr] = reactExports.useState(false);
  const [uploadForce, setUploadForce] = reactExports.useState(false);
  reactExports.useEffect(() => {
    if (!upload || !api$1) {
      setPre(null);
      setPreErr(false);
      return;
    }
    let alive = true;
    void api$1.uploadPreflight().then((v) => {
      if (alive) {
        setPre(v);
        setPreErr(false);
      }
    }).catch(() => {
      if (alive) {
        setPre(null);
        setPreErr(true);
      }
    });
    return () => {
      alive = false;
    };
  }, [upload, api$1, s == null ? void 0 : s.dataset]);
  const uploadBlocked = !!upload && (!pre || !pre.ok && !(pre.needs_force && uploadForce));
  const armedUpload = upload && !uploadBlocked;
  const [known, setKnown] = reactExports.useState(null);
  reactExports.useEffect(() => {
    let alive = true;
    api("/api/training/datasets?hub=false").then((r) => {
      if (alive) setKnown(r.datasets ?? []);
    }).catch(() => {
      if (alive) setKnown(null);
    });
    return () => {
      alive = false;
    };
  }, []);
  const nameWarn = nameVerdict(form.dataset, known);
  const [closed, setClosed] = reactExports.useState(null);
  const [receipt, setReceipt] = reactExports.useState(null);
  const [lastOkAt, setLastOkAt] = reactExports.useState(null);
  const [pollErr, setPollErr] = reactExports.useState(null);
  const [nowMs, setNowMs] = reactExports.useState(() => Date.now());
  const followerPeer = peers.find((p) => p.peer_id === form.follower);
  const deadCams = stoppedCameras(followerPeer == null ? void 0 : followerPeer.cameras, Date.now() / 1e3);
  const camWarning = cameraWarning(deadCams, { peerId: form.follower });
  const jointWarnings = [
    ["leader", form.leader],
    ["follower", form.follower]
  ].map(([slot, pid]) => ({
    slot,
    msg: armJointWarning(peers.find((p) => p.peer_id === pid), { slot, nowS: Date.now() / 1e3 })
  })).filter((x) => !!x.msg);
  const [camAck, setCamAck] = reactExports.useState(false);
  const [refusalAck, setRefusalAck] = reactExports.useState(false);
  const offered = overrideOffered(err);
  const [ackedFlags, setAckedFlags] = reactExports.useState([]);
  reactExports.useEffect(() => {
    setAckedFlags([]);
  }, [form.follower, form.leader, form.dataset]);
  reactExports.useEffect(() => {
    let alive = true;
    api("/api/devices").then((doc) => {
      if (!alive) return;
      const next = {};
      for (const m of Object.values(doc.managed ?? {})) {
        if (m == null ? void 0 : m.peer_id) next[m.peer_id] = { peer_id: m.peer_id, role: m.role, role_volts: m.role_volts };
      }
      setRoles(next);
      const claimed = new Set(Object.values(doc.managed ?? {}).filter((m) => (m == null ? void 0 : m.alive) && (m == null ? void 0 : m.port)).map((m) => m.port));
      setBoards((doc.serial_ports ?? []).filter((p) => {
        var _a2;
        return (_a2 = p.remembered) == null ? void 0 : _a2.peer_id;
      }).map((p) => ({ peer_id: p.remembered.peer_id, claimed: claimed.has(p.device) })));
    }).catch(() => {
      setBoards(null);
    });
    return () => {
      alive = false;
    };
  }, []);
  reactExports.useEffect(() => {
    setForm((f) => ({
      ...f,
      leader: touched.leader ? f.leader : suggestion.leader,
      follower: touched.follower ? f.follower : suggestion.follower
    }));
  }, [suggestion.leader, suggestion.follower, touched.leader, touched.follower]);
  reactExports.useEffect(() => {
    let alive = true;
    getRecordApi().then((a) => {
      if (!alive) return;
      setApi(a);
      a.session().then((sess) => {
        if (alive) {
          setS(sess);
          setLastOkAt(Date.now());
        }
      }).catch((e) => alive && setErr(String(e)));
    });
    return () => {
      alive = false;
    };
  }, []);
  const sRef = reactExports.useRef(s);
  sRef.current = s;
  reactExports.useEffect(() => {
    if (!api$1 || !(s == null ? void 0 : s.dataset)) return;
    let alive = true;
    let pending = false;
    const t = setInterval(() => {
      setNowMs(Date.now());
      if (pending) return;
      pending = true;
      api$1.session().then((sess) => {
        if (alive) {
          setS(sess);
          setLastOkAt(Date.now());
          setPollErr(null);
        }
      }).catch((e) => {
        if (alive) setPollErr(e instanceof Error ? e.message : String(e));
      }).finally(() => {
        pending = false;
      });
    }, 1e3);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [api$1, s == null ? void 0 : s.dataset]);
  const run = async (fn, kind) => {
    if (busy) return;
    setBusy(true);
    setErr(null);
    setRefusalAck(false);
    try {
      setS(await fn());
      setLastOkAt(Date.now());
      setPollErr(null);
    } catch (e) {
      const v = recordFailure({
        kind,
        status: e instanceof HttpError ? e.status : 0,
        message: e instanceof Error ? e.message : String(e)
      });
      setErr(v.text);
      if (v.ambiguous && api$1) {
        try {
          setS(await api$1.session());
          setLastOkAt(Date.now());
          setPollErr(null);
        } catch {
        }
      }
    }
    setBusy(false);
  };
  const runRef = reactExports.useRef(run);
  runRef.current = run;
  const apiRef = reactExports.useRef(api$1);
  apiRef.current = api$1;
  reactExports.useEffect(() => {
    const onKey = (e) => {
      const cur = sRef.current;
      const a = apiRef.current;
      if (!a || !(cur == null ? void 0 : cur.dataset)) return;
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)) return;
      if (e.code === "Space") {
        e.preventDefault();
        const recording2 = cur.phase === "recording";
        void runRef.current(() => recording2 ? a.stopEpisode() : a.startEpisode(), recording2 ? "stop" : "start");
      } else if (e.key === "x" || e.key === "X") {
        if (cur.phase !== "recording") return;
        e.preventDefault();
        void runRef.current(() => a.redoEpisode(), "redo");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const open = !!(s == null ? void 0 : s.dataset);
  const recording = (s == null ? void 0 : s.phase) === "recording";
  const episodes = Array.isArray(s == null ? void 0 : s.episodes) ? s.episodes : [];
  const finished = recording ? episodes.slice(0, -1) : episodes;
  const kept = finished.filter((e) => !(e == null ? void 0 : e.discarded)).length;
  const liveFrames = recording && episodes.length > 0 ? ((_a = episodes[episodes.length - 1]) == null ? void 0 : _a.frames) ?? null : null;
  const fresh = sessionFreshness({ lastOkAtMs: lastOkAt, nowMs, lastError: pollErr, recording });
  const openCopy = openActionCopy(api$1 ? api$1.mock : null);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { ref: sheetRef, className: "train-sheet", role: "dialog", "aria-label": "Record episodes", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "⏺ Record" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "dock-min", onClick: onClose, "aria-label": "close", children: "✕" })
    ] }),
    (api$1 == null ? void 0 : api$1.mock) && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "toast warn", children: "This backend has no /api/record (older server?) — this is a rehearsal. Nothing is written to disk." }),
    !open && (s == null ? void 0 : s.interrupted) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "artifact-hold", role: "status", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        "⏹ ",
        s.interrupted.text
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "hold-next", children: s.interrupted.next.map((n) => /* @__PURE__ */ jsxRuntimeExports.jsx("li", { children: n }, n)) })
    ] }),
    !open && s && /* @__PURE__ */ jsxRuntimeExports.jsxs("form", { className: "train-form", onSubmit: (e) => {
      e.preventDefault();
      setAckedFlags(nextAcknowledged(ackedFlags, offered, refusalAck));
      void run(() => api$1.open({
        dataset: form.dataset.trim(),
        task: form.task.trim(),
        leader: form.leader,
        follower: form.follower,
        target_episodes: wanted.value,
        fps: rate.value,
        // Only ever sent when the operator ticked the box in front of the
        // named camera and its age - never a default, never remembered.
        ...camWarning && camAck ? { ignore_dead_cameras: true } : {},
        ...overrideBodyFlags(nextAcknowledged(ackedFlags, offered, refusalAck))
      }), "open");
    }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "dataset (name or hf repo id)" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            value: form.dataset,
            placeholder: "cagatay/so101-pick-cube",
            onChange: (e) => set("dataset", e.target.value)
          }
        )
      ] }),
      nameWarn && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg", role: "status", children: [
        "⚠ ",
        nameWarn.message,
        nameWarn.suggestion && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          " ",
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "button",
            {
              type: "button",
              className: "btn ghost",
              onClick: () => set("dataset", nameWarn.suggestion),
              children: [
                "use ",
                nameWarn.suggestion
              ]
            }
          )
        ] })
      ] }),
      (() => {
        const sug = suggestDatasetName(form.task, form.dataset, known);
        return sug ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg rec-hint", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
          "button",
          {
            type: "button",
            className: "btn ghost suggest",
            onClick: () => set("dataset", sug),
            children: [
              "name it ",
              sug
            ]
          }
        ) }) : null;
      })(),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "task — what the arm is being taught" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            value: form.task,
            placeholder: "pick up the red cube and place it in the bin",
            onChange: (e) => set("task", e.target.value)
          }
        )
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-row", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "leader — you move this one" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              value: form.leader,
              onChange: (e) => {
                setTouched((t) => ({ ...t, leader: true }));
                set("leader", e.target.value);
              },
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "select…" }),
                candidates.map((c) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: c.peer_id, children: roleLabel(c) }, c.peer_id)),
                Object.entries(hosts).map(([id, h]) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: id, disabled: true, children: [
                  id,
                  " — ",
                  h.why
                ] }, id))
              ]
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "follower — gets recorded" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              value: form.follower,
              onChange: (e) => {
                setTouched((t) => ({ ...t, follower: true }));
                set("follower", e.target.value);
              },
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "select…" }),
                candidates.map((c) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: c.peer_id, children: roleLabel(c) }, c.peer_id)),
                Object.entries(hosts).map(([id, h]) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: id, disabled: true, children: [
                  id,
                  " — ",
                  h.why
                ] }, id))
              ]
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "fps" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              inputMode: "numeric",
              value: form.fps,
              placeholder: "30",
              "aria-invalid": !!rate.problem,
              "aria-describedby": "rec-fps-say",
              onChange: (e) => set("fps", e.target.value)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "rec-fps-say", className: `fieldsay${rate.problem ? " bad" : ""}`, children: rate.problem ?? rate.note ?? "timestamps are derived from this — match your real capture rate" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "episodes" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              inputMode: "numeric",
              value: form.target_episodes,
              "aria-invalid": !!wanted.problem,
              "aria-describedby": "rec-episodes-say",
              onChange: (e) => set("target_episodes", e.target.value)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { id: "rec-episodes-say", className: `fieldsay${wanted.problem ? " bad" : ""}`, children: wanted.problem ?? wanted.note ?? "" })
        ] })
      ] }),
      suggestion.basis === "measured" && !suggestion.note && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg rec-hint", children: "paired from the servo buses — measured, not guessed from the names." }),
      noArms ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg rec-hint", role: "status", children: [
        noArms.text,
        noArms.offerDevices && onDevices && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          " ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", className: "btn ghost", onClick: onDevices, children: "open the devices screen" })
        ] })
      ] }) : suggestion.note && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg rec-hint", children: suggestion.note }),
      problems.map(({ slot, msg }) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn", children: [
        "⚠ ",
        msg
      ] }, slot)),
      problems.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "ackrow", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("input", { type: "checkbox", checked: ack, onChange: (e) => setAck(e.target.checked) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          "my arms really are wired that way — record anyway",
          problems.some((x) => x.slot === "leader") ? " (hand-moving a torqued arm can strip a gear: cut its power first)" : ""
        ] })
      ] }),
      jointWarnings.map(({ slot, msg }) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn", role: "alert", children: [
        "⚠ ",
        msg
      ] }, `joints-${slot}`)),
      camWarning && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn", children: [
        "⚠ ",
        camWarning
      ] }),
      camWarning && /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "ackrow", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("input", { type: "checkbox", checked: camAck, onChange: (e) => setCamAck(e.target.checked) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          "record without ",
          deadCams.length > 1 ? "those cameras" : `the ${deadCams[0].camera} camera`,
          " anyway"
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "rec-hint", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { children: "not sure which arm is which?" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg rec-hint", children: "the leader is the lighter 7.4V arm (no gearbox load — easy to move by hand); the follower is the stronger 12V arm that mirrors it." })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: `train-msg rec-hint${openCopy.cls ? " warn" : ""}`, children: openCopy.hint }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: `btn go wide${openCopy.cls ? ` ${openCopy.cls}` : ""}`,
          type: "submit",
          "aria-label": openCopy.aria,
          disabled: busy || !api$1 || !form.dataset.trim() || !form.task.trim() || !form.leader || !form.follower || form.leader === form.follower || !!wanted.problem || !!rate.problem || problems.length > 0 && !ack || !!camWarning && !camAck || jointWarnings.length > 0,
          children: openCopy.label
        }
      ) }),
      !!form.leader && form.leader === form.follower && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg", children: "⚠ leader and follower must be different arms" })
    ] }),
    (() => {
      const d = diskNoticeView(s == null ? void 0 : s.disk_notice, { recording: open });
      if (!d) return null;
      return /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "div",
        {
          className: `train-msg ${d.tone}`,
          "data-testid": d.testid,
          ...d.urgent ? { role: "alert" } : { "aria-live": "polite" },
          children: [
            "⚠ ",
            d.headline,
            d.advice && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "fieldsay", children: d.advice })
          ]
        }
      );
    })(),
    open && (s == null ? void 0 : s.camera_notice) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn rec-camera-notice", role: "alert", children: [
      "⚠ ",
      s.camera_notice.message
    ] }),
    open && (s == null ? void 0 : s.fps_notice) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn rec-fps-notice", role: "alert", children: [
      "⚠ ",
      s.fps_notice.detail,
      (() => {
        const sug = fpsSuggestion(s.fps_notice);
        if (!sug) return null;
        return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "rec-fps-fix", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn ghost",
              type: "button",
              onClick: () => set("fps", sug.fps),
              children: sug.label
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "fieldsay", children: sug.why })
        ] });
      })()
    ] }),
    open && (s == null ? void 0 : s.motion_notice) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg warn rec-motion-notice", role: "alert", children: [
      "⚠ ",
      s.motion_notice.message
    ] }),
    open && s && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-form", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "rec-counter", "aria-live": "polite", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: kept }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          " / ",
          s.target_episodes,
          " episodes"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "rec-task", children: [
          s.dataset,
          " — “",
          s.task,
          "”"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "rec-rate", title: "declared fps vs the rate actually captured", children: [
          s.fps,
          " fps",
          s.fps_achieved != null && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: s.fps_notice ? "rec-rate-bad" : "rec-rate-ok", children: [
            " ",
            "· ",
            s.fps_achieved,
            " captured"
          ] }),
          s.motion_notice && /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "rec-rate-bad", title: s.motion_notice.message, children: [
            " ",
            "· not moving"
          ] })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-actions", children: !recording ? /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn go wide", onClick: () => void run(() => api$1.startEpisode(), "start"), disabled: busy, children: [
        "⏺ start episode ",
        kept + 1
      ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn wide", onClick: () => void run(() => api$1.redoEpisode(), "redo"), disabled: busy, children: "↺ redo" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn go wide rec-live", onClick: () => void run(() => api$1.stopEpisode(), "stop"), disabled: busy, children: [
          "⏹ stop & keep",
          liveFrames !== null ? ` · ${liveFrames}f${staleSuffix(fresh)}` : ""
        ] })
      ] }) }),
      fresh.text && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: `toast ${fresh.tone === "bad" ? "bad" : "warn"}`, children: fresh.text }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-msg", children: recording ? `recording — drive ${s.follower} with ${s.leader}, then stop (or redo to throw this one away)` : "start when both arms are in position" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg rec-keys", "aria-hidden": "true", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "space" }),
        " ",
        recording ? "stop & keep" : "start",
        " · ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("kbd", { children: "X" }),
        " redo"
      ] })
    ] }),
    open && s && /* @__PURE__ */ jsxRuntimeExports.jsx(FollowerLive, { peer: peers.find((p) => p.peer_id === s.follower), recording }),
    open && s && episodes.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "rec-strip", role: "list", "aria-label": "recorded episodes", children: finished.slice().reverse().map((ep) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { role: "listitem", className: `rec-ep${ep.discarded ? " dead" : ""}`, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "rec-ep-head", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("b", { children: [
          "ep ",
          ep.index
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          ep.frames,
          "f · ",
          ep.duration_s,
          "s"
        ] })
      ] }),
      Object.entries(ep.thumbnails).length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "rec-thumbs", children: Object.entries(ep.thumbnails).slice(0, 3).map(([cam, url]) => /* @__PURE__ */ jsxRuntimeExports.jsx(AuthedImg, { path: url, alt: `${cam} thumbnail of episode ${ep.index}` }, cam)) }),
      ep.discarded ? /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "rec-ep-gone", children: "discarded" }) : /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => void run(() => api$1.discard(ep.index), "discard"), disabled: busy, children: "✕ discard" })
    ] }, ep.index)) }),
    open && s && !recording && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-form", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field check", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("input", { type: "checkbox", checked: upload, onChange: (e) => setUpload(e.target.checked) }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "upload to the Hugging Face Hub after finishing" })
      ] }),
      upload && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: pre && !pre.ok ? "hint bad" : "hint", children: [
          "publishes as ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: (pre == null ? void 0 : pre.destination) ?? s.dataset ?? "(unnamed)" }),
          " — a dataset can only be pushed under the name it was recorded with. It will be ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "public" }),
          " unless your Hub namespace defaults to private, and if the push fails the episodes stay on this machine: finishing closes the session, so a retry is a ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "huggingface-cli" }),
          " job."
        ] }),
        !pre && !preErr && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "checking whether this machine can publish…" }),
        preErr && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint bad", children: [
          "could not check whether this machine can publish — leaving the upload OFF rather than finding out after the session. Finish without it and push with",
          " ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "huggingface-cli upload" }),
          ", or retry by unticking and ticking again."
        ] }),
        pre && !pre.ok && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint bad", role: "alert", children: [
          "⚠ ",
          pre.detail
        ] }),
        pre && !pre.ok && pre.needs_force && /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "field check", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "checkbox",
              checked: uploadForce,
              onChange: (e) => setUploadForce(e.target.checked)
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: pre.state === "destination_exists" ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            "Replace the dataset already published at ",
            /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: pre.destination }),
            " with this session"
          ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            "I can write to ",
            /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: pre.destination }),
            " — publish there anyway"
          ] }) })
        ] }),
        (pre == null ? void 0 : pre.ok) && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "✓ logged in as ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: pre.user })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "train-actions", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("button", { className: "btn wide", disabled: busy, onClick: () => {
        void (async () => {
          setBusy(true);
          setErr(null);
          try {
            const r = await api$1.close(armedUpload ? { upload: true } : {});
            setClosed(r.detail ?? (r.ok ? `dataset finished with ${kept} episode(s)` : "close failed"));
            setReceipt(r);
            setS(await api$1.session());
          } catch (e) {
            const v = recordFailure({
              kind: "close",
              status: e instanceof HttpError ? e.status : 0,
              message: e instanceof Error ? e.message : String(e)
            });
            setErr(v.text);
            if (v.ambiguous) {
              try {
                setS(await api$1.session());
                setLastOkAt(Date.now());
                setPollErr(null);
              } catch {
              }
            }
          }
          setBusy(false);
        })();
      }, children: [
        "✓ ",
        uploadBlocked ? "finish WITHOUT uploading" : armedUpload ? "finish + publish" : "finish dataset",
        " ",
        "(",
        kept,
        " kept",
        episodes.length - kept ? `, ${episodes.length - kept} discarded` : "",
        ")"
      ] }) })
    ] }),
    closed && !open && (() => {
      const h = trainHandoff(receipt);
      return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "toast", role: "status", children: [
        "✓ ",
        closed,
        h && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          (receipt == null ? void 0 : receipt.root) && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "hint mono small", children: receipt.root }),
          h.caveat && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "hint warn", children: [
            "⚠ ",
            h.caveat
          ] }),
          onTrain && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row", children: /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn", onClick: () => onTrain(h.prefill), children: h.label }) })
        ] })
      ] });
    })(),
    err && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg", role: "alert", children: [
      "✗ ",
      err
    ] }),
    err && offered && /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "ackrow", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "input",
        {
          type: "checkbox",
          checked: refusalAck,
          onChange: (e) => setRefusalAck(e.target.checked)
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        offered.label,
        " ",
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "hint", children: [
          "— ",
          offered.cost
        ] })
      ] })
    ] })
  ] });
}
function FollowerLive({ peer, recording }) {
  var _a;
  if (!peer) {
    return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "toast warn", children: "The follower is not on the mesh — its card left the fleet. Recording would capture nothing." });
  }
  const evidence = cameraEvidence(
    peer.peer_id,
    (_a = peer.presence) == null ? void 0 : _a.cameras,
    Object.keys(peer.cameras ?? {}),
    peer.cameras_requested
  );
  const cams = evidence.kind === "ok" ? evidence.cams : [];
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `train-form rec-live-view${recording ? " armed" : ""}`, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "rec-live-head", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        peer.peer_id,
        " — what the dataset sees"
      ] }),
      recording && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "rec-dot", "aria-hidden": "true" })
    ] }),
    cams.length > 0 ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: cams.length > 1 ? "cams multi" : "cams", children: cams.slice(0, 4).map((c) => {
      var _a2;
      return /* @__PURE__ */ jsxRuntimeExports.jsx(CameraTile, { peerId: peer.peer_id, cam: c, meta: (_a2 = peer.cameras) == null ? void 0 : _a2[c] }, c);
    }) }) : /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "train-msg", title: evidence.kind === "mute" ? "presence lists cameras; no frames have arrived" : void 0, children: [
      "⚠ ",
      evidence.kind === "ok" ? "" : evidence.message
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(JointStrip, { state: peer.state, presence: peer.presence, problem: peer.joint_problem, peerStale: peer.stale })
  ] });
}
function b64uToBuf(s) {
  const norm2 = s.replace(/-/g, "+").replace(/_/g, "/");
  const pad = norm2.length % 4 ? "=".repeat(4 - norm2.length % 4) : "";
  const bin = atob(norm2 + pad);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}
function bufToB64u(buf) {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function prepCreate(opts) {
  const out = { ...opts };
  out.challenge = b64uToBuf(opts.challenge);
  out.user = { ...opts.user, id: b64uToBuf(opts.user.id) };
  if (opts.excludeCredentials) {
    out.excludeCredentials = opts.excludeCredentials.map((c) => ({ ...c, id: b64uToBuf(c.id) }));
  }
  return out;
}
function prepGet(opts) {
  const out = { ...opts };
  out.challenge = b64uToBuf(opts.challenge);
  delete out.allowCredentials;
  return out;
}
function credToJSON(cred) {
  const r = cred.response;
  const out = {
    id: cred.id,
    rawId: bufToB64u(cred.rawId),
    type: cred.type,
    clientExtensionResults: cred.getClientExtensionResults ? cred.getClientExtensionResults() : {},
    response: {}
  };
  if (r.attestationObject !== void 0) {
    out.response.attestationObject = bufToB64u(r.attestationObject);
    out.response.clientDataJSON = bufToB64u(r.clientDataJSON);
  } else {
    out.response.authenticatorData = bufToB64u(r.authenticatorData);
    out.response.clientDataJSON = bufToB64u(r.clientDataJSON);
    out.response.signature = bufToB64u(r.signature);
    out.response.userHandle = r.userHandle ? bufToB64u(r.userHandle) : null;
  }
  return out;
}
function webauthnReady() {
  return typeof window !== "undefined" && window.isSecureContext === true && typeof navigator !== "undefined" && !!navigator.credentials && typeof navigator.credentials.create === "function" && typeof window.PublicKeyCredential !== "undefined";
}
function fetchAuthStatus() {
  return api("/api/auth/status");
}
async function enroll(label2, bootstrap = "") {
  const { challenge_id, options } = await api("/api/auth/register/begin", {
    method: "POST",
    body: JSON.stringify({ label: label2, bootstrap })
  });
  const cred = await navigator.credentials.create({ publicKey: prepCreate(options) });
  if (!cred) throw new Error("passkey creation was cancelled");
  const res = await api("/api/auth/register/finish", {
    method: "POST",
    body: JSON.stringify({ challenge_id, credential: credToJSON(cred) })
  });
  return res.token;
}
function loginFresh(p) {
  return !!p && Date.now() - p.t < 24e4;
}
async function beginLogin() {
  const { challenge_id, options } = await api("/api/auth/login/begin", {
    method: "POST",
    body: JSON.stringify({})
  });
  delete options.allowCredentials;
  return { challenge_id, options, t: Date.now() };
}
async function completeLogin(p, timeoutMs = 75e3) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  let cred;
  try {
    cred = await navigator.credentials.get({ publicKey: prepGet(p.options), signal: ac.signal });
  } catch (e) {
    if (ac.signal.aborted) throw new Error("the authenticator did not answer in time — tap sign in to try again");
    throw e;
  } finally {
    clearTimeout(timer);
  }
  if (!cred) throw new Error("passkey sign-in was cancelled");
  const res = await api("/api/auth/login/finish", {
    method: "POST",
    body: JSON.stringify({ challenge_id: p.challenge_id, credential: credToJSON(cred) })
  });
  return res.token;
}
const __vite_import_meta_env__ = {};
const BUILD = (__vite_import_meta_env__ == null ? void 0 : __vite_import_meta_env__.VITE_BUILD) ?? "dev";
function AuthGate({ children }) {
  const [mode, setMode] = reactExports.useState("checking");
  const [status, setStatus] = reactExports.useState(null);
  const [error, setError] = reactExports.useState("");
  const [busy, setBusy] = reactExports.useState(false);
  const [label2, setLabel] = reactExports.useState("");
  const [bootstrap, setBootstrap] = reactExports.useState("");
  const [showToken, setShowToken] = reactExports.useState(false);
  const [expiring, setExpiring] = reactExports.useState("");
  const [tokenValue, setTokenValue] = reactExports.useState("");
  const prepared = reactExports.useRef(null);
  const verifying = reactExports.useRef(false);
  reactExports.useEffect(() => {
    if (mode !== "open") {
      setExpiring("");
      return;
    }
    let alive = true;
    const check = () => {
      if (!alive) return;
      const v = sessionVerdict(authToken(), Date.now() / 1e3, lastRenewalAt());
      if (v.refusesUntilSignIn) {
        setExpiring("");
        setError(v.text ?? "this sign-in has expired");
        fetchAuthStatus().then((st) => {
          if (alive) {
            setStatus(st);
            setMode(st.setup_required ? "enroll" : "login");
          }
        }).catch(() => {
          if (alive) setMode("login");
        });
        return;
      }
      setExpiring(v.state === "expiring" ? v.text ?? "" : "");
      if (authRefusedRecently() && !verifying.current) {
        verifying.current = true;
        api("/api/fleet").then(() => {
          verifying.current = false;
        }).catch((e) => {
          verifying.current = false;
          const denied = e instanceof HttpError && (e.status === 401 || e.status === 403);
          if (!alive || !denied) return;
          setExpiring("");
          setError("this page is not signed in any more — the server refused it");
          fetchAuthStatus().then((st) => {
            if (alive) {
              setStatus(st);
              setMode(st.setup_required ? "enroll" : "login");
            }
          }).catch(() => {
            if (alive) setMode("login");
          });
        });
      }
    };
    check();
    const t = setInterval(check, 3e4);
    window.addEventListener("focus", check);
    document.addEventListener("visibilitychange", check);
    return () => {
      alive = false;
      clearInterval(t);
      window.removeEventListener("focus", check);
      document.removeEventListener("visibilitychange", check);
    };
  }, [mode]);
  reactExports.useEffect(() => {
    if (mode !== "login" || !webauthnReady()) return;
    let alive = true;
    const arm = () => {
      if (loginFresh(prepared.current)) return;
      beginLogin().then((p) => {
        if (alive) prepared.current = p;
      }).catch(() => {
      });
    };
    arm();
    const t = setInterval(arm, 2e5);
    window.addEventListener("focus", arm);
    document.addEventListener("visibilitychange", arm);
    return () => {
      alive = false;
      clearInterval(t);
      window.removeEventListener("focus", arm);
      document.removeEventListener("visibilitychange", arm);
    };
  }, [mode]);
  function signIn() {
    const p = loginFresh(prepared.current) ? prepared.current : null;
    prepared.current = null;
    if (p) {
      void run(() => completeLogin(p));
    } else {
      setBusy(true);
      setError("");
      beginLogin().then((np) => {
        prepared.current = np;
        setBusy(false);
        setError("ready — tap sign in again");
      }).catch((e) => {
        setBusy(false);
        setError(String(e.message ?? e));
      });
    }
  }
  reactExports.useEffect(() => {
    let alive = true;
    (async () => {
      var _a;
      try {
        const [st, fleet] = await Promise.allSettled([fetchAuthStatus(), api("/api/fleet")]);
        if (!alive) return;
        if (fleet.status === "fulfilled") {
          setMode("open");
          return;
        }
        const denied = fleet.reason instanceof HttpError && (fleet.reason.status === 401 || fleet.reason.status === 403);
        if (!denied) {
          setMode("unreachable");
          setError(String(((_a = fleet.reason) == null ? void 0 : _a.message) ?? fleet.reason));
          return;
        }
        if (st.status !== "fulfilled") {
          setMode("unreachable");
          setError("auth status unavailable");
          return;
        }
        setStatus(st.value);
        setMode(st.value.setup_required ? "enroll" : "login");
      } catch (e) {
        if (alive) {
          setMode("unreachable");
          setError(String(e.message ?? e));
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);
  async function run(fn) {
    var _a;
    setBusy(true);
    setError("");
    try {
      const token = await fn();
      setAuthToken(token);
      setMode("open");
    } catch (e) {
      const msg = e instanceof HttpError ? ((_a = e.body) == null ? void 0 : _a.detail) ?? e.message : e.message;
      setError(String(msg || "the passkey ceremony failed"));
      setBusy(false);
    }
  }
  if (mode === "open") {
    return /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      expiring && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "sessionwarn", role: "status", "aria-live": "polite", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { "aria-hidden": "true", children: "⏳" }),
        " ",
        expiring
      ] }),
      children
    ] });
  }
  if (mode === "checking") {
    return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "authgate", role: "status", "aria-live": "polite", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "authcard", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(StrandsMark, { size: 40 }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "dim", children: "checking access…" })
    ] }) });
  }
  const noWebauthn = (mode === "enroll" || mode === "login") && !webauthnReady();
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "authgate", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "authcard", role: "dialog", "aria-labelledby": "authgate-title", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx(StrandsMark, { size: 40 }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "authhost", children: [
      "strands robots · ",
      window.location.host
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("h1", { id: "authgate-title", children: mode === "unreachable" ? "backend unreachable" : mode === "enroll" ? "create the admin passkey" : "unlock with your passkey" }),
    mode === "unreachable" && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "dim", children: [
      "The dashboard API did not answer. ",
      error && /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: error })
    ] }),
    noWebauthn && /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "authwarn", children: [
      "Passkeys need a secure context. Open this page over ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "https://" }),
      " or",
      " ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "http://localhost" }),
      " - on a plain LAN address the browser disables WebAuthn.",
      " ",
      "Already signed in on the ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "https://" }),
      " address? Use its",
      " ",
      /* @__PURE__ */ jsxRuntimeExports.jsx("em", { children: "open the local address" }),
      " link instead of typing this address — it carries your sign-in here in the URL, so no ceremony is needed."
    ] }),
    mode === "enroll" && !noWebauthn && /* @__PURE__ */ jsxRuntimeExports.jsxs("form", { onSubmit: (e) => {
      e.preventDefault();
      void run(() => enroll(label2.trim() || "admin", bootstrap.trim()));
    }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "dim", children: "No passkey is enrolled yet. The first one becomes the admin key and seals the dashboard - every later visit signs in with it." }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "field", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { htmlFor: "authgate-label", children: "key label" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            id: "authgate-label",
            value: label2,
            placeholder: "e.g. cagatay-iphone",
            onChange: (e) => setLabel(e.target.value),
            autoComplete: "off"
          }
        )
      ] }),
      (status == null ? void 0 : status.bootstrap_required) && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "field", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { htmlFor: "authgate-bootstrap", children: "bootstrap token" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            id: "authgate-bootstrap",
            type: "password",
            value: bootstrap,
            placeholder: "from the machine running the dashboard",
            onChange: (e) => setBootstrap(e.target.value),
            autoComplete: "off"
          }
        )
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "button",
        {
          className: "btn go",
          type: "submit",
          disabled: busy || (status == null ? void 0 : status.bootstrap_required) && !bootstrap.trim(),
          children: busy ? "waiting for the authenticator…" : "create passkey"
        }
      )
    ] }),
    mode === "login" && !noWebauthn && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "dim", children: "This dashboard is sealed with a passkey." }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: signIn, disabled: busy, children: busy ? "waiting for the authenticator…" : "sign in" }),
      !showToken && /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn linklike", type: "button", onClick: () => setShowToken(true), children: "passkey not working? sign in with an access token" }),
      showToken && /* @__PURE__ */ jsxRuntimeExports.jsxs("form", { onSubmit: (e) => {
        e.preventDefault();
        if (tokenValue.trim()) setAuthToken(tokenValue.trim());
      }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "field", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { htmlFor: "authgate-token", children: "access token" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              id: "authgate-token",
              type: "password",
              value: tokenValue,
              placeholder: "from the dashboard machine (tiny can mint one)",
              onChange: (e) => setTokenValue(e.target.value),
              autoComplete: "off"
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "dim", children: "The escape hatch when the passkey ceremony fails on this device: paste a session token minted on the machine running the dashboard. Wrong or expired tokens simply land back on this screen." }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", type: "submit", disabled: !tokenValue.trim(), children: "unlock" })
      ] })
    ] }),
    error && mode !== "unreachable" && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "autherror", role: "alert", children: error }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "dim", style: { fontSize: 11, opacity: 0.55, marginTop: 12 }, children: [
      "build ",
      BUILD
    ] })
  ] }) });
}
const PANELS = ["settings", "activity", "devices", "estop", "training", "record", "sim", "help"];
function panelFromHash(hash) {
  const want = hash.replace(/^#/, "");
  return PANELS.includes(want) ? want : null;
}
function initialPanel() {
  const fromHash = panelFromHash(location.hash);
  if (fromHash) return fromHash;
  const want = new URLSearchParams(location.search).get("panel");
  return PANELS.includes(want ?? "") ? want : null;
}
function Dashboard() {
  var _a, _b, _c;
  const { conn, dashboardId, peers, safetyFlash, mesh, activity, absentChildren, quietChildren, loaded, lastEventAt, everOpen } = useMesh();
  const pwa = usePwa();
  const [panel, setPanel] = reactExports.useState(initialPanel);
  function route(next) {
    setPanel(next);
    const want = next ?? "fleet";
    if (location.hash.replace(/^#/, "") !== want) location.hash = want;
  }
  const shownPanel = reactExports.useRef(panel);
  shownPanel.current = panel;
  reactExports.useEffect(() => {
    route(initialPanel());
    const onHash = () => {
      const next = panelFromHash(location.hash);
      if (shownPanel.current !== next) route(next);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const [trainPrefill, setTrainPrefill] = reactExports.useState(void 0);
  const [snipCopied, setSnipCopied] = reactExports.useState(null);
  const [boards, setBoards] = reactExports.useState(void 0);
  const [settingsTab, setSettingsTab] = reactExports.useState(void 0);
  const [detail, setDetail] = reactExports.useState(null);
  const [busyPeers, setBusyPeers] = reactExports.useState({});
  const [recordMock, setRecordMock] = reactExports.useState(null);
  const liveTwins = reactExports.useMemo(() => new Set(
    Object.values(peers).filter((p) => p.peer_id.endsWith("-twin") && !p.stale).map((p) => p.peer_id)
  ), [peers]);
  const list = reactExports.useMemo(() => Object.values(peers).filter((p) => {
    var _a2;
    const t = (_a2 = p.presence) == null ? void 0 : _a2.robot_type;
    if (t === "gateway") return false;
    if (p.peer_id.endsWith("-safety")) return false;
    return true;
  }).sort((a, b) => {
    var _a2, _b2;
    if (!!a.stale !== !!b.stale) return a.stale ? 1 : -1;
    const at = ((_a2 = a.presence) == null ? void 0 : _a2.robot_type) ?? "z", bt = ((_b2 = b.presence) == null ? void 0 : _b2.robot_type) ?? "z";
    if (at !== bt) return at === "robot" ? -1 : 1;
    return a.peer_id.localeCompare(b.peer_id);
  }), [peers]);
  const anyRunning = Object.values(busyPeers).some(Boolean);
  const fleetEmpty = loaded && list.length === 0;
  const snippet = startSnippet(boards ?? null);
  const homeRoute = fleetEmpty ? ((_a = noArmsVerdict(0, boards === void 0 ? null : boards)) == null ? void 0 : _a.route) ?? null : null;
  reactExports.useEffect(() => {
    if (!fleetEmpty || boards !== void 0) return;
    let alive = true;
    api("/api/devices").then((doc) => {
      if (!alive) return;
      const claimed = new Set(Object.values(doc.managed ?? {}).filter((m) => (m == null ? void 0 : m.alive) && (m == null ? void 0 : m.port)).map((m) => m.port));
      setBoards((doc.serial_ports ?? []).map((p) => {
        var _a2, _b2;
        return {
          peer_id: ((_a2 = p.remembered) == null ? void 0 : _a2.peer_id) ?? "",
          claimed: claimed.has(p.device),
          device: p.device,
          robot_name: ((_b2 = p.remembered) == null ? void 0 : _b2.robot_name) ?? null
        };
      }));
    }).catch(() => {
      if (alive) setBoards(null);
    });
    return () => {
      alive = false;
    };
  }, [fleetEmpty, boards]);
  const fleetHosts = reactExports.useMemo(() => armHosts(list.map((q) => {
    var _a2;
    return {
      peer_id: q.peer_id,
      joints: Object.keys(((_a2 = q.state) == null ? void 0 : _a2.joints) ?? {}).length
    };
  })), [list]);
  const pairInputs = reactExports.useMemo(() => list.map((q) => {
    var _a2, _b2;
    return {
      peer_id: q.peer_id,
      joints: Object.keys(((_a2 = q.state) == null ? void 0 : _a2.joints) ?? {}).length,
      role: q.role ?? null,
      role_volts: q.role_volts ?? null,
      role_source: q.role_source ?? null,
      robot_type: ((_b2 = q.presence) == null ? void 0 : _b2.robot_type) ?? null
    };
  }), [list]);
  reactExports.useEffect(() => {
    void pwa.keepAwake(anyRunning);
  }, [anyRunning]);
  const [dark, setDark] = reactExports.useState([]);
  reactExports.useEffect(() => {
    let live = true;
    void serverRoutePaths().then((paths) => {
      if (live) setDark(darkRoutes(paths));
    }).catch(() => {
    });
    return () => {
      live = false;
    };
  }, [backendKey()]);
  reactExports.useEffect(() => {
    let live = true;
    void getRecordApi().then((a) => {
      if (live) setRecordMock(a.mock);
    }).catch(() => {
    });
    return () => {
      live = false;
    };
  }, []);
  reactExports.useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      const verdict2 = hotkeyVerdict({
        key: e.key,
        metaKey: e.metaKey,
        ctrlKey: e.ctrlKey,
        altKey: e.altKey,
        shiftKey: e.shiftKey,
        targetTag: el == null ? void 0 : el.tagName,
        editable: el == null ? void 0 : el.isContentEditable,
        repeat: e.repeat
      });
      if (!verdict2) return;
      if (verdict2 === "close") {
        route(null);
        return;
      }
      if (e.metaKey || e.ctrlKey) e.preventDefault();
      route(verdict2 === "estop" ? "estop" : "help");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const detailPeer = detail ? peers[detail] : void 0;
  reactExports.useEffect(() => {
    if (detail && !peers[detail]) setDetail(null);
  }, [detail, peers]);
  const [linkTick, setLinkTick] = reactExports.useState(0);
  const link = linkHealth({
    conn,
    browserOnline: pwa.online,
    lastEventAt,
    everOpen,
    meshOnline: mesh.online,
    // list.length, NOT the non-stale count: what is RENDERED is what can mislead.
    peerCount: list.length,
    now: Date.now(),
    sessionExpired: sessionVerdict(authToken(), Date.now() / 1e3).refusesUntilSignIn
  });
  reactExports.useEffect(() => {
    const id = setInterval(() => setLinkTick((t) => t + 1), 1e3);
    return () => clearInterval(id);
  }, []);
  const [refused, setRefused] = reactExports.useState(null);
  const [health, setHealth] = reactExports.useState(void 0);
  reactExports.useEffect(() => {
    if (conn !== "open") return;
    let alive = true;
    const poll = () => {
      api("/api/health").then((h) => {
        if (alive) {
          setRefused((h == null ? void 0 : h.refused_handshakes) ?? null);
          setHealth(h);
        }
      }).catch(() => {
      });
    };
    poll();
    const id = setInterval(poll, 6e4);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [conn]);
  const notice = serverNotice(refused);
  const stale = reactExports.useMemo(() => staleServerNotice(health, fleetFieldGaps(peers)), [health, peers]);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "stage", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx(EstopButton, { onClick: () => route("estop"), posture: estopPosture(link) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(
      FleetBar,
      {
        conn,
        peerCount: list.filter((p) => !p.stale).length,
        dashboardId,
        safetyFlash,
        mesh,
        online: pwa.online,
        installable: pwa.installable,
        activityCount: activity.length,
        absentChildren,
        quietChildren,
        recordMock,
        onInstall: () => void pwa.install(),
        onSettings: () => {
          setSettingsTab(void 0);
          route("settings");
        },
        onWireSecurity: () => {
          setSettingsTab("mesh");
          route("settings");
        },
        onActivity: () => route("activity"),
        onDevices: () => route("devices"),
        onTraining: () => {
          setTrainPrefill(void 0);
          route("training");
        },
        onRecord: () => route("record"),
        onSim: () => route("sim"),
        onHelp: () => route("help")
      }
    ),
    pwa.needRefresh && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "toast", children: [
      "A new dashboard version is ready",
      pwa.bundleAge() ? ` — this tab loaded ${pwa.bundleAge()}` : "",
      ".",
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", onClick: pwa.update, children: "reload" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "hint", children: reloadImpact(Object.keys(busyPeers).filter((id) => busyPeers[id])).text })
    ] }),
    link.headline && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `toast ${link.commandsWork ? "" : "warn"}`, role: "status", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: link.headline }),
      " ",
      link.detail
    ] }),
    dark.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "toast warn", role: "status", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "Older server." }),
      " ",
      darkFeatureMessage(dark),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("summary", { className: "muted small", children: [
          "which ",
          dark.length === 1 ? "one" : "ones"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("ul", { className: "muted small", children: dark.map((p) => /* @__PURE__ */ jsxRuntimeExports.jsx("li", { children: p }, p)) })
      ] })
    ] }),
    notice.text && !link.headline && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "toast warn", role: "status", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "A client is being refused repeatedly" }),
      " ",
      notice.text
    ] }),
    stale.text && !notice.text && !link.headline && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "toast", role: "status", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("b", { children: "This server is older than this page" }),
      " ",
      stale.text
    ] }),
    conn === "unauthorized" ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "empty-fleet", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "empty-icon", children: "🔒" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "This dashboard requires a token" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: backendLabel() }),
        " refused the connection. Paste the token the server was started with:"
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(TokenPrompt, {}),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
        "It is the value of ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "--auth-token" }),
        " / ",
        /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "DASHBOARD_AUTH_TOKEN" }),
        "."
      ] })
    ] }) : list.length === 0 ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "empty-fleet", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "empty-icon", children: conn === "open" ? "📡" : "🔌" }),
      conn !== "open" ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("h2", { children: [
          "Not connected to ",
          backendLabel()
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: conn === "connecting" ? "Opening the mesh socket…" : "The dashboard API is unreachable." }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "If the API runs elsewhere, point this browser at it in Settings → Connection." }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => route("settings"), children: "open settings" })
      ] }) : mesh.online === false ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: "The dashboard's mesh session is down" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: "The API is up, but it is not on the robot mesh — so no peer can be seen or commanded." }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: "Check the mesh endpoints, then restart the session." }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => route("settings"), children: "mesh settings" })
      ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { children: loaded ? "No robots on the mesh yet" : "Loading the fleet…" }),
        loaded && homeRoute && /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", role: "status", children: homeRoute }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { children: "Start one anywhere on your network:" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { className: "startsnip", children: snippet.code }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "row", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost tiny", onClick: async () => {
            try {
              if (!navigator.clipboard) throw new Error("this page is not a secure origin, so the browser blocks copying");
              await navigator.clipboard.writeText(snippet.code);
              setSnipCopied("copied");
            } catch (e) {
              setSnipCopied((e == null ? void 0 : e.message) ?? String(e));
            }
          }, children: "copy" }),
          snipCopied && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: snipCopied === "copied" ? "muted small" : "warn small", role: "status", children: snipCopied })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "hint", children: snippet.provenance }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "hint", children: [
          "Set ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "STRANDS_MESH_LOCAL_DEV=1" }),
          " + ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("code", { children: "STRANDS_MESH_MULTICAST=true" }),
          " for local dev."
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn ghost", onClick: () => route("devices"), children: "spawn one here" })
      ] })
    ] }) : /* @__PURE__ */ jsxRuntimeExports.jsxs("main", { className: "grid", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(LanHint, {}),
      (() => {
        const lb = lockoutBanner(list);
        return lb ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: `lockout-banner ${lb.severity}`, role: "status", style: { gridColumn: "1 / -1" }, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { "aria-hidden": "true", children: "🛑" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: lb.text })
        ] }) : null;
      })(),
      list.map((p) => {
        var _a2;
        return /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: `the card for ${p.peer_id}`, children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          RobotCard,
          {
            peer: p,
            twinLive: liveTwins.has(`${p.peer_id}-twin`),
            hostsChildren: ((_a2 = fleetHosts[p.peer_id]) == null ? void 0 : _a2.children) ?? null,
            onOpen: setDetail,
            onBusyChange: (id, running) => setBusyPeers((s) => s[id] === running ? s : { ...s, [id]: running })
          },
          p.peer_id
        ) }, p.peer_id);
      })
    ] }),
    detailPeer && /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the robot detail view", onDismiss: () => setDetail(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(
      RobotDetail,
      {
        peer: detailPeer,
        twinLive: liveTwins.has(`${detailPeer.peer_id}-twin`),
        hostsChildren: ((_b = fleetHosts[detailPeer.peer_id]) == null ? void 0 : _b.children) ?? null,
        fleet: pairInputs,
        onOpen: setDetail,
        onClose: () => setDetail(null)
      }
    ) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "settings", onDismiss: () => route(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(SettingsDrawer, { open: panel === "settings", onClose: () => route(null), mesh, initialTab: settingsTab }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the activity log", onDismiss: () => route(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(ActivityLog, { open: panel === "activity", onClose: () => route(null), live: activity }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the devices screen", onDismiss: () => route(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(DevicePanel, { open: panel === "devices", onClose: () => route(null) }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(HelpSheet, { open: panel === "help", onClose: () => route(null) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(
      EstopSheet,
      {
        open: panel === "estop",
        onClose: () => route(null),
        linkWarning: link.commandsWork ? null : link.estopReason,
        meshBacked: mesh.online === true
      }
    ),
    panel === "training" && /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the training screen", onDismiss: () => route(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(TrainingTab, { onClose: () => route(null), prefill: trainPrefill }) }),
    panel === "sim" && /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the simulation screen", onDismiss: () => route(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(SimTab, { onClose: () => route(null) }) }),
    panel === "record" && /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the record screen", onDismiss: () => route(null), children: /* @__PURE__ */ jsxRuntimeExports.jsx(
      RecordPanel,
      {
        peers: list.filter((p) => !p.stale),
        onClose: () => route(null),
        onDevices: () => route("devices"),
        onTrain: (prefill) => {
          setTrainPrefill(prefill);
          route("training");
        }
      }
    ) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { label: "the chat dock", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
      AgentDock,
      {
        onSettings: () => route("settings"),
        startOpen: new URLSearchParams(location.search).get("panel") === "chat",
        exampleRobot: (_c = list.find((p) => {
          var _a2;
          return !p.stale && ((_a2 = p.presence) == null ? void 0 : _a2.robot_type) === "robot";
        })) == null ? void 0 : _c.peer_id
      }
    ) })
  ] });
}
function TokenPrompt() {
  const [token, setToken] = reactExports.useState("");
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "form",
    {
      className: "tokenprompt",
      onSubmit: (e) => {
        e.preventDefault();
        setAuthToken(token);
      },
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: "password",
            value: token,
            placeholder: "dashboard token",
            "aria-label": "dashboard token",
            onChange: (e) => setToken(e.target.value)
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { className: "btn go", type: "submit", disabled: !token.trim(), children: "unlock" })
      ]
    }
  );
}
function App() {
  const key = reactExports.useSyncExternalStore(subscribeAuth, backendKey);
  return /* @__PURE__ */ jsxRuntimeExports.jsx(ConfigProvider, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(AuthGate, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(Dashboard, {}) }) }, key);
}
client.createRoot(document.getElementById("root")).render(
  /* @__PURE__ */ jsxRuntimeExports.jsx(React.StrictMode, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(App, {}) })
);
