/**
 * Where the backend lives, and how we talk to it. The dashboard is a mesh *peer*, not a hub -
 * the API it drives can be on this origin, on a robot across the LAN, or on a box behind a
 * VPN.
 */

import { routeKnown, staleRouteMessage, unroutedByDetail } from './serverAge'
import { detailSentence } from './detailSentence'

const BASE_KEY = 'strands.backend'
const TOKEN_KEY = 'strands.token'

/** `robot.lan:8080` -> `http://robot.lan:8080`; trailing slashes trimmed. */
export function normalize(raw: string): string {
  const value = (raw ?? '').trim()
  if (!value) return ''
  const withScheme = /^[a-z]+:\/\//i.test(value) ? value : `http://${value}`
  try {
    const url = new URL(withScheme)
    // ws:// typed into the field is a natural mistake - accept it.
    if (url.protocol === 'ws:') url.protocol = 'http:'
    if (url.protocol === 'wss:') url.protocol = 'https:'
    // Only a scheme fetch can actually speak.
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return ''
    return url.origin
  } catch {
    return ''
  }
}

let cachedBase: string | null = null
let absorbedUrl = false
/** `?backend=` from the URL, once — null when the URL said nothing. */
let urlBase: string | null = null

/** Take the credentials off the URL. */
function absorbUrl(): void {
  if (absorbedUrl) return
  absorbedUrl = true
  try {
    const params = new URLSearchParams(location.search)
    const fromToken = params.get('token')
    if (fromToken) localStorage.setItem(TOKEN_KEY, fromToken)
    urlBase = params.get('backend')
    // Scrub what was absorbed: a ?token= URL outlives its token in history,
    // share sheets and screenshots, and must not be re-sent on reload.
    if (fromToken !== null || urlBase !== null) {
      try {
        params.delete('token')
        params.delete('backend')
        const rest = params.toString()
        history.replaceState(null, '', `${location.pathname}${rest ? `?${rest}` : ''}${location.hash || ''}`)
      } catch { /* no history (a test stub): the values are absorbed either way */ }
    }
  } catch {
    urlBase = null // no location (a test, a worker): the stored values are the whole truth
  }
}

export function backendBase(): string {
  absorbUrl()
  if (cachedBase !== null) return cachedBase
  // ?backend=... wins once, then persists.
  if (urlBase !== null) {
    cachedBase = normalize(urlBase)
    localStorage.setItem(BASE_KEY, cachedBase)
    return cachedBase
  }
  cachedBase = normalize(localStorage.getItem(BASE_KEY) ?? '')
  return cachedBase
}

export function authToken(): string {
  absorbUrl()
  return (localStorage.getItem(TOKEN_KEY) ?? '').trim()
}

// Auth/backend changes must reach React: localStorage writes emit no event in the
// writing tab, so components subscribe here (App keys ConfigProvider off backendKey()).
const authListeners = new Set<() => void>()
export function subscribeAuth(fn: () => void): () => void {
  authListeners.add(fn)
  return () => { authListeners.delete(fn) }
}
function notifyAuth(): void {
  for (const fn of authListeners) fn()
}

export function setAuthToken(token: string): void {
  const value = token.trim()
  if (value) localStorage.setItem(TOKEN_KEY, value)
  else localStorage.removeItem(TOKEN_KEY)
  notifyAuth()
}

/** Human label for the connection chip. */
export function backendLabel(): string {
  const base = backendBase()
  return base ? base.replace(/^https?:\/\//, '') : `${location.host} (this origin)`
}

/** Identity of the current connection. */
export function backendKey(): string {
  return `${backendBase()}|${authToken() ? 'auth' : 'open'}`
}

export function setBackendBase(raw: string): void {
  cachedBase = normalize(raw)
  if (cachedBase) localStorage.setItem(BASE_KEY, cachedBase)
  else localStorage.removeItem(BASE_KEY)
  // The route list belongs to the server we were talking to.
  forgetLiveRoutes()
  notifyAuth() // backendKey() changed
}

export function apiUrl(path: string): string {
  const base = backendBase()
  return base ? `${base}${path}` : path
}

export function wsUrl(path: string): string {
  const base = backendBase()
  const origin = base || location.origin
  const url = new URL(path, origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  // A browser cannot set headers on a WebSocket handshake. The server reads the
  // session from the `strands_dash` cookie the login ceremony set and never from
  // the query string (a token there lands in access logs), so nothing is added
  // here: a socket is admitted exactly when the page's cookie is.
  return url.toString()
}

export class HttpError extends Error {
  status: number
  body: any
  constructor(status: number, message: string, body?: any) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.body = body
  }
}

/**
 * fetch + auth + JSON + *real* errors. `catch {}` around a fetch is how a dashboard ends up
 * showing a robot as idle when the command never landed.
 */
let _liveRoutes: string[] | null = null
let _liveRoutesTried = false
let _liveRoutesAt = 0

export const LIVE_ROUTES_TTL_MS = 60_000

/** The running server's route table (openapi.json), cached with a TTL. */
export async function serverRoutePaths(): Promise<string[] | null> { return liveRoutes() }

async function liveRoutes(): Promise<string[] | null> {
  if (_liveRoutesTried && Date.now() - _liveRoutesAt < LIVE_ROUTES_TTL_MS) return _liveRoutes
  _liveRoutesTried = true
  _liveRoutesAt = Date.now()
  try {
    const token = authToken()
    const res = await fetch(apiUrl('/openapi.json'), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) {
      // Guarded route, so a 401/403 here is as good a witness as any other refusal. It stays SILENT
      // otherwise (a server without /openapi.json is not an error) — only the accounting is added.
      noteAuthRefusal(res.status)
      return null
    }
    noteAuthAccepted('/openapi.json')
    const doc = await res.json()
    const paths = doc && doc.paths && typeof doc.paths === 'object' ? Object.keys(doc.paths) : []
    _liveRoutes = paths.length ? paths : null
  } catch {
    _liveRoutes = null // a server without /openapi.json, or no network: stay silent
  }
  return _liveRoutes
}

/** Test seam, and what a backend switch calls to forget what the OLD server routed. */
export function forgetLiveRoutes(): void {
  _liveRoutes = null
  _liveRoutesTried = false
  _liveRoutesAt = 0
}

let _refusedAt: number | null = null

export function noteAuthRefusal(status: number, at: number = Date.now()): void {
  if (status === 401 || status === 403) _refusedAt = at
}

/**
 * The server's own PUBLIC_PATHS (server.py): these answer 200 whether or not the credentials
 * are any good, because the middleware never looks at them.
 */
const PROVES_NOTHING = [
  '/api/health',
  '/api/auth/status',
  '/api/auth/register/',
  '/api/auth/login/',
]

/**
 * Clear it when a GUARDED request succeeds — that is the only answer which proves the
 * credentials work.
 */
export function noteAuthAccepted(path?: string): void {
  if (path !== undefined && PROVES_NOTHING.some(p => path.startsWith(p))) return
  _refusedAt = null
}

/** Has this page been refused recently enough to explain a socket that never opened? */
export function authRefusedRecently(withinMs = 60_000, now: number = Date.now()): boolean {
  return _refusedAt !== null && now - _refusedAt <= withinMs
}

let lastRenewalAtS = 0

/** When this page last had a session renewal accepted (seconds, 0 = never). */
export function lastRenewalAt(): number {
  return lastRenewalAtS
}

export function absorbRenewedSession(res: { headers?: { get(name: string): string | null } } | null): boolean {
  let offered: string | null = null
  try {
    offered = res?.headers?.get('X-Session-Token') ?? null
  } catch {
    return false // a Response-like without real headers (a stub, a blob shim) is not an error
  }
  const fresh = (offered ?? '').trim()
  if (!fresh) return false
  const current = (localStorage.getItem(TOKEN_KEY) ?? '').trim()
  if (!current || fresh === current) return false
  if (fresh.split('.').length !== 3) return false
  setAuthToken(fresh)
  lastRenewalAtS = Date.now() / 1000
  return true
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const token = authToken()
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) }
  if (init.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json'
  if (token) headers['Authorization'] = `Bearer ${token}`

  let res: Response
  try {
    res = await fetch(apiUrl(path), { ...init, headers })
  } catch (e) {
    throw new HttpError(0, `cannot reach ${backendLabel()}: ${e instanceof Error ? e.message : e}`)
  }
  absorbRenewedSession(res)
  const text = await res.text()
  let body: any = text
  try { body = text ? JSON.parse(text) : null } catch { /* keep raw text */ }
  if (!res.ok) {
    noteAuthRefusal(res.status)
    const detail = (body && (body.detail ?? body.error)) || text || res.statusText
    let message = detailSentence(detail) || text || res.statusText
    if (res.status === 404 && (routeKnown(await liveRoutes(), path) === false
        || unroutedByDetail(body && (body.detail ?? null)))) {
      message = staleRouteMessage(path)
    }
    throw new HttpError(res.status, message, body)
  }
  noteAuthAccepted(path)
  return body as T
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body === undefined ? '{}' : JSON.stringify(body) })

/** DELETE. */
export const del = <T = any>(path: string) => api<T>(path, { method: 'DELETE' })

/** Authed fetch of a binary endpoint (camera previews), returned as an object URL. */
export async function apiBlob(path: string): Promise<string> {
  const token = authToken()
  const headers: Record<string, string> = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  let res: Response
  try {
    res = await fetch(apiUrl(path), { headers })
  } catch (e) {
    throw new HttpError(0, `cannot reach ${backendLabel()}: ${e instanceof Error ? e.message : e}`)
  }
  absorbRenewedSession(res)
  if (!res.ok) {
    noteAuthRefusal(res.status)
    const text = await res.text()
    let detail: unknown = text || res.statusText
    try { detail = JSON.parse(text).detail ?? detail } catch { /* raw text */ }
    // Same rail as api(): a camera preview refused with a structured detail (409 PermissionError,
    // 503 with the driver's words) is read by a person too.
    throw new HttpError(res.status, detailSentence(detail) || text || res.statusText)
  }
  noteAuthAccepted(path)
  return URL.createObjectURL(await res.blob())
}
