/**
 * WebAuthn (passkey) client ceremonies for the dashboard auth gate. The server
 * (dashboard/auth.py) speaks the standard wire format: options come down with
 * base64url-encoded binary fields, credentials go back the same way.
 */
import { api } from './endpoints'

// ---- base64url <-> ArrayBuffer (WebAuthn wire format) ----

export function b64uToBuf(s: string): ArrayBuffer {
  const norm = s.replace(/-/g, '+').replace(/_/g, '/')
  const pad = norm.length % 4 ? '='.repeat(4 - (norm.length % 4)) : ''
  const bin = atob(norm + pad)
  const buf = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
  return buf.buffer
}

export function bufToB64u(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf)
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

// ---- options JSON -> live structs the browser API accepts ----

export function prepCreate(opts: any): PublicKeyCredentialCreationOptions {
  const out = { ...opts }
  out.challenge = b64uToBuf(opts.challenge)
  out.user = { ...opts.user, id: b64uToBuf(opts.user.id) }
  if (opts.excludeCredentials) {
    out.excludeCredentials = opts.excludeCredentials.map((c: any) => ({ ...c, id: b64uToBuf(c.id) }))
  }
  return out
}

export function prepGet(opts: any): PublicKeyCredentialRequestOptions {
  const out = { ...opts }
  out.challenge = b64uToBuf(opts.challenge)
  delete out.allowCredentials
  return out
}

/** PublicKeyCredential -> the JSON shape auth.py verifies. */
export function credToJSON(cred: any): any {
  const r = cred.response
  const out: any = {
    id: cred.id,
    rawId: bufToB64u(cred.rawId),
    type: cred.type,
    clientExtensionResults: cred.getClientExtensionResults ? cred.getClientExtensionResults() : {},
    response: {} as any,
  }
  if (r.attestationObject !== undefined) {
    out.response.attestationObject = bufToB64u(r.attestationObject)
    out.response.clientDataJSON = bufToB64u(r.clientDataJSON)
  } else {
    out.response.authenticatorData = bufToB64u(r.authenticatorData)
    out.response.clientDataJSON = bufToB64u(r.clientDataJSON)
    out.response.signature = bufToB64u(r.signature)
    out.response.userHandle = r.userHandle ? bufToB64u(r.userHandle) : null
  }
  return out
}

/** WebAuthn only exists in a secure context (https:// or http://localhost). */
export function webauthnReady(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.isSecureContext === true &&
    typeof navigator !== 'undefined' &&
    !!navigator.credentials &&
    typeof navigator.credentials.create === 'function' &&
    typeof (window as any).PublicKeyCredential !== 'undefined'
  )
}

export interface AuthStatus {
  enabled: boolean
  setup_required: boolean
  bootstrap_required: boolean
  rp_id: string
  secure_context: boolean
  rpid_usable: boolean
  authenticated: boolean
  credentials: Array<{ id: string; label: string }>
}

export function fetchAuthStatus(): Promise<AuthStatus> {
  return api<AuthStatus>('/api/auth/status')
}

// ---- ceremonies (browser only) ----

/** First-time (or additional) passkey enrollment. Returns the session token. */
export async function enroll(label: string, bootstrap = ''): Promise<string> {
  const { challenge_id, options } = await api('/api/auth/register/begin', {
    method: 'POST',
    body: JSON.stringify({ label, bootstrap }),
  })
  const cred = await navigator.credentials.create({ publicKey: prepCreate(options) })
  if (!cred) throw new Error('passkey creation was cancelled')
  const res = await api('/api/auth/register/finish', {
    method: 'POST',
    body: JSON.stringify({ challenge_id, credential: credToJSON(cred) }),
  })
  return res.token as string
}

/**
 * A login challenge fetched AHEAD of the tap. iOS Safari only opens the Face ID sheet while
 * the tap's transient user-activation is alive — an awaited network round-trip inside the
 * click handler spends it, and credentials.get() then hangs forever without an error.
 */
export interface PreparedLogin { challenge_id: string; options: any; t: number }

/** Server keeps challenges 300s; treat ours as fresh for 240s. */
export function loginFresh(p: PreparedLogin | null): p is PreparedLogin {
  return !!p && Date.now() - p.t < 240_000
}

export async function beginLogin(): Promise<PreparedLogin> {
  const { challenge_id, options } = await api('/api/auth/login/begin', {
    method: 'POST',
    body: JSON.stringify({}),
  })
  // Discoverable-credential flow: drop the allow list.
  delete (options as any).allowCredentials
  return { challenge_id, options, t: Date.now() }
}

/** Run the authenticator ceremony for a prepared challenge. */
export async function completeLogin(p: PreparedLogin, timeoutMs = 75_000): Promise<string> {
  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort(), timeoutMs)
  let cred: any
  try {
    cred = await navigator.credentials.get({ publicKey: prepGet(p.options), signal: ac.signal })
  } catch (e: any) {
    if (ac.signal.aborted) throw new Error('the authenticator did not answer in time — tap sign in to try again')
    throw e
  } finally { clearTimeout(timer) }
  if (!cred) throw new Error('passkey sign-in was cancelled')
  const res = await api('/api/auth/login/finish', {
    method: 'POST',
    body: JSON.stringify({ challenge_id: p.challenge_id, credential: credToJSON(cred) }),
  })
  return res.token as string
}

/** Sign in with an already-enrolled passkey. Returns the session token. */
export async function login(): Promise<string> {
  return completeLogin(await beginLogin())
}
