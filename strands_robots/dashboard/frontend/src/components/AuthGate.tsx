/**
 * AuthGate - the passkey door in front of the dashboard. Decision on mount (and after every
 * backend/token change, because App remounts): 1. GET /api/auth/status (public) + probe GET
 * /api/fleet (guarded) in parallel. 2. Fleet answered 200 -> open.
 */
import { useEffect, useRef, useState } from 'react'
import { api, setAuthToken, authToken, authRefusedRecently, HttpError, lastRenewalAt } from '../lib/endpoints'
import { sessionVerdict } from '../lib/sessionExpiry'
import {
  fetchAuthStatus, enroll, webauthnReady, type AuthStatus,
  beginLogin, completeLogin, loginFresh, type PreparedLogin,
} from '../lib/passkey'
import StrandsMark from './StrandsMark'

/** Build stamp so a human (or a screenshot) can tell WHICH gate they're on. */
const BUILD = (import.meta as any).env?.VITE_BUILD ?? 'dev'

type Mode = 'checking' | 'open' | 'enroll' | 'login' | 'unreachable'

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<Mode>('checking')
  const [status, setStatus] = useState<AuthStatus | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [label, setLabel] = useState('')
  const [bootstrap, setBootstrap] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [expiring, setExpiring] = useState('')
  const [tokenValue, setTokenValue] = useState('')
  // Login challenge fetched AHEAD of the tap: iOS Safari only opens the Face ID sheet while the
  // tap's user-activation is alive, so the click handler must reach credentials.get() without
  // awaiting the network first.
  const prepared = useRef<PreparedLogin | null>(null)
  const verifying = useRef(false)

  useEffect(() => {
    if (mode !== 'open') { setExpiring(''); return }
    let alive = true
    const check = () => {
      if (!alive) return
      const v = sessionVerdict(authToken(), Date.now() / 1000, lastRenewalAt())
      if (v.refusesUntilSignIn) {
        setExpiring('')
        setError(v.text ?? 'this sign-in has expired')
        // Which door to show depends on whether a credential exists at all; ask, but never
        // leave the operator on a stale "open" page if that request is the one being refused.
        fetchAuthStatus()
          .then(st => { if (alive) { setStatus(st); setMode(st.setup_required ? 'enroll' : 'login') } })
          .catch(() => { if (alive) setMode('login') })
        return
      }
      setExpiring(v.state === 'expiring' ? (v.text ?? '') : '')

      if (authRefusedRecently() && !verifying.current) {
        verifying.current = true
        api('/api/fleet')
          .then(() => { verifying.current = false })
          .catch(e => {
            verifying.current = false
            const denied = e instanceof HttpError && (e.status === 401 || e.status === 403)
            if (!alive || !denied) return
            setExpiring('')
            setError('this page is not signed in any more — the server refused it')
            fetchAuthStatus()
              .then(st => { if (alive) { setStatus(st); setMode(st.setup_required ? 'enroll' : 'login') } })
              .catch(() => { if (alive) setMode('login') })
          })
      }
    }
    check()
    // 30s is fine for a five-minute warning and costs nothing: no network, one base64 decode.
    const t = setInterval(check, 30_000)
    // A phone spends most of its life with the tab hidden; the timer may be throttled to
    // minutes there, so re-check the moment it comes back rather than trusting the interval.
    window.addEventListener('focus', check)
    document.addEventListener('visibilitychange', check)
    return () => {
      alive = false; clearInterval(t)
      window.removeEventListener('focus', check)
      document.removeEventListener('visibilitychange', check)
    }
  }, [mode])

  useEffect(() => {
    if (mode !== 'login' || !webauthnReady()) return
    let alive = true
    const arm = () => {
      if (loginFresh(prepared.current)) return
      beginLogin().then(p => { if (alive) prepared.current = p }).catch(() => {})
    }
    arm()
    const t = setInterval(arm, 200_000) // refresh before the 300s server TTL
    window.addEventListener('focus', arm)
    document.addEventListener('visibilitychange', arm)
    return () => {
      alive = false; clearInterval(t)
      window.removeEventListener('focus', arm)
      document.removeEventListener('visibilitychange', arm)
    }
  }, [mode])

  /** Tap handler: synchronous path into the authenticator when armed. */
  function signIn() {
    const p = loginFresh(prepared.current) ? prepared.current : null
    prepared.current = null
    if (p) {
      void run(() => completeLogin(p)) // first await inside = credentials.get()
    } else {
      // Not armed (first paint, stale, or begin failed): fetch then re-arm and
      // ask for one more tap rather than risk a dead sheet on iOS.
      setBusy(true); setError('')
      beginLogin()
        .then(np => { prepared.current = np; setBusy(false); setError('ready — tap sign in again') })
        .catch(e => { setBusy(false); setError(String((e as Error).message ?? e)) })
    }
  }

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const [st, fleet] = await Promise.allSettled([fetchAuthStatus(), api('/api/fleet')])
        if (!alive) return
        if (fleet.status === 'fulfilled') { setMode('open'); return }
        const denied = fleet.reason instanceof HttpError && (fleet.reason.status === 401 || fleet.reason.status === 403)
        if (!denied) { setMode('unreachable'); setError(String((fleet.reason as Error)?.message ?? fleet.reason)); return }
        if (st.status !== 'fulfilled') { setMode('unreachable'); setError('auth status unavailable'); return }
        setStatus(st.value)
        setMode(st.value.setup_required ? 'enroll' : 'login')
      } catch (e) {
        if (alive) { setMode('unreachable'); setError(String((e as Error).message ?? e)) }
      }
    })()
    return () => { alive = false }
  }, [])

  async function run(fn: () => Promise<string>) {
    setBusy(true); setError('')
    try {
      const token = await fn()
      setAuthToken(token) // notifies subscribeAuth -> App remounts with the new key
      setMode('open')     // and this gate instance opens NOW, not on the next refresh
    } catch (e) {
      const msg = e instanceof HttpError ? (e.body?.detail ?? e.message) : (e as Error).message
      setError(String(msg || 'the passkey ceremony failed'))
      setBusy(false)
    }
  }

  if (mode === 'open') {
    // role=status, not alert: the session still works, and a modal here would interrupt a
    // recording to announce something that is five minutes away.
    return (
      <>
        {expiring && (
          <div className="sessionwarn" role="status" aria-live="polite">
            <span aria-hidden="true">⏳</span> {expiring}
          </div>
        )}
        {children}
      </>
    )
  }

  if (mode === 'checking') {
    return (
      <div className="authgate" role="status" aria-live="polite">
        <div className="authcard"><StrandsMark size={40} /><p className="dim">checking access…</p></div>
      </div>
    )
  }

  const noWebauthn = (mode === 'enroll' || mode === 'login') && !webauthnReady()

  return (
    <div className="authgate">
      <div className="authcard" role="dialog" aria-labelledby="authgate-title">
        <StrandsMark size={40} />
        {/* Identity on the gate: an anonymous credential prompt is what a phishing page looks like. */}
        <p className="authhost">strands robots · {window.location.host}</p>
        <h1 id="authgate-title">
          {mode === 'unreachable' ? 'backend unreachable'
            : mode === 'enroll' ? 'create the admin passkey'
            : 'unlock with your passkey'}
        </h1>

        {mode === 'unreachable' && (
          <p className="dim">
            The dashboard API did not answer. {error && <code>{error}</code>}
          </p>
        )}

        {noWebauthn && (
          <p className="authwarn">
            Passkeys need a secure context. Open this page over <code>https://</code> or{' '}
            <code>http://localhost</code> - on a plain LAN address the browser disables WebAuthn.
            {' '}Already signed in on the <code>https://</code> address? Use its{' '}
            <em>open the local address</em> link instead of typing this address — it carries
            your sign-in here in the URL, so no ceremony is needed.
          </p>
        )}

        {mode === 'enroll' && !noWebauthn && (
          <form onSubmit={e => { e.preventDefault(); void run(() => enroll(label.trim() || 'admin', bootstrap.trim())) }}>
            <p className="dim">
              No passkey is enrolled yet. The first one becomes the admin key and seals the
              dashboard - every later visit signs in with it.
            </p>
            <div className="field">
              <label htmlFor="authgate-label">key label</label>
              <input id="authgate-label" value={label} placeholder="e.g. cagatay-iphone"
                     onChange={e => setLabel(e.target.value)} autoComplete="off" />
            </div>
            {status?.bootstrap_required && (
              <div className="field">
                <label htmlFor="authgate-bootstrap">bootstrap token</label>
                <input id="authgate-bootstrap" type="password" value={bootstrap}
                       placeholder="from the machine running the dashboard"
                       onChange={e => setBootstrap(e.target.value)} autoComplete="off" />
              </div>
            )}
            <button className="btn go" type="submit"
                    disabled={busy || (status?.bootstrap_required && !bootstrap.trim())}>
              {busy ? 'waiting for the authenticator…' : 'create passkey'}
            </button>
          </form>
        )}

        {mode === 'login' && !noWebauthn && (
          <>
            <p className="dim">This dashboard is sealed with a passkey.</p>
            <button className="btn go" onClick={signIn} disabled={busy}>
              {busy ? 'waiting for the authenticator…' : 'sign in'}
            </button>
            {!showToken && (
              <button className="btn linklike" type="button" onClick={() => setShowToken(true)}>
                passkey not working? sign in with an access token
              </button>
            )}
            {showToken && (
              <form onSubmit={e => { e.preventDefault(); if (tokenValue.trim()) setAuthToken(tokenValue.trim()) }}>
                <div className="field">
                  <label htmlFor="authgate-token">access token</label>
                  <input id="authgate-token" type="password" value={tokenValue}
                         placeholder="from the dashboard machine (tiny can mint one)"
                         onChange={e => setTokenValue(e.target.value)} autoComplete="off" />
                </div>
                <p className="dim">
                  The escape hatch when the passkey ceremony fails on this device: paste a
                  session token minted on the machine running the dashboard. Wrong or expired
                  tokens simply land back on this screen.
                </p>
                <button className="btn go" type="submit" disabled={!tokenValue.trim()}>unlock</button>
              </form>
            )}
          </>
        )}

        {error && mode !== 'unreachable' && <p className="autherror" role="alert">{error}</p>}
        <p className="dim" style={{ fontSize: 11, opacity: 0.55, marginTop: 12 }}>build {BUILD}</p>
      </div>
    </div>
  )
}
