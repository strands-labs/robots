import { useEffect, useState } from 'react'

import { api, post } from '../lib/endpoints'
import { DISMISS_KEY, HintBody, handoffHref, lanHintVerdict, readDismissed } from '../lib/lanHint'

export default function LanHint() {
  const [body, setBody] = useState<HintBody | null>(null)
  const [dismissed, setDismissed] = useState<string[]>(() =>
    readDismissed(typeof localStorage === 'undefined' ? null : localStorage),
  )
  // Hooks live above the early return; `leaving` is only readable once the hint shows.
  const [leaving, setLeaving] = useState(false)

  useEffect(() => {
    let alive = true
    // A network-topology hint is never worth an error toast: on any failure (404 from a
    // server that predates it, 401 before login, offline) it stays quiet.
    api<HintBody>('/api/network/hint')
      .then(b => { if (alive) setBody(b) })
      .catch(() => {})
    return () => { alive = false }
  }, [])

  const verdict = lanHintVerdict({
    body,
    origin: typeof location === 'undefined' ? '' : location.origin,
    dismissed,
  })
  if (!verdict.show) return null

  const dismiss = () => {
    const next = [...dismissed, verdict.url]
    setDismissed(next)
    try { localStorage.setItem(DISMISS_KEY, JSON.stringify(next)) } catch { /* private mode */ }
  }

  /**
   * The LAN page is plain http — no WebAuthn — so the sign-in must ride along. Minted at
   * CLICK time (a hint can sit on screen longer than the token lives), and every failure
   * (old server 404, auth off, network) navigates to the plain link: today's behavior.
   */
  const go = async (e: React.MouseEvent<HTMLAnchorElement>) => {
    // A modified click asks for a new tab/window — let the browser have the plain link.
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return
    e.preventDefault()
    if (leaving) return
    setLeaving(true)
    let href = verdict.url
    try { href = handoffHref(verdict.url, await post<{ token?: string | null }>('/api/auth/handoff')) }
    catch { /* the plain link is the honest fallback */ }
    location.href = href
  }

  return (
    <div className="lan-hint" role="status" style={{ gridColumn: '1 / -1' }}>
      <span aria-hidden="true">&#127968;</span>
      <span className="lan-hint-text">{verdict.text}</span>
      {/* A plain link, not a redirect: leaving https for http is the operator's choice to make, and a silent downgrade would be indefensible. */}
      <a className="lan-hint-go" href={verdict.url} onClick={go}>
        {leaving ? 'carrying your sign-in over…' : 'open the local address'}
      </a>
      <button className="lan-hint-dismiss" onClick={dismiss} aria-label="dismiss this hint">
        &times;
      </button>
    </div>
  )
}
