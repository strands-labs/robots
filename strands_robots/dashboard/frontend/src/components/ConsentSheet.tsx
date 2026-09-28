import { useEffect, useRef, useState } from 'react'
import {
  afterApproval,
  approveConsent,
  blockedReason,
  canApprove,
  severity,
  type ConsentNeed,
} from '../lib/consent'

type Props = {
  need: ConsentNeed
  /** 'spawn' can retry immediately; a running peer must be respawned first. */
  target: 'spawn' | 'peer'
  onCancel: () => void
  /** Called only when the grant landed AND an immediate retry makes sense. */
  onRetry: () => void
}

export default function ConsentSheet({ need, target, onCancel, onRetry }: Props) {
  const cancel = useRef<HTMLButtonElement>(null)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const allowed = canApprove(need)
  const danger = severity(need) === 'danger'

  // Focus the safe choice: approving grants a permission, and no permission
  // should be one stray Enter away.
  useEffect(() => {
    cancel.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  const approve = async () => {
    setBusy(true); setError(null)
    try {
      const result = await approveConsent(need)
      const verdict = afterApproval(result, target)
      setNote(verdict.note)
      if (verdict.retryNow) onRetry()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="sheet-backdrop" onClick={onCancel}>
      <div className="sheet consent-sheet" onClick={e => e.stopPropagation()}
           role="dialog" aria-modal="true" aria-label="Security approval">
        <h2>{danger ? '⚠️' : '🔒'} {need.title}</h2>
        <p className="cs-risk">{need.risk}</p>

        {need.grants?.length ? (
          <>
            <div className="cs-label">approving allows</div>
            <ul className="cs-grants">
              {need.grants.map(g => <li key={g}>{g}</li>)}
            </ul>
          </>
        ) : null}

        {!allowed ? (
          <p className="cs-blocked">{blockedReason(need)}</p>
        ) : null}

        {/* both of these are ANSWERS TO PRESSING "approve", rendered inside a modal — the shape EstopSheet and CameraConfigSheet already fixed. */}
        {note ? <p className="cs-note" role="status">{note}</p> : null}
        {error ? <p className="cs-error" role="alert">could not save the approval: {error}</p> : null}

        {need.message ? (
          <details className="cs-raw">
            <summary>what the refusal said</summary>
            <pre>{need.message}</pre>
          </details>
        ) : null}

        {need.env_var ? (
          <p className="hint">
            Stored on this machine as <code>{need.env_var}</code> — it survives a restart, and it is
            listed under Settings → Security → “Permissions you granted”, where you can revoke it.
          </p>
        ) : null}

        <div className="sheet-actions">
          <button className="btn ghost" ref={cancel} onClick={onCancel}>
            {note ? 'close' : 'cancel'}
          </button>
          <button
            className={`btn ${danger ? 'danger' : 'primary'}`}
            disabled={!allowed || busy || !!note}
            onClick={approve}
          >
            {busy ? 'saving…' : target === 'spawn' ? 'approve & start' : 'approve'}
          </button>
        </div>
      </div>
    </div>
  )
}
