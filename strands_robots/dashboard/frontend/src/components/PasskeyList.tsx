import { useCallback, useEffect, useState } from 'react'
import { api, del, HttpError } from '../lib/endpoints'
import { passkeyRows, passkeySummary, revokeRefusal, type Credential } from '../lib/passkeyList'

export default function PasskeyList({ authRequired }: { authRequired: boolean }) {
  const [creds, setCreds] = useState<Credential[] | null>(null)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState('')
  const [absent, setAbsent] = useState(false)

  const load = useCallback(async () => {
    try {
      const r = await api<{ credentials?: Credential[] }>('/api/auth/credentials')
      setCreds(r.credentials ?? [])
    } catch (e) {
      // An older server has no such route; say nothing rather than claim there are no keys.
      if (e instanceof HttpError && e.status === 404) setAbsent(true)
      else setMsg(e instanceof Error ? e.message : String(e))
    }
  }, [])
  useEffect(() => { void load() }, [load])

  if (absent) return null
  const rows = passkeyRows(creds)

  const revoke = async (id: string, label: string) => {
    setMsg(''); setBusy(id)
    try {
      await del(`/api/auth/credentials/${encodeURIComponent(id)}`)
      setMsg(`removed ${label}`)
      await load()
    } catch (e) {
      setMsg(`✗ ${revokeRefusal(e instanceof HttpError ? e.status : 0,
        e instanceof Error ? e.message : String(e))}`)
    } finally { setBusy('') }
  }

  return (
    <div>
      <h4>Devices that can sign in</h4>
      {creds === null ? <p className="hint">reading…</p> : (
        <>
          <p className="hint">{passkeySummary(rows, authRequired)}</p>
          {rows.map(r => (
            <div key={r.id} className="row between cg-row">
              <span className="muted small">
                <b>{r.label}</b>{r.when ? ` · ${r.when}` : ''}
                {!r.revocable && <span className="hint small"> — {r.reason}</span>}
              </span>
              <button className="btn ghost tiny danger" disabled={!r.revocable || busy === r.id}
                      title={r.revocable ? 'this device can no longer sign in' : r.reason}
                      onClick={() => void revoke(r.id, r.label)}>
                {busy === r.id ? 'removing…' : 'remove'}
              </button>
            </div>
          ))}
        </>
      )}
      {msg && <p className={msg.startsWith('✗') ? 'warn small' : 'hint small'}>{msg}</p>}
      <p className="hint">
        Removing a passkey takes effect immediately; a browser already signed in keeps its session
        until it reloads.
      </p>
    </div>
  )
}
