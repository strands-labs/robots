/**
 * The calibration wizard — runs lerobot-calibrate through the dashboard (the backend
 * drives the real CLI under a pty; see dashboard/calibration_run.py). This component
 * is deliberately thin: every sentence and button comes from lib/calibrateWizard.ts,
 * where it is pinned by tests. The terminal command remains available in the drawer
 * as a fallback — this wizard is the same procedure, not a different one.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api, post } from '../lib/endpoints'
import { confirmSheet, wizardView, type WizardStatus } from '../lib/calibrateWizard'

interface Props {
  port: string
  role: 'follower' | 'leader'
  model: string
  deviceId: string
  /** the calibration list above should reload after a save */
  onSaved?: () => void
  onClose?: () => void
}

export default function CalibrateWizard({ port, role, model, deviceId, onSaved, onClose }: Props) {
  const [phase, setPhase] = useState<'confirm' | 'running' | 'closed'>('confirm')
  const [status, setStatus] = useState<WizardStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const sid = useRef<string | null>(null)
  const savedTold = useRef(false)

  // Poll while a run is live. The backend is the state machine; we only look.
  useEffect(() => {
    if (phase !== 'running' || !sid.current) return
    let stop = false
    const tick = async () => {
      try {
        const s = await api<WizardStatus>(`/api/calibration/run/${sid.current}`)
        if (stop) return
        setStatus(s)
        if (s.step === 'saved' && !savedTold.current) {
          savedTold.current = true
          onSaved?.()
        }
      } catch (e) {
        if (!stop) setError((e as Error)?.message ?? String(e))
      }
    }
    void tick()
    const t = setInterval(() => {
      // saved/failed screens are stable — stop asking
      void tick()
    }, 600)
    return () => { stop = true; clearInterval(t) }
  }, [phase, onSaved])

  const start = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const s = await post<WizardStatus>('/api/calibration/run', { role, model, device_id: deviceId, port })
      sid.current = s.id
      savedTold.current = false
      setStatus(s)
      setPhase('running')
    } catch (e) {
      setError((e as Error)?.message ?? String(e))
    } finally {
      setBusy(false)
    }
  }, [role, model, deviceId, port])

  const press = useCallback(async (key: 'enter' | 'c' | 'cancel' | 'close') => {
    if (key === 'close') { setPhase('closed'); onClose?.(); return }
    if (!sid.current) return
    setBusy(true)
    setError(null)
    try {
      const path = key === 'cancel'
        ? `/api/calibration/run/${sid.current}/cancel`
        : `/api/calibration/run/${sid.current}/key`
      const s = await post<WizardStatus>(path, key === 'cancel' ? {} : { key })
      setStatus(s)
    } catch (e) {
      setError((e as Error)?.message ?? String(e))
    } finally {
      setBusy(false)
    }
  }, [onClose])

  if (phase === 'closed') return null

  if (phase === 'confirm') {
    const c = confirmSheet({ port, deviceId, model })
    return (
      <div className="calibwizard" role="dialog" aria-label={c.title}>
        <h4>{c.title}</h4>
        <p>{c.body}</p>
        {error && <div className="result bad">⚠ {error}</div>}
        <div className="row">
          <button className="btn" disabled={busy} onClick={() => void start()}>
            {busy ? 'starting…' : 'start — the arm goes limp now'}
          </button>
          <button className="btn ghost" onClick={() => { setPhase('closed'); onClose?.() }}>not now</button>
        </div>
      </div>
    )
  }

  if (!status) return <p className="hint">starting the calibration session…</p>
  const v = wizardView(status)

  return (
    <div className="calibwizard" role="dialog" aria-label={v.title} aria-live="polite">
      <h4 className={v.tone === 'bad' ? 'warn' : undefined}>{v.title}</h4>
      <p>{v.body}</p>

      {v.motors && v.motors.length > 0 && (
        <table className="jointtable">
          <thead>
            <tr><th>joint</th><th>min</th><th>now</th><th>max</th><th></th></tr>
          </thead>
          <tbody>
            {v.motors.map(m => (
              <tr key={m.name} className={v.unmoved.includes(m.name) ? 'dead' : undefined}>
                <td>{m.name}</td>
                <td className="mono">{m.min}</td>
                <td className="mono">{m.pos}</td>
                <td className="mono">{m.max}</td>
                <td>{v.unmoved.includes(m.name) ? 'has not moved yet' : '✓'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {v.motors && v.motors.length === 0 && (
        <p className="hint">waiting for the first position read…</p>
      )}

      {error && <div className="result bad">⚠ {error}</div>}
      {v.detail && (
        <details>
          <summary>raw output</summary>
          <pre className="logtail">{v.detail}</pre>
        </details>
      )}

      <div className="row">
        {v.buttons.map(b => (
          <button
            key={b.key}
            className={b.primary ? 'btn' : b.danger ? 'btn ghost danger' : 'btn ghost'}
            disabled={busy}
            onClick={() => void press(b.key)}
          >
            {b.label}
          </button>
        ))}
      </div>
    </div>
  )
}
