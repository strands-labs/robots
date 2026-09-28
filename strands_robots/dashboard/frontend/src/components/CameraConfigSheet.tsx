import { useEffect, useRef, useState } from 'react'
import { api, post } from '../lib/endpoints'
import { CamRow, applySummary, configFromRows, focusTarget, parseIndexOrPath, previewRateNote, rowsFromConfig } from '../lib/cameraConfig'
import { useConfig } from '../lib/useConfig'
import { forgetJointFailure } from '../lib/useJointFailure'

interface Detected { index: number; label?: string | null; in_use_by?: string | null }
interface Mode { width: number; height: number; fps: number }
interface Probe { busy?: boolean; error?: string; modes?: Mode[] }

export default function CameraConfigSheet({ peerId, onClose, focusCam = null, startAdding = false }: {
  peerId: string
  onClose: () => void
  /** opened from a specific camera on the robot surface: land on that row */
  focusCam?: string | null
  /** opened from an "add camera" affordance: start with a fresh row */
  startAdding?: boolean
}) {
  const { config } = useConfig()
  const [rows, setRows] = useState<CamRow[] | null>(null)
  // The fastest capture rate the operator has asked for — the one a slower
  // publish rate would visibly contradict. Blank fps rows claim nothing, so
  // they are not counted.
  const askedFps = (rows ?? []).map(r => Number(r.fps)).filter(n => Number.isFinite(n) && n > 0)
  const rateNote = previewRateNote(askedFps.length ? Math.max(...askedFps) : null, config?.mesh?.camera_hz)
  const [detected, setDetected] = useState<Detected[]>([])
  const [notManaged, setNotManaged] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState<string | null>(null)
  // Captured once: the sheet is remounted per open, and a re-render must not re-add a row.
  const opened = useRef({ focusCam, startAdding })
  // Autofocus belongs to the FIRST paint of the loaded rows only — edits remount nothing.
  const focus = rows !== null ? focusTarget(rows, opened.current.focusCam, opened.current.startAdding) : null

  useEffect(() => {
    let alive = true
    api<any>('/api/devices').then(doc => {
      if (!alive) return
      const m = doc?.managed?.[peerId]
      if (!m) { setNotManaged(true); setRows([]) }
      else {
        const base = rowsFromConfig(m.cameras ?? {})
        // The affordances the sheet was opened FROM (a tile's gear, an add button)
        // are applied once, on the config as loaded.
        setRows(opened.current.startAdding
          ? [...base, { name: '', indexOrPath: '', fps: '', width: '', height: '' }]
          : base)
      }
      setDetected((doc?.cameras ?? []).map((c: any) => ({
        index: c.index, label: c.label ?? c.name ?? null, in_use_by: c.claimed_by ?? null,
      })))
    }).catch(e => { if (alive) { setError(e?.message ?? String(e)); setRows([]) } })
    return () => { alive = false }
  }, [peerId])

  const edit = (i: number, patch: Partial<CamRow>) =>
    setRows(rs => rs!.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const remove = (i: number) => setRows(rs => rs!.filter((_, j) => j !== i))
  const add = (indexOrPath = '') =>
    setRows(rs => [...(rs ?? []), { name: '', indexOrPath, fps: '', width: '', height: '' }])

  const [probes, setProbes] = useState<Record<number, Probe>>({})
  const probe = async (idx: number) => {
    setProbes(p => ({ ...p, [idx]: { ...p[idx], busy: true, error: undefined } }))
    try {
      const r = await api<any>(`/api/devices/camera/${idx}/modes`)
      setProbes(p => ({ ...p, [idx]: { modes: r?.modes ?? [] } }))
    } catch (e: any) {
      // 409 = it is streaming for a robot right now; 404 = the running
      // dashboard predates this route. Either way the free-text fields
      // still work — the probe is a helper, not a gate.
      setProbes(p => ({
        ...p,
        [idx]: {
          error: e?.status === 404
            ? 'this dashboard process predates mode probing — type the values by hand'
            : (e?.message ?? String(e)),
        },
      }))
    }
  }

  const check = rows ? configFromRows(rows) : { cameras: null }

  const apply = async () => {
    if (!rows) return
    setBusy(true); setError(null)
    try {
      forgetJointFailure(peerId)
      const r = await post<any>(`/api/devices/${encodeURIComponent(peerId)}/cameras`, { cameras: check.cameras })
      if (r?.error) setError(r.error)
      // The settle rail answers running / starting / failed / gone.
      else if (r?.status === 'gone') setError(`${peerId} despawned during the respawn window — check devices › logs`)
      else setDone(r?.status === 'running'
        ? `✓ ${peerId} is back on the mesh with the new cameras`
        : `respawned, not yet announced on the mesh (status: ${r?.status ?? 'starting'}) — watch its card, or check devices › logs if it stays away`)
    } catch (e: any) {
      // A 404 here has one honest meaning on this codebase: the running
      // dashboard process predates this route.
      setError(e?.status === 404 && /Not Found|no such|unknown managed/i.test(e?.message ?? '')
        ? (e?.message?.includes('unknown managed')
          ? e.message
          : 'this dashboard process predates the camera-reconfigure rail — it needs a (terminal-started) restart to pick it up')
        : e?.message ?? String(e))
    } finally {
      setBusy(false); setConfirming(false)
    }
  }

  const free = detected.filter(d => !d.in_use_by || d.in_use_by === peerId)

  return (
    <div className="sheet-backdrop" onClick={busy ? undefined : onClose}>
      <div className="sheet" onClick={e => e.stopPropagation()}>
        <h3>cameras — {peerId}</h3>
        {rows === null && <p className="hint">reading the current config…</p>}
        {notManaged && (
          <p className="hint">
            This robot was not spawned by this dashboard, so its process (and its cameras)
            belong to whoever started it — change the config where it runs.
          </p>
        )}
        {rows !== null && !notManaged && !done && (
          <>
            {rows.length === 0 && <p className="hint">No cameras attached. Add one below.</p>}
            {rows.map((r, i) => {
              const iop = parseIndexOrPath(r.indexOrPath)
              const idx = typeof iop === 'number' ? iop : null
              const pr = idx !== null ? probes[idx] : undefined
              return (
                <div key={i}>
                  <div className="cam-config-row">
                    <input placeholder="name (top / wrist)" aria-label={`camera ${i + 1} name`} value={r.name}
                           autoFocus={focus?.index === i && focus.field === 'name'}
                           onChange={e => edit(i, { name: e.target.value })} />
                    <input placeholder="index or path" aria-label={`camera ${i + 1} index or path`} value={r.indexOrPath} onChange={e => edit(i, { indexOrPath: e.target.value })} />
                    <input placeholder="fps" aria-label={`camera ${i + 1} fps`} inputMode="numeric" value={r.fps}
                           autoFocus={focus?.index === i && focus.field === 'fps'}
                           onChange={e => edit(i, { fps: e.target.value })} />
                    <input placeholder="width" aria-label={`camera ${i + 1} width`} inputMode="numeric" value={r.width} onChange={e => edit(i, { width: e.target.value })} />
                    <input placeholder="height" aria-label={`camera ${i + 1} height`} inputMode="numeric" value={r.height} onChange={e => edit(i, { height: e.target.value })} />
                    <button className="btn ghost" title="detach this camera" onClick={() => remove(i)}>detach</button>
                  </div>
                  {idx !== null && (
                    <div className="cam-config-modes">
                      {!pr?.modes && (
                        <button className="btn ghost" disabled={pr?.busy} onClick={() => probe(idx)}
                                title="set + read back each candidate on the device — offers only what the camera agreed to">
                          {pr?.busy ? 'asking the camera…' : 'real modes'}
                        </button>
                      )}
                      {pr?.error && <span className="hint">⚠ {pr.error}</span>}
                      {pr?.modes && pr.modes.length === 0 && (
                        <span className="hint">the camera verified no modes — the driver's defaults still work</span>
                      )}
                      {pr?.modes?.map(m => (
                        <button key={`${m.width}x${m.height}@${m.fps}`} className="btn ghost"
                                title="fill fps/size with a mode this camera verified"
                                onClick={() => edit(i, { fps: String(m.fps), width: String(m.width), height: String(m.height) })}>
                          {m.width}×{m.height} @ {m.fps}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
            <div className="cam-config-add">
              <button className="btn ghost" onClick={() => add()}>+ add camera</button>
              {free.map(d => (
                <button key={d.index} className="btn ghost" title={d.label ?? undefined}
                        onClick={() => add(String(d.index))}>
                  + #{d.index}{d.label ? ` ${d.label}` : ''}
                </button>
              ))}
            </div>
            <p className="hint">
              Blank fps/size = the driver's defaults (640×480 @ 30). Detaching all cameras is allowed —
              the robot streams joints only.
            </p>
            {/* The fps field is the CAMERA's capture rate; what this dashboard receives is the mesh publish rate. */}
            {rateNote && <p className="hint">⏱ {rateNote}</p>}
            {/* a sheet is focus-trapped, so a refusal that renders silently reads as "the button did nothing" — the exact defect EstopSheet fixed with role=alert on its failure headline. */}
            {check.error && <div className="result bad" role="alert">✗ {check.error}</div>}
            {error && <div className="result bad" role="alert">⚠ {error}</div>}
            {!confirming ? (
              <div className="sheet-actions">
                <button className="btn go" disabled={!!check.error || busy} onClick={() => setConfirming(true)}>
                  apply…
                </button>
                <button className="btn ghost" onClick={onClose} disabled={busy}>cancel</button>
              </div>
            ) : (
              <>
                <p className="hint">{applySummary(rows, peerId)}</p>
                <div className="sheet-actions">
                  <button className="btn danger" onClick={apply} disabled={busy}>
                    {busy ? 'restarting…' : 'restart with these cameras'}
                  </button>
                  <button className="btn ghost" onClick={() => setConfirming(false)} disabled={busy}>back</button>
                </div>
              </>
            )}
          </>
        )}
        {done && (
          <>
            {/* The success half announces too, politely: the respawn it reports takes seconds, and an operator who cannot see the sheet has no other signal that it finished. */}
            <div className="result ok" role="status">{done}</div>
            <div className="sheet-actions">
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}
        {notManaged && (
          <div className="sheet-actions">
            <button className="btn ghost" onClick={onClose}>close</button>
          </div>
        )}
      </div>
    </div>
  )
}
