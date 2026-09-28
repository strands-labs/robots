import { useCallback, useEffect, useRef, useState } from 'react'
import { api, apiUrl, post, del, wsUrl, HttpError } from '../lib/endpoints'
import { makeTwin, type TwinInstance } from '../lib/twinLoader'
import type { SimLockout } from './EstopSheet'

/**
 * The Sim tab: this dashboard's own simulated robots (routes_sim.py).
 *
 *   GET  /api/sim                      list sessions          POST /api/sim {robot, mirror?}   start one
 *   GET  /api/sim/{sid}                snapshot               DELETE /api/sim/{sid}            stop it
 *   POST /api/sim/{sid}/joints         {positions}            POST /api/sim/{sid}/reset        home pose
 *   GET  /api/sim/{sid}/stream.mjpg    rendered camera        GET  /api/sim/ports              servo buses to mirror
 *   WS   /ws/telemetry/{sid}?poses=1   JSON snapshot at 15 Hz, then one binary poses frame the twin draws
 *   GET  /api/safety                   the lockout; every motion route answers 423 while it is `locked`
 *
 * Nothing here is invented client-side: a session card shows what the socket
 * last said, the sliders send targets and let the snapshot report where the
 * joints actually went, and the lockout line is only ever written from a server answer.
 */

interface Snapshot {
  id: string
  robot: string
  state: string
  sim_time: number
  steps: number
  joint_names: string[]
  qpos: number[]
  fps: number
  cameras: string[]
  error: string | null
  model_path?: string | null
  created?: number
  source?: string
  bus?: { hz?: number; age_ms?: number | null; error?: string | null } | null
  lockout?: SimLockout
}

interface SimRobot { name: string; label: string }

async function simRobots(): Promise<SimRobot[]> {
  // main's fleet.py: registry rows with has_sim/model_local. The mesh lane's fleet
  // route answers peers instead; then the branch's registry route is the list.
  try {
    const f = await api<{ robots?: any[] }>('/api/fleet?mode=sim')
    const rows = (f.robots ?? []).filter(r => r && typeof r.name === 'string' && (r.has_sim ?? true))
    if (rows.length) return rows.map(r => ({ name: r.name, label: `${r.name}${r.joints ? ` · ${r.joints} dof` : ''}${r.model_local === false ? ' · asset not downloaded' : ''}` }))
  } catch { /* fall through */ }
  const reg = await api<{ robots?: any[] }>('/api/robots/registry')
  return (reg.robots ?? []).filter(r => r && r.has_sim).map(r => ({ name: r.name, label: `${r.name}${r.joints ? ` · ${r.joints} dof` : ''}` }))
}

function lockoutText(l: SimLockout | null): string {
  if (!l) return 'lockout not read yet'
  if (l.state === 'locked') return `e-stop engaged · ${l.by || 'dashboard'} · ${l.reason ?? ''}`
  if (l.state === 'unknown') return `lockout unknown · ${l.reason ?? ''}`
  return `clear · ${l.reason ?? ''}`
}

export default function SimTab({ onClose }: { onClose: () => void }) {
  const [robots, setRobots] = useState<SimRobot[]>([])
  const [robot, setRobot] = useState('so101')
  const [ports, setPorts] = useState<{ port: string; likely_servo_bus?: boolean }[]>([])
  const [mirror, setMirror] = useState('')
  const [sessions, setSessions] = useState<Snapshot[]>([])
  const [lockout, setLockout] = useState<SimLockout | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const readLockout = useCallback(async () => {
    try { setLockout((await api<{ lockout: SimLockout }>('/api/safety')).lockout) } catch { /* the line keeps its last server answer */ }
  }, [])

  /** A failed request is a message, not an e-stop: the line is re-read, never painted. */
  const failed = useCallback(async (e: any) => {
    setMsg(e instanceof HttpError && e.status === 423 ? `refused: ${e.message}` : (e?.message ?? String(e)))
    await readLockout()
  }, [readLockout])

  const refresh = useCallback(async () => {
    try {
      const { sessions } = await api<{ sessions: Snapshot[] }>('/api/sim')
      setSessions(sessions)
    } catch (e) { await failed(e) }
    await readLockout()
  }, [failed, readLockout])

  useEffect(() => {
    void refresh()
    simRobots().then(list => {
      setRobots(list)
      if (list.length && !list.some(r => r.name === 'so101')) setRobot(list[0].name)
    }).catch(e => setMsg(`registry: ${e?.message ?? e}`))
    api<{ ports: { port: string; likely_servo_bus?: boolean }[] }>('/api/sim/ports').then(r => setPorts(r.ports ?? [])).catch(() => setPorts([]))
  }, [refresh])

  const start = async (e: React.FormEvent) => {
    e.preventDefault()
    setStarting(true); setMsg(null)
    try {
      const body: Record<string, any> = { robot }
      if (mirror) body.mirror = { port: mirror }
      await post('/api/sim', body)
      await refresh()
    } catch (err) { await failed(err) } finally { setStarting(false) }
  }

  const toggleEstop = async () => {
    const action = lockout?.state === 'locked' ? 'resume' : 'estop'
    try {
      setLockout((await post<{ lockout: SimLockout }>(`/api/safety/${action}`)).lockout)
      setMsg(null)
    } catch (e) { await failed(e) }
  }

  return (
    <div className="train-sheet sim-sheet" role="dialog" aria-label="Simulation">
      <div className="train-head">
        <h2>🧊 Simulation</h2>
        <button className="dock-min" onClick={onClose} aria-label="close simulation" title="Escape">✕</button>
      </div>

      <form className="train-form sim-new" onSubmit={start}>
        <div className="train-row">
          <label className="field"><span>robot</span>
            <select value={robot} onChange={e => setRobot(e.target.value)} disabled={starting}>
              {!robots.some(r => r.name === robot) && <option value={robot}>{robot}</option>}
              {robots.map(r => <option key={r.name} value={r.name}>{r.label}</option>)}
            </select>
          </label>
          <label className="field"><span>source</span>
            <select value={mirror} onChange={e => setMirror(e.target.value)} disabled={starting}>
              <option value="">physics (steps in this process)</option>
              {ports.map(p => (
                <option key={p.port} value={p.port}>
                  mirror {p.port.replace(/^\/dev\//, '')}{p.likely_servo_bus ? ' · servo bus' : ''}
                </option>
              ))}
            </select>
          </label>
          <div className="train-actions sim-actions">
            <button className="btn go" type="submit" disabled={starting || lockout?.state === 'locked'}>
              {starting ? 'starting…' : 'start'}
            </button>
          </div>
        </div>
        <div className={`sim-lockout ${lockout?.state ?? 'unknown'}`} role="status">
          <span>{lockoutText(lockout)}</span>
          <button type="button" className={lockout?.state === 'locked' ? 'btn go' : 'btn danger'} onClick={toggleEstop}>
            {lockout?.state === 'locked' ? 'RESUME' : 'E-STOP (sim)'}
          </button>
        </div>
        {msg && <div className="train-msg warn" role="alert">{msg}</div>}
        <p className="hint">
          A mirror session reads the servo bus at that port and never writes it: the arm decides the
          pose, the twin follows. A physics session accepts joint targets from the sliders below and
          from the agent (which asks first).
        </p>
      </form>

      <div className="sim-sessions">
        {sessions.length === 0 && (
          <p className="hint">No session yet. Pick a robot and press start - it steps in this process and streams here.</p>
        )}
        {sessions.map(s => (
          <SessionCard key={s.id} initial={s} onLockout={setLockout} onGone={refresh} onFailed={failed} />
        ))}
      </div>
    </div>
  )
}

function SessionCard({ initial, onLockout, onGone, onFailed }: {
  initial: Snapshot
  onLockout: (l: SimLockout) => void
  onGone: () => void
  onFailed: (e: any) => Promise<void>
}) {
  const [snap, setSnap] = useState<Snapshot>(initial)
  const [view, setView] = useState<'twin' | 'cam'>('twin')
  const [twinErr, setTwinErr] = useState<string | null>(null)
  // Slider targets the operator is dragging; absent = follow the snapshot.
  const [targets, setTargets] = useState<Record<string, number>>({})
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const twinRef = useRef<TwinInstance | null>(null)
  const sendTimer = useRef<number | null>(null)
  const pending = useRef<Record<string, number>>({})
  const mirror = !!snap.source && snap.source.startsWith('real:')
  const live = snap.state !== 'stopped' && snap.state !== 'error'

  // The twin: built once per card, fed by the same socket that feeds the strip.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let disposed = false
    makeTwin(canvas, initial.id).then(t => {
      if (disposed) { t.dispose(); return }
      twinRef.current = t
      t.load().catch(e => setTwinErr(`twin: ${e?.message ?? e}`))
    }).catch(e => setTwinErr(`twin unavailable: ${e?.message ?? e}`))
    return () => { disposed = true; twinRef.current?.dispose(); twinRef.current = null }
  }, [initial.id])

  useEffect(() => {
    const ws = new WebSocket(wsUrl(`/ws/telemetry/${initial.id}?poses=1`))
    ws.binaryType = 'arraybuffer'
    ws.onmessage = ev => {
      if (ev.data instanceof ArrayBuffer) { twinRef.current?.poses(ev.data); return }
      let m: Snapshot
      try { m = JSON.parse(ev.data) } catch { return }
      setSnap(m)
      if (m.lockout) onLockout(m.lockout)
      if (m.state === 'stopped' || m.state === 'error') ws.close()
    }
    ws.onclose = ev => { if (ev.code === 4404) void onGone() }
    return () => ws.close()
  }, [initial.id, onLockout, onGone])

  /** Sliders coalesce into one POST per 80 ms: the route is a target, not a stream. */
  const setJoint = (name: string, value: number) => {
    setTargets(t => ({ ...t, [name]: value }))
    pending.current[name] = value
    if (sendTimer.current != null) return
    sendTimer.current = window.setTimeout(async () => {
      sendTimer.current = null
      const positions = pending.current
      pending.current = {}
      try { await post(`/api/sim/${initial.id}/joints`, { positions }) }
      catch (e) { setTargets({}); await onFailed(e) }
    }, 80)
  }

  const reset = async () => {
    setTargets({})
    try { await post(`/api/sim/${initial.id}/reset`) } catch (e) { await onFailed(e) }
  }
  const stop = async () => {
    try { await del(`/api/sim/${initial.id}`) } catch (e) { await onFailed(e) }
    await onGone()
  }

  const camSrc = view === 'cam' ? apiUrl(`/api/sim/${initial.id}/stream.mjpg`) : ''
  const busLine = snap.bus
    ? (snap.error || snap.bus.error || (snap.bus.age_ms == null ? 'waiting for the bus' : `read ${snap.bus.age_ms} ms ago · torque untouched`))
    : snap.error

  return (
    <article className={`sim-session ${snap.state}`}>
      <div className="sim-head">
        <b>{snap.robot}</b>
        <span className="badge mono">{snap.id}</span>
        <span className={`badge ${snap.state === 'frozen' || snap.state === 'error' ? 'danger' : snap.state === 'running' ? '' : 'warn'}`}>{snap.state}</span>
        {mirror && <span className="badge warn" title={`Reads the servo bus at ${snap.source!.slice(5)}; never writes it`}>mirror · read-only</span>}
        <span className="spacer" />
        <div className="viewsel" role="tablist">
          <button type="button" role="tab" className={`chip${view === 'twin' ? ' on' : ''}`} aria-selected={view === 'twin'} onClick={() => setView('twin')}>Twin</button>
          <button type="button" role="tab" className={`chip${view === 'cam' ? ' on' : ''}`} aria-selected={view === 'cam'} onClick={() => setView('cam')}>Camera</button>
        </div>
      </div>
      <div className="sim-view">
        <canvas ref={canvasRef} className="twin" hidden={view !== 'twin'} />
        {/* only stream while shown: an MJPG socket per hidden card is a renderer nobody watches */}
        {view === 'cam' && <img className="cam" alt={`${snap.robot} camera`} src={camSrc} />}
        {twinErr && view === 'twin' && <div className="hint warn sim-overlay">{twinErr}</div>}
      </div>
      <div className="sim-joints">
        {snap.joint_names.map((name, i) => {
          const q = snap.qpos[i] ?? 0
          const v = targets[name] ?? q
          return (
            <label key={name} className="sim-joint">
              <span className="label">{name}</span>
              <input type="range" min={-Math.PI} max={Math.PI} step={0.005} value={v}
                     disabled={mirror || !live}
                     aria-label={`${name} target, radians`}
                     onChange={e => setJoint(name, Number(e.target.value))} />
              <span className="val mono">{q.toFixed(3)}</span>
            </label>
          )
        })}
      </div>
      <div className="sim-foot">
        <span className="mono">{snap.bus ? `${snap.bus.hz ?? 0} Hz bus` : `t=${snap.sim_time.toFixed(2)}s`}</span>
        {snap.fps ? <span className="mono">{snap.fps} fps</span> : null}
        {busLine && <span className="hint" title={busLine}>{busLine}</span>}
        <span className="spacer" />
        {!mirror && <button type="button" className="btn ghost" onClick={reset} disabled={!live}>reset</button>}
        <button type="button" className="btn ghost danger" onClick={stop}>stop</button>
      </div>
    </article>
  )
}
