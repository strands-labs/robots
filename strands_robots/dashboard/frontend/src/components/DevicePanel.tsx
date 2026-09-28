import { useCallback, useEffect, useRef, useState } from 'react'
import { boardListEmptyLine, managedListEmptyLine, hardwareSummaryValue } from '../lib/boardList'
import { deathVerdict, retainedOutputIsStartup } from '../lib/childDeath'
import { useDialogFocus } from '../lib/useDialogFocus'
import { numField } from '../lib/numField'
import { type CameraRow, camerasField } from '../lib/cameraRows'
import { findConsent, type ConsentNeed } from '../lib/consent'
import { spawnNotice, type SpawnNotice } from '../lib/spawnNotice'
import ConsentSheet from './ConsentSheet'
import { api, post, HttpError } from '../lib/endpoints'
import { deviceActionFailure, type DeviceAction } from '../lib/deviceOutcome'
import CalibrationSection from './CalibrationSection'
import CameraGallery, { type CameraInfo, type CameraName, type CameraProblem } from './CameraGallery'
import { normalizeRegistry, type RegistryRobot } from '../lib/registry'
import { calibratePlan, knownCalibrationId, type SpawnProfile } from '../lib/calibrateCommand'
import CalibrateWizard from './CalibrateWizard'
import { rememberedLine } from '../lib/rememberedBoard'
import { parseCalibrationList, type CalibrationEntry } from '../lib/calibration'
import { calibrationVerdict } from '../lib/calibrationMatch'
import { rescanReport, hardwareKey, type RescanReport } from '../lib/rescanReport'
import { isLatestRequest } from '../lib/requestOrder'
import { peerNameField } from '../lib/peerName'
import { portChoice, blocksSpawn } from '../lib/portChoice'
import { safeFilename, snippetRefusal, hubAddressMissing, cleanHubHost } from '../lib/deploySnippet'
import { forgetJointFailure } from '../lib/useJointFailure'

interface SerialPort {
  device: string
  description?: string
  vid?: string | null
  pid?: string | null
  serial_number?: string | null
  likely_robot?: string | null
  role?: string | null
  role_volts?: number | null
  role_source?: string | null
  role_measured_at?: number | null
  remembered?: {
    peer_id: string
    robot_name?: string | null
    mode?: string | null
    cameras: string[]
    robot_id?: string
    saved_at?: number | null
    camera_health?: { ok: boolean; text: string; cameras: { name: string; index?: number; state: string; reason?: string; remedy?: string }[] } | null
  } | null
}

interface RoleVerdict {
  role: string
  reason: string
  volts?: number | null
  remedy?: string
  remembered?: boolean
  remember_problem?: string
  mismatch?: { labelled: string, measured: string, message: string, remedy: string } | null
}

interface DeviceDoc {
  serial_ports: SerialPort[]
  cameras: CameraInfo[]
  camera_names?: CameraName[]
  camera_problem?: CameraProblem | null
  managed: Record<string, Managed>
}

interface Managed {
  peer_id: string
  robot_name: string
  mode: string
  port?: string | null
  alive: boolean
  started_at?: number
  log_tail?: string[]
  returncode?: number | null
}

/** Local hardware, and the robot processes this dashboard owns. */
export default function DevicePanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [doc, setDoc] = useState<DeviceDoc | null>(null)
  const sheetRef = useRef<HTMLElement | null>(null)
  useDialogFocus(sheetRef, open)
  const [robots, setRobots] = useState<RegistryRobot[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [snip, setSnip] = useState<{ device: string; code: string; filename: string } | null>(null)
  const [logs, setLogs] = useState<{ peer: string; lines: string[] } | null>(null)
  const [consent, setConsent] = useState<ConsentNeed | null>(null)
  const [notice, setNotice] = useState<SpawnNotice | null>(null)
  const retry = useRef<{ fn: () => Promise<any>; label: string; kind: DeviceAction } | null>(null)

  // spawn form
  const [robotName, setRobotName] = useState('')
  const [mode, setMode] = useState<'sim' | 'real'>('sim')
  const [port, setPort] = useState('')
  const [camRows, setCamRows] = useState<CameraRow[]>([{ name: 'main', index: '' }])
  const [camFps, setCamFps] = useState('')
  const [camW, setCamW] = useState('')
  const [camH, setCamH] = useState('')
  const [robotId, setRobotId] = useState('')
  const [peerName, setPeerName] = useState('')
  // The real calibration ids on this machine. `null` = not read yet / failed,
  // which the verdict deliberately treats as "say nothing" rather than a guess.
  const [calibIds, setCalibIds] = useState<CalibrationEntry[] | null>(null)
  /** The remembered spawn profiles, keyed by serial. */
  const [profiles, setProfiles] = useState<Record<string, SpawnProfile> | null>(null)
  const [roles, setRoles] = useState<Record<string, RoleVerdict>>({})
  const [measuring, setMeasuring] = useState<string | null>(null)
  // R5: which port's calibrate panel is open, the family the operator confirmed
  // for it, and the last copy attempt's verdict (copying can genuinely fail —
  // see copyCommand).
  const [hubDraft, setHubDraft] = useState('')
  const [calibFor, setCalibFor] = useState<string | null>(null)
  const [calibFamily, setCalibFamily] = useState<Record<string, string>>({})
  const [copied, setCopied] = useState<string | null>(null)
  // The verdict for the scan the operator ASKED for (rescan), plus the in-flight flag: a
  // serial+camera enumeration takes seconds, and a button that still looks idle invites a second
  // click that stacks another scan on the first.
  const [scan, setScan] = useState<RescanReport | null>(null)
  const [scanning, setScanning] = useState(false)
  const scannedAt = useRef<number | null>(null)
  // What the last verdict described. A background poll that finds different
  // hardware retires it: a verdict must never outlive its evidence.
  const scanKey = useRef<string | null>(null)
  // A rescan (`?refresh=1`) re-probes serial AND every camera index, which takes seconds, while
  // the 5s background poll reads the CACHED enumeration.
  const loadSeq = useRef(0)
  const inFlight = useRef(0)

  const load = useCallback(async (refresh = false) => {
    const mine = ++loadSeq.current
    inFlight.current += 1
    try {
      const next = await api<DeviceDoc>(`/api/devices${refresh ? '?refresh=1' : ''}`)
      // A newer load owns the screen. The caller still receives this doc, so a
      // rescan can see what its OWN scan found, but it must not paint it.
      if (!isLatestRequest(mine, loadSeq.current)) {
        return { ok: true as const, after: next, superseded: true }
      }
      setDoc(next)
      scannedAt.current = Date.now()
      setError(null)
      // A poll that changes the hardware on screen invalidates the last rescan
      // verdict ("unchanged: 2 serial ports" next to one port is a new lie).
      if (!refresh && scanKey.current !== null && hardwareKey(next) !== scanKey.current) {
        scanKey.current = null
        setScan(null)
      }
      return { ok: true as const, after: next }
    } catch (e: any) {
      const msg = e?.message ?? String(e)
      // On a rescan the failure is reported BY the verdict (which also says the
      // visible list is now stale); a second red line would say half of it.
      if (!refresh) setError(msg)
      if (!isLatestRequest(mine, loadSeq.current)) {
        return { ok: false as const, error: msg, superseded: true }
      }
      return { ok: false as const, error: msg }
    } finally {
      inFlight.current -= 1
    }
  }, [])

  const rescan = useCallback(async () => {
    if (scanning) return
    setScanning(true)
    const before = doc
    const beforeAtMs = scannedAt.current
    const outcome = await load(true)
    // Superseded by a newer load: the rows on screen are not this scan's result,
    // so a verdict about them would be a claim about a list nobody is looking at.
    if ((outcome as { superseded?: boolean }).superseded) { setScanning(false); return }
    const verdict = rescanReport(before, outcome, { beforeAtMs, nowMs: Date.now() })
    scanKey.current = outcome.ok ? hardwareKey(outcome.after) : null
    setScan(verdict)
    setError(null)
    setScanning(false)
  }, [doc, load, scanning])

  useEffect(() => {
    if (!open) return
    void load()
    void api<{ robots: any }>('/api/robots/registry')
      .then(r => setRobots(normalizeRegistry(r.robots)))
      .catch(() => setRobots([]))
    // Managed children die on their own (import errors, unplugged bus); poll so an `alive: false`
    // row does not sit there looking healthy.
    const id = setInterval(() => { if (inFlight.current === 0) void load() }, 5000)
    return () => clearInterval(id)
  }, [open, load])

  // Read the calibration ids once the panel opens: the spawn form's id field is checked against
  // them, so a typo is caught before an arm runs on raw servo counts.
  useEffect(() => {
    if (!open) return
    let alive = true
    api<{ status?: string; text?: string }>('/api/calibration')
      .then(r => { if (alive && r?.text) setCalibIds(parseCalibrationList(r.text).entries) })
      .catch(() => { /* unchecked is a state the verdict handles */ })
    api<{ profiles?: Record<string, SpawnProfile> }>('/api/devices/profiles')
      .then(r => { if (alive) setProfiles(r?.profiles ?? {}) })
      .catch(() => { /* stays null: the command falls back to an invented id and says so */ })
    return () => { alive = false }
  }, [open])

  if (!open) return null

  /** Every mutating device action. */
  const act = async (fn: () => Promise<any>, label: string, kind: DeviceAction = 'spawn') => {
    setBusy(true); setStatus(null); setConsent(null); setNotice(null)
    retry.current = { fn, label, kind }
    try {
      const r = await fn()
      setStatus(r?.error ? `⚠ ${r.error}` : `${label}: ${r?.peer_id ?? 'ok'}`)
      // 200-with-error is how a settled-then-dead spawn reports itself; the
      // consent hint rides in that same body.
      setConsent(findConsent(r))
      // A spawn can SUCCEED and still be about to fail on the wire: the calibration gap arrives in
      // the same 200 body as the pid.
      setNotice(spawnNotice(r))
      await load()
    } catch (e: any) {
      const v = deviceActionFailure({
        kind,
        status: e instanceof HttpError ? e.status : 0,
        message: e?.message ?? String(e),
      })
      setStatus(v.text)
      setConsent(findConsent(e?.body))
      setNotice(spawnNotice(e?.body))
      if (v.ambiguous) await load()
    } finally {
      setBusy(false)
    }
  }

  const camNums = {
    fps: camFps === '' ? null : numField(camFps, { what: 'fps', min: 1, max: 240 }),
    width: camW === '' ? null : numField(camW, { what: 'pixels wide', min: 64, max: 7680 }),
    height: camH === '' ? null : numField(camH, { what: 'pixels high', min: 64, max: 4320 }),
  }
  // The composed rows judged into the spawn payload — or a refusal in words, before the button.
  const camField = camerasField(camRows, {
    fps: camNums.fps && !camNums.fps.problem ? camNums.fps.value : null,
    width: camNums.width && !camNums.width.problem ? camNums.width.value : null,
    height: camNums.height && !camNums.height.problem ? camNums.height.value : null,
  })
  const camProblem =
    [camNums.fps, camNums.width, camNums.height].find(v => v?.problem)?.problem ?? camField.problem
  // A correction we make must be admitted, not just the refusals: 12.5 fps becomes 12, and the
  // operator who typed 12.5 would otherwise never learn which of the two the camera was given.
  const camNote = [
    ...[camNums.fps, camNums.width, camNums.height].map(v => v?.note),
    camField.note, // shared-index warning: sendable, but said out loud
  ].filter(Boolean).join(' · ') || null
  const anyCam = camRows.some(r => r.index !== '')

  /* Judged against BOTH the live children and the remembered profiles: a name that collides with
     either is the 409 the server would answer, and it costs nothing to say so before the button. */
  const nameVerdict = peerNameField(peerName, {
    existing: [
      ...Object.keys(doc?.managed ?? {}),
      ...Object.values(profiles ?? {}).map(p => (p as any)?.peer_id).filter(Boolean),
    ],
    robotName,
    mode,
  })

  /**
   * Spawning under a name that was mute before: the previous process's excuse is not this one's,
   * so the cached "why has this arm no joints" verdict is dropped BEFORE the request — the new
   * peer can announce itself before this promise resolves, and its card must not greet it with
   * the old reason.
   */
  const spawn = () => act(() => (forgetJointFailure(nameVerdict.value), post('/api/devices/spawn', {
    robot_name: robotName,
    peer_id: nameVerdict.value,
    mode,
    port: mode === 'real' ? port || null : null,
    // The camera config must be a MAPPING per entry ({index_or_path: N, ...}); a bare int here is
    // the exact ValueError an operator once hit live: "Camera 'main' config must be a mapping ...
    // got int: 3". camerasField owns that shape (and every name/index collision) in one place.
    cameras: camField.problem ? null : camField.value,
    robot_id: robotId || null,
  })), 'spawned')

  const showLogs = async (peer: string) => {
    try {
      const r = await api<{ lines: string[] }>(`/api/devices/logs/${encodeURIComponent(peer)}`)
      setLogs({ peer, lines: r.lines ?? [] })
    } catch (e: any) {
      setLogs({ peer, lines: [`⚠ ${e?.message ?? String(e)}`] })
    }
  }

  /** Bring a remembered board back up. */
  const writeSnippet = async (p: SerialPort, hubHost?: string) => {
    setSnip(null); setNotice(null)
    try {
      const r = await post<{ snippet: string; filename?: string; peer_id?: string }>(
        '/api/deploy/snippet', { serial: p.serial_number, ...(hubHost ? { hub_host: hubHost } : {}) })
      setSnip({ device: p.device, code: r.snippet, filename: safeFilename(r.filename, r.peer_id) })
    } catch (e) {
      const status = e instanceof HttpError ? e.status : 0
      setNotice({ text: `✗ ${snippetRefusal(status, e instanceof Error ? e.message : String(e))}` } as SpawnNotice)
    }
  }

  const respawnRemembered = (p: SerialPort) =>
    act(() => post('/api/devices/spawn-remembered', { port: p.device }),
        `spawned ${p.remembered?.peer_id ?? 'it'} from its saved profile`)

  const measureRole = async (port: string) => {
    setMeasuring(port)
    try {
      const v = await api<RoleVerdict>(`/api/devices/arm-role?port=${encodeURIComponent(port)}`)
      setRoles(r => ({ ...r, [port]: v }))
      // The verdict is remembered server-side against the board's serial, so
      // reload to pick up the badge rather than mirroring it in two places.
      if (v.remembered) void load(false)
    } catch (e: any) {
      setRoles(r => ({
        ...r,
        [port]: { role: 'unknown', reason: e?.message ?? String(e) },
      }))
    } finally {
      setMeasuring(null)
    }
  }

  const managed = Object.values(doc?.managed ?? {})
  const freePorts = doc?.serial_ports ?? []
  const claimedPorts = new Set(managed.filter(m => m.alive && m.port).map(m => m.port as string))

  const portVerdict = portChoice({
    chosen: port,
    known: freePorts.map(p => p.device),
    claimed: [...claimedPorts],
    scanned: doc !== null,
  })

  // The servo-board rows shadow `busy` with a per-row "this bus is claimed" flag, so an action in
  // flight has to be captured under its own name or every button in that list stays live during it.
  const acting = busy

  /**
   * Which robot family this board is, and HOW we know — the model name in the calibrate command
   * is half family, so its provenance has to be visible.
   */
  const familyFor = (p: SerialPort): { family: string; source: string } => {
    const running = managed.find(m => m.alive && m.port === p.device)
    if (running?.robot_name) return { family: running.robot_name, source: 'the arm running on this port' }
    const picked = (calibFamily[p.device] ?? '').trim()
    if (picked) return { family: picked, source: 'your pick' }
    if (robotName.trim()) return { family: robotName.trim(), source: 'the robot selected in the spawn form above' }
    if (p.likely_robot) return { family: p.likely_robot, source: 'a guess from the USB id — confirm it' }
    return { family: '', source: '' }
  }

  /** Copy, and say so honestly when it fails. */
  const copyCommand = async (port: string, command: string) => {
    try {
      if (!navigator.clipboard) throw new Error('this page is not a secure origin, so the browser blocks copying')
      await navigator.clipboard.writeText(command)
      setCopied(`${port}\u0000ok`)
    } catch (e: any) {
      setCopied(`${port}\u0000${e?.message ?? String(e)}`)
    }
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside ref={sheetRef} className="drawer wide" onClick={e => e.stopPropagation()}>
        <header className="drawer-head">
          <h2>Devices</h2>
          <div>
            <button className="btn ghost" onClick={() => void rescan()} disabled={busy || scanning}>
              {scanning ? 'scanning…' : 'rescan'}
            </button>
            <button className="btn ghost" onClick={onClose} aria-label="close devices" title="Escape">✕</button>
          </div>
        </header>

        <div className="drawer-body">
          {error && <div className="result bad">⚠ {error}</div>}
          {scan && (
            <div className={scan.tone === 'bad' ? 'result bad' : scan.tone === 'warn' ? 'result warn' : 'result ok'}>
              {scan.text}
            </div>
          )}
          {/* The child started; this says why the arm may still show no joints. */}
          {notice && (
            <div className="result warn" role="status">
              ⚠ {notice.text}
              <button className="btn ghost" style={{ marginLeft: 8 }} onClick={() => setNotice(null)}>
                dismiss
              </button>
            </div>
          )}
          {consent && (
            <ConsentSheet
              need={consent}
              target="spawn"
              onCancel={() => setConsent(null)}
              onRetry={() => { const again = retry.current; setConsent(null); if (again) void act(again.fn, again.label, again.kind) }}
            />
          )}

          <section>
            {/* No count before the scan answers: `(0)` is a claim, and an unanswered request would make the heading agree with the empty list about a fleet that may be running. */}
            <h3>Managed robots{doc !== null ? ` (${managed.length})` : ''}</h3>
            {/* `None.` used to render from `doc?.managed ?? {}`, so a failed /api/devices reported zero children while children were running, publishing to the mesh and holding serial ports — and the (0) in the heading agreed with it. */}
            {managed.length === 0 && (() => {
              const line = managedListEmptyLine({ scanned: doc !== null, error })
              return <p className="hint" role="status">{line.message}</p>
            })()}
            <ul className="devlist">
              {managed.map(m => (
                <li key={m.peer_id} className={m.alive ? '' : 'dead'}>
                  <span className={m.alive ? 'dot on' : 'dot off'} role="img"
                        aria-label={m.alive ? 'process running' : 'process exited'}
                        title={m.alive ? 'process running' : 'process exited'} />
                  <b>{m.peer_id}</b>
                  <span className="meta">
                    {m.robot_name} · {m.mode}{m.port ? ` · ${m.port}` : ''}

                  </span>
                  <span className="devactions">
                    <button className="btn ghost" onClick={() => void showLogs(m.peer_id)}>logs</button>
                    <button className="btn ghost danger" disabled={busy}
                            onClick={() => void act(() => post('/api/devices/despawn', { peer_id: m.peer_id }), 'despawned', 'despawn')}>
                      {m.alive ? 'despawn' : 'remove'}
                    </button>
                  </span>
                  {!m.alive && (() => {
                    const v = deathVerdict(m.returncode)
                    return <div className={v.unexplained ? 'deathnote unexplained' : 'deathnote'}>{v.phrase}</div>
                  })()}
                  {!m.alive && m.log_tail?.length ? (() => {
                    // The ring keeps 10 lines.
                    const startup = retainedOutputIsStartup({ lines: m.log_tail, startedAt: m.started_at })
                    return (
                      <>
                        {startup === true && (
                          <div className="meta">its startup output — it printed nothing after that, so these are not its last words</div>
                        )}
                        <pre className="logtail">{m.log_tail.slice(-6).join('\n')}</pre>
                      </>
                    )
                  })() : null}
                </li>
              ))}
            </ul>
          </section>

          {logs && (
            <section>
              <h3>
                {logs.peer} log
                <button className="btn ghost" onClick={() => setLogs(null)}>close</button>
              </h3>
              <pre className="logtail tall">{logs.lines.join('\n') || '(no output yet)'}</pre>
            </section>
          )}

          <section>
            <h3>Spawn</h3>
            <label className="field">
              <span>Name</span>
              <input value={peerName} placeholder="left-arm (optional)"
                     aria-invalid={nameVerdict.problem ? true : undefined}
                     onChange={e => setPeerName(e.target.value)} />
            </label>
            {nameVerdict.problem
              ? <p className="hint bad" role="alert">⚠ {nameVerdict.problem}{nameVerdict.suggestion && (
                  <> <button type="button" className="btn ghost tiny"
                             onClick={() => setPeerName(nameVerdict.suggestion!)}>use {nameVerdict.suggestion}</button></>
                )}</p>
              : nameVerdict.note ? <p className="hint">{nameVerdict.note}</p> : null}
            <div className="row">
              <label className="field">
                <span>Robot</span>
                <select value={robotName} onChange={e => setRobotName(e.target.value)}>
                  <option value="">select…</option>
                  {robots.map(r => <option key={r.name} value={r.name}>{r.label}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Mode</span>
                <select value={mode} onChange={e => setMode(e.target.value as 'sim' | 'real')}>
                  <option value="sim">sim (MuJoCo)</option>
                  <option value="real">real hardware</option>
                </select>
              </label>
            </div>
            {mode === 'real' && (
              <>
                <div className="row">
                  <label className="field">
                    <span>Servo bus</span>
                    <select value={port} onChange={e => setPort(e.target.value)}>
                      <option value="">select a port…</option>
                      {freePorts.map(p => (
                        <option key={p.device} value={p.device} disabled={claimedPorts.has(p.device)}>
                          {p.device}{p.likely_robot ? ` (${p.likely_robot})` : ''}
                          {claimedPorts.has(p.device) ? ' — in use' : ''}
                        </option>
                      ))}
                    </select>
                    {/* said where the choice was made, instead of arriving from a serial driver inside a child process minutes later. */}
                    {(portVerdict.kind === 'vanished' || portVerdict.kind === 'claimed') && (
                      <em className="field-err" role="alert">
                        ⚠ {portVerdict.detail} — {portVerdict.remedy}
                      </em>
                    )}
                  </label>
                  <label className="field">
                    <span>Calibration id</span>
                    <input value={robotId} placeholder="lerobot id (optional)"
                           list="calib-ids"
                           onChange={e => setRobotId(e.target.value)} />
                    <datalist id="calib-ids">
                      {/* A SUGGESTION IS AN ENDORSEMENT: an id whose own name says the value was missing (None/null — see lib/calibration idProblem) is never offered here, because picking it would hand a real arm limits from that accident. */}
                      {(calibIds ?? []).filter(c => c.id && !c.problem).map(c => (
                        <option key={`${c.deviceType}/${c.model}/${c.id}`} value={c.id}>{c.model}</option>
                      ))}
                    </datalist>
                  </label>
                </div>
                <details className="hint">
                  <summary>why the calibration id matters</summary>
                  <p className="hint">
                    A real robot moves as soon as a task runs. The calibration id must match the one
                    used by <code>lerobot-calibrate</code>, or the joint limits will be wrong.
                  </p>
                </details>
                {(() => {
                  // The prose above was the ONLY check until now: this compares the
                  // typed id against the files that actually exist. A warning, never
                  // a block - spawning before calibrating is legitimate.
                  const v = calibrationVerdict(robotId, calibIds, robotName)
                  if (!v.note) return null
                  return (
                    <p className={v.warn ? 'hint warn' : 'hint ok'}>
                      {v.warn ? '⚠ ' : '✓ '}{v.note}
                      {v.suggestion && (
                        <> <button type="button" className="btn ghost tiny"
                                   onClick={() => setRobotId(v.suggestion!)}>use {v.suggestion}</button></>
                      )}
                    </p>
                  )
                })()}
              </>
            )}
            {/* One row per camera the robot will carry: a NAME (the lerobot config key —
                main, wrist, top…) plus a physical index. An arm with two cameras is the
                normal case here, not the exotic one. */}
            {camRows.map((row, i) => (
              <div className="row" key={i}>
                <label className="field">
                  <span>{i === 0 ? 'Camera name' : `Camera ${i + 1} name`}</span>
                  <input type="text" value={row.name} placeholder={i === 0 ? 'main' : 'wrist'}
                         onChange={e => setCamRows(rows =>
                           rows.map((r, j) => (j === i ? { ...r, name: e.target.value } : r)))} />
                </label>
                <label className="field">
                  <span>index</span>
                  <select value={row.index}
                          onChange={e => setCamRows(rows =>
                            rows.map((r, j) => (j === i ? { ...r, index: e.target.value } : r)))}>
                    <option value="">none</option>
                    {(doc?.cameras ?? []).map(c => (
                      // claimed-by-a-robot is disabled; an index another ROW picked stays
                      // selectable — that case warns below instead of blocking (the operator
                      // may know something we don't).
                      <option key={c.index} value={c.index} disabled={!!c.claimed_by}>
                        index {c.index}
                        {c.width ? ` — ${c.width}×${c.height}` : ''}
                        {c.fps ? ` @ ${c.fps}fps` : ''}
                        {c.claimed_by ? ` — claimed by ${c.claimed_by}` : ''}
                        {camRows.some((o, j) => j !== i && o.index === String(c.index)) ? ' — also picked above' : ''}
                      </option>
                    ))}
                  </select>
                </label>
                {camRows.length > 1 && (
                  <button type="button" className="btn ghost tiny" aria-label={`remove camera ${i + 1}`}
                          onClick={() => setCamRows(rows => rows.filter((_, j) => j !== i))}>
                    ✕
                  </button>
                )}
              </div>
            ))}
            {/* the next unclaimed name is offered, so add-click-add composes main → wrist → top */}
            <button type="button" className="btn ghost tiny"
                    onClick={() => setCamRows(rows => [...rows, {
                      name: ['main', 'wrist', 'top'].find(n => !rows.some(r => r.name.trim().toLowerCase() === n)) ?? '',
                      index: '',
                    }])}>
              + add camera
            </button>
            {anyCam && (
              <>
                <div className="row">
                  <label className="field">
                    <span>FPS</span>
                    <input type="number" inputMode="numeric" min={1} max={120}
                           value={camFps} placeholder="30"
                           onChange={e => setCamFps(e.target.value)} />
                  </label>
                  <label className="field">
                    <span>Width</span>
                    <input type="number" inputMode="numeric" min={64} step={2}
                           value={camW} placeholder="640"
                           onChange={e => setCamW(e.target.value)} />
                  </label>
                  <label className="field">
                    <span>Height</span>
                    <input type="number" inputMode="numeric" min={64} step={2}
                           value={camH} placeholder="480"
                           onChange={e => setCamH(e.target.value)} />
                  </label>
                </div>
                {camProblem && <p className="hint bad" role="alert">⚠ {camProblem}</p>}
                {!camProblem && camNote && <p className="hint">{camNote}</p>}
                <p className="hint">
                  Blank = the driver's defaults (640×480 @ 30). A setting the camera can't do
                  fails loudly at spawn — check the log tail, not the stream.
                </p>
              </>
            )}
            <div className="sheet-actions">
              <button className="btn go"
                      disabled={busy || !robotName || (mode === 'real' && !port) || !!camProblem || !!nameVerdict.problem
                                || (mode === 'real' && blocksSpawn(portVerdict))}
                      onClick={spawn}>
                spawn
              </button>
            </div>
          </section>

          <section>
            <h3>Cameras</h3>
            <CameraGallery cameras={doc?.cameras ?? []} names={doc?.camera_names ?? []}
                           problem={doc?.camera_problem ?? null}
                           scanned={doc !== null} error={error} />
            <details className="hint">
              <summary>why some cameras are not previewed</summary>
              <p className="hint">
                Camera indices owned by a running robot are never re-probed — opening one steals
                frames from its capture thread mid-episode.
              </p>
            </details>
          </section>

          <CalibrationSection />

          <section>
            <h3>Servo boards</h3>
            <p className="hint">leader ↔ follower is measured off the bus voltage (7.4V vs 12V), never guessed from a name.</p>
            <details className="hint">
              <summary>how the measurement works</summary>
              <p className="hint">
                A follower arm runs a 12V servo bus, a leader 7.4V — so the role can be read off the
                hardware instead of inherited from a name. The read touches one register
                (<code>Present_Voltage</code>) and cannot move the arm. A servo bus has a single owner,
                so an arm that is running must be despawned before it can be measured.
              </p>
            </details>
            <ul className="boardlist">
              {/* `no servo board detected` used to appear whenever this array was empty — including while the first scan was in flight and when it FAILED (401 through the tunnel, dead dashboard). */}
              {freePorts.length === 0 && (() => {
                const line = boardListEmptyLine({ scanned: doc !== null, error })
                return <li className="muted" role="status">{line.message}</li>
              })()}
              {freePorts.map(p => {
                const v = roles[p.device]
                const busy = claimedPorts.has(p.device)
                return (
                  <li key={p.device}>
                    <div className="row between">
                      <span className="mono">{p.device}</span>
                      {p.role
                        ? <span className={'rolebadge ' + p.role}>
                            {p.role}{p.role_volts ? ' · ' + p.role_volts + 'V' : ''}
                          </span>
                        : <span className="rolebadge unmeasured">role not measured</span>}
                    </div>
                    <div className="row between">
                      <span className="muted small">
                        {p.serial_number ? 'serial ' + p.serial_number : 'no serial number'}
                      </span>
                      <button className="btn ghost" disabled={busy || measuring === p.device}
                              title={busy
                                ? 'this arm is running and owns its bus — despawn it first'
                                : 'reads one register; cannot move the arm'}
                              onClick={() => void measureRole(p.device)}>
                        {measuring === p.device
                          ? 'reading…'
                          : busy ? 'running — despawn to measure' : 'measure role'}
                      </button>
                      <button className="btn ghost"
                              aria-expanded={calibFor === p.device}
                              title="show the exact lerobot-calibrate command for this arm — the dashboard runs nothing"
                              onClick={() => { setCopied(null); setCalibFor(calibFor === p.device ? null : p.device) }}>
                        {calibFor === p.device ? 'hide calibrate command' : 'calibrate…'}
                      </button>
                    </div>
                    {/* after a restart `managed` is empty and this board reads as unknown hardware, though its whole spawn payload is on disk. */}
                    {p.remembered && (() => {
                      // The line, and the trap inside it: the memory's id and peer name are just names someone
                      // typed, while the badge above is a MEASUREMENT.
                      const mem = rememberedLine(p.remembered, { ...p, calibrations: calibIds })!
                      return (
                      <div className="row between remembered">
                        <span className="muted small">
                          last spawned as <b>{mem.summary}</b>
                          {mem.calibrationId ? ` · calibration id ${mem.calibrationId}` : ''}
                          {mem.warning && <span className="warn small"> ⚠ {mem.warning}</span>}
                          {/* An id that cannot load is a different kind of news from a name that lies: this respawn will fail the way it failed last time. role=alert because it appears without the operator asking anything. */}
                          {mem.idProblem &&
                            <span className="warn small" role="alert"> ⚠ {mem.idProblem}</span>}
                          {/* the saved camera indices are the least stable part of the memory. */}
                          {p.remembered.camera_health?.text &&
                            <span className="warn small"> ⚠ {p.remembered.camera_health.text}</span>}
                          {(p.remembered.camera_health?.cameras ?? [])
                            .filter(c => c.remedy && c.state !== 'ready' && c.state !== 'unchecked')
                            .map(c => (
                              <span key={c.name} className="hint small"> {c.name}: {c.remedy}</span>
                            ))}
                        </span>
                        {/* reads the SAME remembered payload the respawn button uses, so the file and the button can never describe different rigs. */}
                        <button className="btn ghost" title="write the Python file that recreates this rig on another machine"
                                onClick={() => void writeSnippet(p)}>deploy .py</button>
                        <button className="btn ghost" disabled={acting || claimedPorts.has(p.device)}
                                title={claimedPorts.has(p.device)
                                  ? 'something is already running on this bus — despawn it first'
                                  : 'starts a child process with the saved payload; a real arm will be energised'}
                                onClick={() => void respawnRemembered(p)}>
                          {claimedPorts.has(p.device)
                            ? 'already running'
                            : `spawn ${p.remembered.peer_id} again`}
                        </button>
                      </div>
                      )
                    })()}
                    {snip?.device === p.device && (
                      <div className="dev-snippet">
                        <div className="row between">
                          <span className="muted small">{snip.filename} — runs this rig on another machine</span>
                          <span>
                            <button className="btn ghost tiny" onClick={() => {
                              void navigator.clipboard?.writeText(snip.code)
                              setCopied(`${p.device}\u0000copied ${snip.filename}`)
                            }}>copy</button>{' '}
                            <button className="btn ghost tiny" onClick={() => {
                              /* A data: URL would be truncated by some browsers at a few hundred KB and
                                 this file grows with the camera list; a blob is exact. */
                              const url = URL.createObjectURL(new Blob([snip.code], { type: 'text/x-python' }))
                              const a = document.createElement('a')
                              a.href = url; a.download = snip.filename; a.click()
                              setTimeout(() => URL.revokeObjectURL(url), 4000)
                            }}>download</button>{' '}
                            <button className="btn ghost tiny" onClick={() => setSnip(null)}>close</button>
                          </span>
                        </div>
                        <pre className="snippet">{snip.code}</pre>
                        {(() => {
                          /** The file went out with NO hub address, so the peer it starts would connect to nothing. */
                          const gap = hubAddressMissing(snip.code)
                          if (!gap) return null
                          const draft = cleanHubHost(hubDraft)
                          const typed = hubDraft.trim() !== ''
                          return (
                            <div>
                              <p className="muted small">
                                No hub address — {gap}. The machine that runs this file needs an
                                address it can reach THIS dashboard at; type the one you know.
                              </p>
                              <div className="row">
                                <label className="field">
                                  <span>Hub address</span>
                                  <input value={hubDraft} placeholder="gpu.lan:7447"
                                         aria-invalid={typed && !draft.host ? true : undefined}
                                         onChange={e => setHubDraft(e.target.value)} />
                                </label>
                                <button className="btn ghost tiny" disabled={!draft.host}
                                        onClick={() => { if (draft.host) void writeSnippet(p, draft.host) }}>
                                  rebuild with this address
                                </button>
                              </div>
                              {typed && draft.why && <p className="muted small">{draft.why}</p>}
                            </div>
                          )
                        })()}
                      </div>
                    )}
                    {calibFor === p.device && (() => {
                      const { family, source } = familyFor(p)
                      const plan = calibratePlan({ ...p, robot_id: knownCalibrationId(profiles, p) }, family)
                      const verdict = copied?.startsWith(p.device + '\u0000') ? copied.split('\u0000')[1] : null
                      return (
                        <div className="calibcmd">
                          <p className="muted small">{plan.reason}</p>
                          {plan.command ? (
                            <>
                              <p className="muted small">
                                model <span className="mono">{plan.deviceModel}</span> — family from {source}
                              </p>
                              {plan.idNote && (
                                <p className={plan.idWarn ? 'hint warn' : 'hint'}>
                                  {plan.idWarn ? '⚠ ' : ''}{plan.idNote}
                                </p>
                              )}
                              {/* The wizard runs the SAME lerobot-calibrate the terminal would,
                                  as a managed session — no terminal needed. */}
                              <CalibrateWizard
                                port={p.device}
                                role={(p.role === 'leader' ? 'leader' : 'follower')}
                                model={plan.deviceModel!}
                                deviceId={plan.deviceId!}
                                onSaved={() => {
                                  api<{ status?: string; text?: string }>('/api/calibration')
                                    .then(r => { if (r?.text) setCalibIds(parseCalibrationList(r.text).entries) })
                                    .catch(() => { /* the receipt names the path either way */ })
                                }}
                              />
                              <details>
                                <summary>run it in a terminal instead</summary>
                                {/* Selectable text first: the copy button is a convenience, not the only route. */}
                                <code className="cmdline">{plan.command}</code>
                                <div className="row">
                                  <button className="btn ghost" onClick={() => void copyCommand(p.device, plan.command!)}>
                                    copy
                                  </button>
                                  {verdict === 'ok' && <span className="muted small">copied — paste it in a terminal</span>}
                                  {verdict && verdict !== 'ok' &&
                                    <span className="warn small">⚠ could not copy: {verdict} — select the line above instead</span>}
                                </div>
                                <p className="hint">
                                  Same procedure as the wizard: it will ask you to move the arm through its range
                                  BY HAND. When the file lands, press <em>reload</em> in Calibration above to see it.
                                </p>
                              </details>
                            </>
                          ) : (
                            <>
                              {plan.needsMeasurement &&
                                <p className="hint">use <em>measure role</em> on this row first — one register read, no motion.</p>}
                              {!family && robots.length > 0 && (
                                <label className="row">
                                  <span className="muted small">which arm is this?</span>
                                  <select value={calibFamily[p.device] ?? ''}
                                          onChange={e => setCalibFamily({ ...calibFamily, [p.device]: e.target.value })}>
                                    <option value="">— pick the robot type —</option>
                                    {robots.map(r => <option key={r.name} value={r.name}>{r.label}</option>)}
                                  </select>
                                </label>
                              )}
                            </>
                          )}
                        </div>
                      )
                    })()}
                    {v && (
                      <p className={v.mismatch ? 'warn small' : 'muted small'}>
                        {v.reason}
                        {v.remedy ? ' — ' + v.remedy : ''}
                        {v.mismatch ? ' ⚠ ' + v.mismatch.message + ' — ' + v.mismatch.remedy : ''}
                        {v.remembered ? ' · remembered for this board' : ''}
                        {v.remember_problem ? ' · ' + v.remember_problem : ''}
                      </p>
                    )}
                  </li>
                )
              })}
            </ul>
          </section>

          <section>
            <h3>Detected hardware</h3>
            <dl className="kv">
              {/* This is the row someone screenshots when asking why their arm is missing, so it is the last place that may overstate: both values said `none` while the request was in flight or had failed. */}
              <dt>serial</dt>
              <dd className="mono">
                {hardwareSummaryValue({
                  scanned: doc !== null, error,
                  items: freePorts.map(p => p.device),
                  emptyNote: '(a servo bus shows up as /dev/tty.usbmodem* or /dev/ttyACM*)',
                })}
              </dd>
              <dt>cameras</dt>
              <dd className="mono">
                {hardwareSummaryValue({
                  scanned: doc !== null, error,
                  items: (doc?.cameras ?? []).map(c => `#${c.index}${c.claimed_by ? `→${c.claimed_by}` : ''}`),
                  emptyNote: 'answered a probe',
                })}
              </dd>
            </dl>
          </section>
        </div>

        {status && <footer className="drawer-foot">{status}</footer>}
      </aside>
    </div>
  )
}
