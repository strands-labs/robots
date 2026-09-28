import { useEffect, useMemo, useRef, useState } from 'react'
import { cameraEvidence, cameraPlaceholder } from '../lib/cameraEvidence'
import { useDialogFocus } from '../lib/useDialogFocus'
import type { Peer, StreamStep } from '../types'
import { useTask } from '../lib/useTask'
import { twinButtonCopy } from '../lib/twinButton'
import { statusSentence, peerStatusFields } from '../lib/statusSentence'
import { teleopView, stopVerdict, startVerdict, type TeleopView } from '../lib/teleopView'
import { leaderOptions, pairPlan, teleopSubject, type PairInput } from '../lib/teleopPair'
import { mirrorPlan, mirrorSentence } from '../lib/mirrorToSim'
import { useJointFailure } from '../lib/useJointFailure'
import { api } from '../lib/endpoints'
import { useTelemetry } from '../lib/useTelemetry'
import CameraTile from './CameraTile'
import JointStrip from './JointStrip'
import TelemetryStrip from './TelemetryStrip'
import SensorStrip from './SensorStrip'
import RunForm from './RunForm'
import CameraConfigSheet from './CameraConfigSheet'

const HISTORY = 40

function fmt(v: unknown): string {
  if (typeof v === 'number') return v.toFixed(3)
  if (Array.isArray(v)) return `[${v.map(x => (typeof x === 'number' ? x.toFixed(2) : String(x))).join(', ')}]`
  if (v && typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/** Single-robot stage: one big camera, the whole joint table, and the policy's own step stream. */
export default function RobotDetail({ peer, twinLive = false, hostsChildren, fleet, onOpen, onClose }: {
  peer: Peer
  /** a '<id>-twin' peer is live in the fleet */
  twinLive?: boolean
  hostsChildren?: string[] | null
  fleet?: PairInput[] | null
  /** open another peer's detail — the host→arm teleop redirect uses it */
  onOpen?: (peerId: string) => void
  onClose: () => void
}) {
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin } = useTask(peer)
  const cams = Object.keys(peer.cameras ?? {})
  // R2: same words as the card, from the same pure module.
  const twin = twinButtonCopy({ peerId: peer.peer_id, twinLive, busy: twinBusy })
  const [cam, setCam] = useState<string | null>(null)
  const [teleop, setTeleop] = useState<TeleopView | null | 'asking' | 'unreachable'>(null)
  const askTeleop = async () => {
    setTeleop('asking')
    try { setTeleop(teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`))) }
    // A failed ASK is not an idle arm: say the ask failed. endpoints already explains a 404 from an
    // older server, and swallowing this into "no teleop" would be the same lie the counters told.
    catch { setTeleop('unreachable') }
  }
  const [stopArmed, setStopArmed] = useState(false)
  const [stopped, setStopped] = useState<{ ok: boolean; line: string } | null>(null)
  const [startArmed, setStartArmed] = useState<string | null>(null)
  const [started, setStarted] = useState<{ ok: boolean; line: string } | null>(null)
  const startTeleop = async (leaderId: string) => {
    setStartArmed(null); setStarted(null); setStopped(null); setTeleop('asking')
    try {
      await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop/publish`, { method: 'POST', body: JSON.stringify({}) })
    } catch (e) {
      setTeleop('unreachable'); setStarted({ ok: false, line: `${leaderId} could not start publishing, so nothing was pointed at it: ${(e as Error).message}` }); return
    }
    try {
      // confirmed: this call is only reachable through the armed 'confirm — … MOVES with it' button above.
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/receive`, { method: 'POST', body: JSON.stringify({ source_peer_id: leaderId, confirmed: true }) })
    } catch (e) {
      /** HALF-BUILT CHAIN: the leader IS publishing now and the follower refused. */
      setTeleop('unreachable'); setStranded(leaderId)
      setStarted({ ok: false, line: `${leaderId} is publishing its joints, but ${peer.peer_id} would not follow it: ${(e as Error).message} — nothing is moving, and ${leaderId} is still on the wire until you stop it below` })
      return
    }
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`)) } catch { after = null }
    setTeleop(after ?? 'unreachable')
    setStarted(startVerdict(after))
  }

  const [stranded, setStranded] = useState<string | null>(null)
  const [strandedResult, setStrandedResult] = useState<{ ok: boolean; line: string } | null>(null)
  const stopStranded = async (leaderId: string) => {
    setStrandedResult(null)
    try { await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop/stop`, { method: 'POST' }) }
    catch (e) {
      setStrandedResult({ ok: false, line: `${leaderId} would not stop publishing: ${(e as Error).message} — it is still on the wire` }); return
    }
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop`)) } catch { after = null }
    const v = stopVerdict(after)
    setStrandedResult({ ok: v.ok, line: `${leaderId}: ${v.line}` })
  }

  const stopTeleop = async () => {
    setStopArmed(false); setStopped(null); setTeleop('asking')
    try { await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/stop`, { method: 'POST' }) }
    catch (e) { setTeleop('unreachable'); setStopped({ ok: false, line: `stop was refused: ${(e as Error).message}` }); return }
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`)) } catch { after = null }
    setTeleop(after ?? 'unreachable')
    setStopped(stopVerdict(after))
  }

  /**
   * MIRROR TO SIM — the real arm's own affordance for "hand-move me and my
   * twin copies it". Same rail as teleop (publish on THIS arm, receive on the
   * twin's robot), but discovered from the REAL arm's card, because the
   * operator standing at the arm should not need to know the twin's child
   * peer id to use it. The "following" indicator is derived by ASKING the
   * follower again, never from the POSTs returning 200 (the /teleop/receive
   * lesson: it once said "started" for a stream whose every frame was refused).
   */
  const mirror = peer.presence?.robot_type === 'robot' && (fleet?.length ?? 0) > 0
    ? mirrorPlan(peer.peer_id, fleet!)
    : null
  const [mirrorArmed, setMirrorArmed] = useState(false)
  const [mirrorBusy, setMirrorBusy] = useState(false)
  /** the follower this card started a mirror on — the stop button's target */
  const [mirrorOn, setMirrorOn] = useState<string | null>(null)
  const [mirrorLine, setMirrorLine] = useState<{ ok: boolean; line: string } | null>(null)
  const askMirror = async (followerId: string) => {
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(followerId)}/teleop`)) } catch { after = null }
    return after
  }
  const startMirror = async (followerId: string) => {
    setMirrorArmed(false); setMirrorBusy(true); setMirrorLine(null)
    try {
      // publish is read-only on this arm: the mover is whoever receives.
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/publish`, { method: 'POST', body: JSON.stringify({}) })
    } catch (e) {
      setMirrorBusy(false)
      setMirrorLine({ ok: false, line: `${peer.peer_id} could not start publishing its joints, so nothing was mirrored: ${(e as Error).message}` })
      return
    }
    try {
      // confirmed: reached only through the armed confirm button; the follower here is a sim twin anyway.
      await api(`/api/robots/${encodeURIComponent(followerId)}/teleop/receive`, { method: 'POST', body: JSON.stringify({ source_peer_id: peer.peer_id, confirmed: true }) })
    } catch (e) {
      setMirrorBusy(false); setMirrorOn(followerId) // half-built: the stop button must appear
      setMirrorLine({ ok: false, line: `${peer.peer_id} is publishing, but ${followerId} would not follow: ${(e as Error).message} — stop below removes the publisher` })
      return
    }
    const after = await askMirror(followerId)
    setMirrorBusy(false); setMirrorOn(followerId)
    const v = startVerdict(after)
    setMirrorLine({ ok: v.ok, line: v.ok ? `mirror live — ${followerId} is copying this arm's joints` : v.line })
  }
  const refreshMirror = async (followerId: string) => {
    setMirrorBusy(true)
    const after = await askMirror(followerId)
    setMirrorBusy(false)
    if (!after) { setMirrorLine({ ok: false, line: `${followerId} did not answer when asked — nothing confirms the mirror state` }); return }
    setMirrorLine({ ok: after.streaming && after.tone === 'ok', line: `${followerId}: ${after.headline}` })
  }
  const stopMirror = async (followerId: string) => {
    // ONE TAP, never armed: stopping a mirror only removes commands.
    setMirrorBusy(true); setMirrorLine(null)
    let failed = ''
    try { await api(`/api/robots/${encodeURIComponent(followerId)}/teleop/stop`, { method: 'POST' }) }
    catch (e) { failed = `${followerId} refused stop: ${(e as Error).message}` }
    try { await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/stop`, { method: 'POST' }) }
    catch (e) { failed = failed ? `${failed}; ` : ''; failed += `${peer.peer_id} refused stop: ${(e as Error).message}` }
    const after = await askMirror(followerId)
    setMirrorBusy(false)
    const v = stopVerdict(after)
    if (v.ok && !failed) setMirrorOn(null)
    setMirrorLine(failed ? { ok: false, line: failed } : { ok: v.ok, line: v.ok ? 'mirror stopped — nothing is being copied' : v.line })
  }
  const sheetRef = useRef<HTMLElement | null>(null)
  useDialogFocus(sheetRef)
  const [camConfig, setCamConfig] = useState<{ cam: string | null; add: boolean } | null>(null)
  const canConfig = peer.presence?.robot_type === 'robot' && peer.origin !== 'external'
  const [steps, setSteps] = useState<StreamStep[]>([])
  const lastStep = useRef<number | null>(null)

  const active = cam && cams.includes(cam) ? cam : cams[0] ?? null

  useEffect(() => {
    const s = peer.stream
    if (!s || s.step === lastStep.current) return
    lastStep.current = s.step
    setSteps(prev => [s, ...prev].slice(0, HISTORY))
  }, [peer.stream])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const p = peer.presence
  const offline = !!peer.stale
  const joints = Object.entries(peer.state?.joints ?? {})

  /**
   * AN ARM WITH NO JOINTS IS THE MOST IMPORTANT SENTENCE ON THIS SCREEN, and until iter 487 it
   * was the one sentence nowhere on it: both real arms have been jointless for three days while
   * presence said connected and camera frames kept flowing, the cause a WARNING in the middle of
   * a ring buffer whose tail reads "hardware connected … online".
   */
  const { line: whyNoJoints } = useJointFailure(peer.peer_id, joints.length === 0)

  const telemetry = useTelemetry(peer)
  const status = (p?.robot_type ?? '?') === 'robot'
    ? statusSentence(peerStatusFields(peer, telemetry, hostsChildren))
    : null

  const stepHz = useMemo(() => {
    if (steps.length < 2) return 0
    const span = steps[0].t - steps[steps.length - 1].t
    return span > 0 ? (steps.length - 1) / span : 0
  }, [steps])

  return (
    <div className="detail-backdrop" onClick={onClose}>
      <section ref={sheetRef} className={`detail${offline ? ' stale' : ''}`}
               role="dialog" aria-label={`Robot ${peer.peer_id}`} onClick={e => e.stopPropagation()}>
        <header className="detail-head">
          <span className={`typebadge ${p?.robot_type ?? '?'}`}>{p?.robot_type ?? '?'}</span>
          <h2>{peer.peer_id}</h2>
          {/* three states told apart by COLOUR ALONE said nothing to a screen reader and nothing under forced colours. */}
          <span className={offline ? 'dot off' : running ? 'dot busy' : 'dot on'} role="img"
                aria-label={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'}
                title={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'} />
          {p?.hostname && <span className="host">{p.hostname}</span>}
          {p?.robot_type === 'robot' && !peer.peer_id.endsWith('-twin') && (
            <button className={`twinbtn${twin.cls ? ` ${twin.cls}` : ''}`} onClick={toggleTwin}
                    disabled={twinBusy} title={twin.title} aria-label={twin.aria}
                  aria-pressed={twin.pressed}>{twin.label}</button>
          )}
          {p?.robot_type === 'robot' && peer.origin === 'external' && (
            <span className="originbadge"
                  title={'started outside this dashboard (your own script, or another machine).\n'
                    + 'Everything here works normally except the three things that need a local\n'
                    + 'child process: logs, camera reconfigure and despawn.'}>
              external
            </span>
          )}
          {p?.robot_type === 'robot' && (
            <button className="btn ghost" onClick={() => setCamConfig({ cam: null, add: false })}
                    disabled={peer.origin === 'external'}
                    title={peer.origin === 'external'
                      ? 'this robot was started outside the dashboard, so it has no local process to '
                        + 'restart — change its cameras where it was launched (its own script or machine)'
                      : 'attach / detach cameras, change fps and resolution (restarts the robot)'}>
              cameras
            </button>
          )}
          <button className="btn ghost" onClick={askTeleop} disabled={teleop === 'asking'}
                  title="is this arm following another arm, or publishing its own joints? (reads only)">
            {teleop === 'asking' ? 'asking…' : 'teleop'}
          </button>
          <button className="btn ghost" onClick={onClose} aria-label="close this robot" title="Escape">✕</button>
        </header>
        {teleop === 'unreachable' && (
          <p className="hint warn" role="status">could not ask this arm about teleop — it may be busy or gone; its own log (devices › logs) is where a refusal appears</p>
        )}
        {teleop && typeof teleop === 'object' && (
          <div className={`hint ${teleop.tone === 'warn' ? 'warn' : ''}`} role="status">
            <b>teleop:</b> {teleop.headline}
            {teleop.streaming && <span className="muted small"> · frames are on the wire</span>}
            {teleop.detail && <div className="muted small">{teleop.detail}</div>}
            {/* The envelope is a SAFETY bound: the screen names the consent that widens it and never widens it here. */}
            {teleop.streaming && (
              stopArmed
                ? (
                  <div className="row small">
                    <button className="btn danger" onClick={stopTeleop}>confirm — stop teleop on {peer.peer_id}</button>
                    <button className="btn ghost" onClick={() => setStopArmed(false)}>keep it running</button>
                  </div>
                )
                : <button className="btn ghost small" onClick={() => setStopArmed(true)}
                          title="stop the teleop stream on this arm — it only removes commands, it cannot move anything">
                    stop teleop
                  </button>
            )}
            {/* U22 slice 3a: what would it TAKE to teleop this arm? */}
            {!teleop.streaming && (fleet?.length ?? 0) > 0 && (() => {
              // A PROCESS card (a sim twin, a multi-robot host) is not the thing to teleop:
              // the arm lives on a child peer — send the operator there instead of listing
              // leader refusals this card can never satisfy.
              const subject = teleopSubject(peer.peer_id, fleet!)
              if (subject) {
                return (
                  <div className="small muted">
                    {subject.why}
                    <div className="row small">
                      {subject.children.map(c => (
                        <button key={c} className="btn ghost small" onClick={() => onOpen?.(c)}
                                disabled={!onOpen} title={`open ${c} — teleop starts from the arm itself`}>
                          open {c}
                        </button>
                      ))}
                    </div>
                  </div>
                )
              }
              const opts = leaderOptions(peer.peer_id, fleet!)
              const usable = opts.filter(o => o.ok)
              if (usable.length) {
                const plan = pairPlan(peer.peer_id, usable[0].peer_id, fleet!)
                return (
                  <div className="small muted">
                    could follow: {usable.map(o => `${o.peer_id} (${o.why})`).join(', ')}
                    {plan && !plan.blockers.length && (
                      <> · starting asks for {plan.consents.join(' + ')} first, {plan.physical
                        ? 'because frames move a real arm'
                        : 'because raw degree frames are refused by the unit envelope even in sim'}</>
                    )}
                    {plan?.notes.map((n, i) => <div key={i}>{n}</div>)}
                    {/* One armed step per candidate leader. */}
                    <div className="row small">
                      {usable.map(o => (startArmed === o.peer_id ? (
                        <span key={o.peer_id} className="row small">
                          <button className={plan?.physical === false ? 'btn' : 'btn danger'} onClick={() => startTeleop(o.peer_id)}>
                            {plan?.physical === false
                              ? `confirm — ${peer.peer_id} (sim) follows ${o.peer_id}; nothing physical moves`
                              : `confirm — hand-guide ${o.peer_id}, and ${peer.peer_id} MOVES with it`}
                          </button>
                          <button className="btn ghost" onClick={() => setStartArmed(null)}>cancel</button>
                        </span>
                      ) : (
                        <button key={o.peer_id} className="btn ghost small" onClick={() => setStartArmed(o.peer_id)}
                                title={plan?.physical === false
                                  ? `${peer.peer_id} is simulated — it will mirror ${o.peer_id}'s joints in the sim`
                                  : `${peer.peer_id} will follow ${o.peer_id}'s joints and move`}>
                          follow {o.peer_id}
                        </button>
                      )))}
                    </div>
                  </div>
                )
              }
              return (
                <div className="small muted">
                  no arm on this fleet can lead it yet:
                  {opts.map(o => <div key={o.peer_id}>· {o.peer_id} — {o.why}</div>)}
                </div>
              )
            })()}
            {teleop.consentKind && (
              <div className="muted small">every frame is outside the safety envelope — settings › consent › {teleop.consentKind} is where that bound is widened, deliberately and by you</div>
            )}
          </div>
        )}
        {/* THE VERDICT ABOUT MY OWN ACTION MUST NOT LIVE INSIDE THE ARM'S STATUS BLOCK. */}
        {stopped && (
          <div className={`small ${stopped.ok ? 'muted' : 'warn'}`} role="status">{stopped.line}</div>
        )}
        {/* The result of a START must OUTLIVE the state it created. */}
        {started && (
          <div className={`small ${started.ok ? 'muted' : 'warn'}`} role="status">{started.line}</div>
        )}
        {stranded && (
          <div className="row small">
            <button className="btn ghost small" onClick={() => stopStranded(stranded)}
                    disabled={strandedResult?.ok} title={`stop ${stranded} publishing its joints`}>
              stop {stranded} publishing
            </button>
            {strandedResult && (
              <span className={strandedResult.ok ? 'muted' : 'warn'} role="status">{strandedResult.line}</span>
            )}
          </div>
        )}
        {/* MIRROR TO SIM — from the real arm's own card. Hand-move the arm, the twin copies it. */}
        {mirror && (
          <div className="hint" role="status">
            <b>mirror to sim:</b>{' '}
            {mirrorOn ? (
              <>
                <span className={mirrorLine?.ok ? '' : 'muted'}>
                  {mirrorLine?.line ?? `mirror started on ${mirrorOn}`}
                </span>
                <div className="row small">
                  {/* one tap, never armed: stop only removes commands */}
                  <button className="btn danger small" onClick={() => stopMirror(mirrorOn)} disabled={mirrorBusy}>
                    stop mirror
                  </button>
                  <button className="btn ghost small" onClick={() => refreshMirror(mirrorOn)} disabled={mirrorBusy}
                          title="ask the twin what it is actually doing — the indicator is measured, not assumed">
                    {mirrorBusy ? 'asking…' : 'check'}
                  </button>
                </div>
              </>
            ) : (
              <>
                <span className="muted small">{mirrorSentence(mirror)}</span>
                {mirror.notes.map((n, i) => <div key={i} className="muted small">{n}</div>)}
                {mirror.follower && (
                  <div className="row small">
                    {mirrorArmed ? (
                      <>
                        {/* not the danger style: the follower is a sim, nothing physical moves */}
                        <button className="btn" onClick={() => startMirror(mirror.follower!)} disabled={mirrorBusy}>
                          confirm — publish this arm's joints; {mirror.follower} (sim) copies them
                        </button>
                        <button className="btn ghost" onClick={() => setMirrorArmed(false)}>cancel</button>
                      </>
                    ) : (
                      <button className="btn ghost small" onClick={() => setMirrorArmed(true)} disabled={mirrorBusy}
                              title="start the joint stream: this arm publishes (read-only), the twin follows — nothing physical moves">
                        mirror to {mirror.follower}
                      </button>
                    )}
                  </div>
                )}
                {mirrorLine && !mirrorLine.ok && (
                  <div className="warn small" role="status">{mirrorLine.line}</div>
                )}
              </>
            )}
          </div>
        )}
        {/* the safety sentence belongs on THIS surface most — it is what an operator reads while walking up to the arm. */}
        {whyNoJoints && (
          <p className="hint warn" role="status">{whyNoJoints}</p>
        )}
        {status && (
          <div className={`status-ribbon ${status.severity}`} role="status">
            <b>{status.word}</b> {status.text}
          </div>
        )}

        {offline && (
          <div className="stale-note">
            no heartbeat — last seen{' '}
            {peer.last_seen ? `${Math.round(Date.now() / 1000 - peer.last_seen)}s ago` : 'unknown'}.
            Commands will time out.
          </div>
        )}

        {camConfig && (
          <CameraConfigSheet peerId={peer.peer_id} focusCam={camConfig.cam} startAdding={camConfig.add}
                             onClose={() => setCamConfig(null)} />
        )}

        <div className="detail-body">
          <div className="detail-stage">
            {active
              ? <CameraTile peerId={peer.peer_id} cam={active} meta={peer.cameras?.[active]} big />
              : (() => {
                  // "this peer publishes none" was a denial of the presence THIS SAME peer announces
                  // (lib/cameraEvidence): on a machine where macOS blocks capture, both arms announce top+wrist
                  // and deliver nothing, and the detail screen is where the operator comes to find out why.
                  const ph = cameraPlaceholder(cameraEvidence(peer.peer_id, peer.presence?.cameras, cams, peer.cameras_requested))
                  return (
                    <div className="camtile big">
                      <div className="camstate" title={ph?.title}>
                        <b>{ph?.head ?? 'no camera'}</b><span>{ph?.sub ?? ''}</span>
                      </div>
                    </div>
                  )
                })()}
            {(cams.length > 1 || canConfig) && (
              <div className="camswitch">
                {cams.length > 1 && cams.map(c => (
                  /**
                   * aria-pressed, not colour alone: `.chip.on` is the ONLY thing that said which camera is on
                   * screen, so a screen reader announced two identical "wrist, button" controls and voice
                   * control could not confirm a switch.
                   */
                  <button key={c} className={c === active ? 'chip on' : 'chip'}
                          aria-pressed={c === active} onClick={() => setCam(c)}>{c}</button>
                ))}
                {canConfig && (
                  <button className="chip addcam" onClick={() => setCamConfig({ cam: null, add: true })}
                          title="attach another camera to this robot (applying restarts it)">
                    + add camera
                  </button>
                )}
              </div>
            )}
            <TelemetryStrip peer={peer} />
            <RunForm
              peerId={peer.peer_id}
              presence={p}
              running={running}
              busy={busy}
              disabled={offline}
              onRun={run}
              onStop={stop}
            />
            {outcome && (
              <div className={outcome.ok ? 'result ok' : 'result bad'}>
                <span>{outcome.ok ? '✓' : outcome.ambiguous ? '⚠ unknown —' : '✗'} {outcome.text}</span>
                {outcome.detail && <details><summary>details</summary><pre>{outcome.detail}</pre></details>}
              </div>
            )}
            {phase === 'stopping' && <div className="hint">stop sent, waiting for the peer to confirm…</div>}
          </div>

          <div className="side">
            {/* Above Joints because a rover has no joints and every one of these instead: the
                strip renders nothing at all on an arm, so this costs an arm's card nothing. */}
            <SensorStrip peer={peer} />
            <h3>Joints ({joints.length})</h3>
            <JointStrip state={peer.state} presence={p} problem={peer.joint_problem} peerStale={peer.stale} />
            {joints.length > 0 && (
              <table className="jointtable">
                <thead><tr><th>joint</th><th>pos</th><th>vel</th></tr></thead>
                <tbody>
                  {joints.map(([name, v]) => {
                    const obj = v && typeof v === 'object' && !Array.isArray(v)
                      ? v as { position?: number; velocity?: number }
                      : null
                    const pos = obj?.position ?? (typeof v === 'number' ? v : Array.isArray(v) ? v[0] : undefined)
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className="mono">{typeof pos === 'number' ? pos.toFixed(4) : '—'}</td>
                        <td className="mono">{obj?.velocity !== undefined ? obj.velocity.toFixed(3) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}

            <h3>Peer</h3>
            <dl className="kv">
              <dt>tool</dt><dd>{p?.tool_name ?? '—'}</dd>
              <dt>hardware</dt>
              <dd className={p?.connected === false ? 'bad' : ''}>
                {p?.connected === false ? 'not connected' : p?.hw ?? (p?.connected ? 'connected' : '—')}
              </dd>
              <dt>task</dt>
              <dd>{peer.state?.task?.status ?? p?.task_status ?? 'idle'}
                {peer.state?.task?.instruction ? ` — ${peer.state.task.instruction}` : ''}</dd>
              {p?.sim_robots?.length ? (
                <><dt>sim bodies</dt><dd className="mono">{p.sim_robots.join(', ')}</dd></>
              ) : null}
              {p?.action_keys?.length ? (
                <><dt>action keys</dt><dd className="mono">{p.action_keys.join(', ')}</dd></>
              ) : null}
              {p?.topics?.length ? (
                <><dt>topics</dt><dd className="mono small">{p.topics.join('\n')}</dd></>
              ) : null}
            </dl>

            <h3>
              Policy steps {steps.length > 0 && <em>{stepHz > 0 ? `${stepHz.toFixed(1)} Hz` : ''}</em>}
            </h3>
            {steps.length === 0 ? (
              <p className="hint">
                Nothing yet. Steps only arrive while a policy is running and only from the moment
                this view opened — the stream topic is not replayed.
              </p>
            ) : (
              <ul className="steps">
                {steps.map(s => (
                  <li key={`${s.step}-${s.t}`}>
                    <span className="stepno">#{s.step}</span>
                    <span className="mono">
                      {Object.entries(s.action ?? {}).slice(0, 8)
                        .map(([k, v]) => `${k}=${fmt(v)}`).join('  ') || '(no action)'}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
