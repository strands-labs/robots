import { useEffect, useState } from 'react'
import type { Peer } from '../types'
import { useTask } from '../lib/useTask'
import { busRecoveryBadge } from '../lib/busRecoveries'
import { lockoutBadge } from '../lib/lockoutBadge'
import { useTelemetry } from '../lib/useTelemetry'
import { peerStatusFields, ribbonDetail, statusSentence } from '../lib/statusSentence'
import { twinButtonCopy } from '../lib/twinButton'
import { deadCameraNote, stoppedCameras } from '../lib/cameraFreshness'
import CameraTile from './CameraTile'
import CameraConfigSheet from './CameraConfigSheet'
import JointStrip from './JointStrip'
import TelemetryStrip from './TelemetryStrip'
import RunForm from './RunForm'
import ConsentSheet from './ConsentSheet'
import { useJointFailure } from '../lib/useJointFailure'

export default function RobotCard({ peer, twinLive = false, onOpen, onBusyChange, hostsChildren}: {
  peer: Peer
  /** a '<id>-twin' peer is live in the fleet (App reads the same snapshot) */
  twinLive?: boolean
  hostsChildren?: string[] | null
  onOpen?: (peerId: string) => void
  onBusyChange?: (peerId: string, running: boolean) => void
}) {
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin, consent, clearConsent, retryLast } = useTask(peer)

  // The sheet opens on request: a refusal must not steal focus from an
  // operator who is watching an arm move.
  const [sheet, setSheet] = useState(false)
  // Camera settings straight from the card: which camera, or an add.
  const [camSheet, setCamSheet] = useState<{ cam: string | null; add: boolean } | null>(null)

  const p = peer.presence
  const type = p?.robot_type ?? '?'
  const cams = Object.keys(peer.cameras ?? {})
  const offline = !!peer.stale
  const telemetry = useTelemetry(peer)
  // R2: a button that spawns a process says so in words.
  const twin = twinButtonCopy({ peerId: peer.peer_id, twinLive, busy: twinBusy })

  const status = type === 'robot'
    ? statusSentence(peerStatusFields(peer, telemetry, hostsChildren))
    : null

  // The app keeps a screen wake lock while anything is moving.
  useEffect(() => { onBusyChange?.(peer.peer_id, running) }, [running, peer.peer_id])  // eslint-disable-line react-hooks/exhaustive-deps

  /** The CAUSE beside the claim. */

  const { badge: whyMute } = useJointFailure(peer.peer_id, Object.keys(peer.state?.joints ?? {}).length === 0)

  return (
    <div className={`card${offline ? ' stale' : ''}${phase === 'failed' ? ' failed' : ''}${running ? ' running' : ''}`}>
      <div className="card-head">
        <span className={`typebadge ${type}`}>{type}</span>
        <button className="peername" title={`open ${peer.peer_id}`} onClick={() => onOpen?.(peer.peer_id)}>
          {peer.peer_id}
        </button>
        {/* Which arm this IS, measured off its bus - the swap the operator reported was visible here first, where a name was the only clue. */}
        {peer.role && (
          <span className={`rolebadge ${peer.role}`}
                title={peer.role_volts
                  ? `measured ${peer.role_volts}V on its servo bus`
                  : 'measured off its servo bus'}>
            {peer.role}
          </span>
        )}
        {/* The e-stop lockout. */}
        {(() => {
          const lb = lockoutBadge(peer.lockout)
          return lb.label ? (
            <span className={`lockbadge ${lb.tone}`} title={lb.title}>{lb.label}</span>
          ) : null
        })()}
        {/* U15: a robot the user started from their own script is a full citizen here - same card, same telemetry, same commands. */}
        {type === 'robot' && peer.origin === 'external' && (
          <span className="originbadge"
                title={'started outside this dashboard (your own script, or another machine).\n'
                  + 'Everything on this card works normally. Only the three things that need a local\n'
                  + 'child process are unavailable: logs, camera reconfigure and despawn.'}>
            external
          </span>
        )}
        {/* the bus cure is silent by design, so the count is the only evidence a cable is failing. */}
        {(() => {
          const bus = busRecoveryBadge(peer.state?.bus_recoveries)
          return bus ? (
            <span className={`badge ${bus.tone}`} title={bus.title}>{bus.label}</span>
          ) : null
        })()}
        {p?.hostname && <span className="host">{p.hostname}</span>}
        {p?.connected === false && type === 'robot' && (
          <span className="badge warn" title="peer is online but its hardware is not connected">hw off</span>
        )}
        {type === 'robot' && !peer.peer_id.includes('__') && !peer.peer_id.endsWith('-twin') && (
          <button className={`twinbtn${twin.cls ? ` ${twin.cls}` : ''}`} onClick={toggleTwin}
                  disabled={twinBusy} title={twin.title} aria-label={twin.aria}
                  aria-pressed={twin.pressed}>{twin.label}</button>
        )}
        <span className={offline ? 'dot off' : running ? 'dot busy' : 'dot on'} role="img"
              aria-label={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'}
              title={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'} />
      </div>

      {whyMute && (

        <div className="small warn" role="status">no joints — {whyMute}</div>

      )}

      {status && (
        <div className={`status-ribbon ${status.severity}`} role="status">
          <b className="status-word">{status.word}</b>
          <span>{ribbonDetail(status)}</span>
        </div>
      )}
      {offline && !status && (
        <div className="stale-note">
          no heartbeat — last seen{' '}
          {peer.last_seen ? `${Math.round(Date.now() / 1000 - peer.last_seen)}s ago` : 'unknown'}
        </div>
      )}

      {(() => {
        const note = deadCameraNote(stoppedCameras(peer.cameras ?? {}, Date.now() / 1000), cams.length)
        return note ? <div className="cam-dead-note" role="status">{note}</div> : null
      })()}

      {(() => {
        // Same gate as RobotDetail's cameras button: a robot started elsewhere has no
        // local process to restart, so the affordance is not offered.
        const canConfig = type === 'robot' && peer.origin !== 'external'
        return (
          <>
            {cams.length > 0 && (
              <div className={cams.length > 1 ? 'cams multi' : 'cams'}>
                {cams.slice(0, 4).map(c => (
                  <CameraTile key={c} peerId={peer.peer_id} cam={c} meta={peer.cameras?.[c]}
                              onConfigure={canConfig ? () => setCamSheet({ cam: c, add: false }) : undefined} />
                ))}
              </div>
            )}
            {canConfig && (
              <button className="chip addcam" onClick={() => setCamSheet({ cam: null, add: true })}
                      title="attach another camera to this robot (applying restarts it)">
                + add camera
              </button>
            )}
            {camSheet && (
              <CameraConfigSheet peerId={peer.peer_id} focusCam={camSheet.cam} startAdding={camSheet.add}
                                 onClose={() => setCamSheet(null)} />
            )}
          </>
        )
      })()}

      <JointStrip state={peer.state} presence={p} problem={peer.joint_problem} peerStale={peer.stale} />
      <TelemetryStrip peer={peer} />

      {peer.stream && (
        <div className="streamline">
          step {peer.stream.step} · {peer.stream.policy || 'policy'} · {peer.stream.instruction?.slice(0, 40)}
        </div>
      )}

      <RunForm
        peerId={peer.peer_id}
        presence={p}
        running={running}
        busy={busy}
        disabled={offline}
        onRun={run}
        onStop={stop}
      />

      {/* ambiguous = the command may have reached the arm; '✗ failed' would be a guess with a hand in the workspace behind it (lib/taskOutcome.ts). */}
      {outcome && (
        <div className={outcome.ok ? 'result ok' : 'result bad'}>
          <span>{outcome.ok ? '✓' : outcome.ambiguous ? '⚠ unknown —' : '✗'} {outcome.text}</span>
          {outcome.detail && <details><summary>details</summary><pre>{outcome.detail}</pre></details>}
          {/* The refusal is answerable: offer the decision where the error is, not in a settings screen the operator has to go find. */}
          {consent && (
            <button className="btn small" onClick={() => setSheet(true)}>review permission…</button>
          )}
        </div>
      )}

      {consent && sheet && (
        <ConsentSheet
          need={consent}
          target="peer"
          onCancel={() => { setSheet(false); clearConsent() }}
          onRetry={() => { setSheet(false); void retryLast() }}
        />
      )}
    </div>
  )
}
