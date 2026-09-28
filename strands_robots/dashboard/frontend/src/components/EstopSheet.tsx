import { useState } from 'react'
import { estopNothingTargeted, signedRailClaim } from '../lib/estopReach'
import type { EstopResult } from '../types'
import { post, HttpError } from '../lib/endpoints'
import { estopFailureVerdict, resumeFailureVerdict, type FailureVerdict } from '../lib/estopOutcome'

/** The dashboard's own lockout (routes_sim.py: /api/safety). */
export interface SimLockout { state: 'clear' | 'locked' | 'unknown' | string; reason?: string; by?: string; since?: number }

/**
 * Which rail a stop goes to. Two exist on the server: the mesh-backed signed rail at
 * /api/mesh/safety/* (fleet-wide lockout, per-peer answers, override code to resume)
 * and the sim-only lockout at /api/safety/* (this process's simulated robots). The
 * fleet view says which one is live: with the mesh bridge up, the stop fires BOTH -
 * a stop is never refused and never too wide - and the resume clears both.
 */
export function estopPaths(meshBacked: boolean): { estop: string[]; resume: string[] } {
  return meshBacked
    ? { estop: ['/api/mesh/safety/estop', '/api/safety/estop'], resume: ['/api/mesh/safety/resume', '/api/safety/resume'] }
    : { estop: ['/api/safety/estop'], resume: ['/api/safety/resume'] }
}

/** Fleet-wide stop, with a per-peer answer. */
export default function EstopSheet({
  open, onClose, linkWarning, meshBacked = false,
}: {
  open: boolean
  onClose: () => void
  /** Set when this page cannot currently deliver the stop (lib/linkHealth). */
  linkWarning?: string | null
  /** true when /ws/mesh reports the bridge online: the signed fleet rail is reachable. */
  meshBacked?: boolean
}) {
  const [firing, setFiring] = useState(false)
  const [result, setResult] = useState<EstopResult | null>(null)
  // The VERDICT, not the message: whether the stop may have fired is the thing
  // the operator has to act on, and only the status can answer that.
  const [error, setError] = useState<FailureVerdict | null>(null)
  const [code, setCode] = useState('')
  const [resuming, setResuming] = useState(false)
  const [resumeMsg, setResumeMsg] = useState<string | null>(null)
  // The sim-only rail's answer, when that is the rail in use.
  const [simLockout, setSimLockout] = useState<SimLockout | null>(null)
  const paths = estopPaths(meshBacked)

  const resume = async () => {
    if (meshBacked && !code.trim()) return
    setResuming(true); setResumeMsg(null)
    try {
      if (meshBacked) {
        const r = await post<{ status?: string; error?: string }>(paths.resume[0], { override_code: code })
        if (r.status === 'ok') { setResumeMsg('✓ lockout cleared - fleet accepting commands again'); setCode('') }
        else setResumeMsg(`✗ ${r.error ?? 'resume rejected'} (wrong code? brute-force cooldown?)`)
      }
      // The sim rail: a resume leaves the lockout `unknown` until the next accepted command proves it clear.
      const sim = await post<{ lockout: SimLockout }>('/api/safety/resume')
      setSimLockout(sim.lockout)
      if (!meshBacked) setResumeMsg(`✓ sim lockout lifted - ${sim.lockout.state}: ${sim.lockout.reason ?? ''}`)
    } catch (e: any) {
      // A resume whose answer never came back MAY have cleared the lockout;
      // reporting "still locked" would be a guess about the fleet's state.
      setResumeMsg(resumeFailureVerdict({
        status: e instanceof HttpError ? e.status : 0,
        message: e?.message ?? String(e),
      }).text)
    } finally {
      setResuming(false)
    }
  }

  const fire = async () => {
    setFiring(true); setError(null)
    try {
      // The sim rail first: it is local, never refused, and answers in microseconds.
      const sim = await post<{ lockout: SimLockout }>('/api/safety/estop')
      setSimLockout(sim.lockout)
      if (meshBacked) setResult(await post<EstopResult>(paths.estop[0]))
      else setResult(simOnlyResult(sim.lockout))
    } catch (e: any) {
      setError(estopFailureVerdict({
        status: e instanceof HttpError ? e.status : 0,
        message: e?.message ?? String(e),
      }))
    } finally {
      setFiring(false)
    }
  }

  if (!open) return null

  const simOnly = result?.targeted.length === 0 && !meshBacked
  const unconfirmed = result
    ? result.counts.not_stopped + result.counts.no_answer
    : 0

  return (
    <div className="sheet-backdrop" onClick={result && unconfirmed > 0 ? undefined : onClose}>
      <div className={`sheet estop-sheet${unconfirmed > 0 ? ' danger' : ''}`} onClick={e => e.stopPropagation()}>
        <h2>🛑 Stop everything</h2>

        {!result && !error && (
          <>
            <p>
              {meshBacked
                ? <>Sends <code>{'{action: "stop"}'}</code> to every peer with a live heartbeat and reports what each one answered.</>
                : <>Freezes every simulated robot this dashboard is running and latches its lockout; no mesh bridge is online, so no fleet peer is reached from here.</>}
            </p>
            {meshBacked && <p className="hint">
              Fires BOTH rails: per-peer stop commands (answered individually below) and the
              signed <code>strands/safety/estop</code> envelope, which engages a fleet-wide
              LOCKOUT - every listening peer refuses further commands until a resume with the
              operator override code. A peer that is wedged or fully off the mesh still needs
              the hardware e-stop. The simulated robots of this dashboard are frozen too.
            </p>}
            {linkWarning && (
              <p className="hint warn">
                ⚠ {linkWarning} Pressing STOP ALL is still worth it — it is sent the moment the
                link returns — but do not wait for it: the arms’ power switch is the only brake
                that does not go through this page.
              </p>
            )}
            <p className="hint">
              tip: <kbd>.</kbd> opens this sheet from anywhere — it works even when a drawer
              or dialog is covering the button.
            </p>
            <div className="sheet-actions">
              <button className="btn danger big" onClick={fire} disabled={firing}>
                {firing ? 'stopping…' : 'STOP ALL ROBOTS'}
              </button>
              <button className="btn ghost" onClick={onClose} disabled={firing}>cancel</button>
            </div>
          </>
        )}

        {error && (
          <>
            {/* "Nothing was sent" was the old line for EVERY failure — including a lost answer, where the stop may well have landed. */}
            <div className="result bad" role="alert">{error.headline}</div>
            <p className="hint warn">{error.advice}</p>
            <div className="sheet-actions">
              <button className="btn danger" onClick={fire}>
                {error.retryRepeats ? 'send the stop again' : 'retry'}
              </button>
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}

        {result && simOnly && (
          <>
            <div className={simLockout?.state === 'locked' ? 'result ok' : 'result bad'} role="status">
              {simLockout?.state === 'locked'
                ? `✓ sim lockout engaged${simLockout.by ? ` by ${simLockout.by}` : ''} - every simulated robot is frozen`
                : `lockout ${simLockout?.state ?? 'unknown'}: ${simLockout?.reason ?? ''}`}
            </div>
            <p className="hint">
              Nothing outside this process was reached: the mesh bridge is offline, so a real arm on the
              desk still needs its power switch.
            </p>
            {simLockout?.state === 'locked' && (
              <div className="resume-box">
                <div className="resume-row">
                  <button className="btn go" onClick={resume} disabled={resuming}>{resuming ? '…' : 'resume simulation'}</button>
                </div>
                {resumeMsg && <div className="hint">{resumeMsg}</div>}
              </div>
            )}
            <div className="sheet-actions">
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}

        {result && !simOnly && (
          <>
            {/* the answer to "did every robot actually stop?" arrives asynchronously, and until now it arrived SILENTLY — nothing announced it. role=alert on the unconfirmed verdict, because "N peers NOT confirmed stopped" is the one sentence in this dashboard that must interrupt whatever a screen reader was saying; the all-clear is polite. */}
            <div className={result.all_stopped ? 'result ok' : 'result bad'}
                 role={result.all_stopped ? 'status' : 'alert'}>
              {result.all_stopped
                ? `✓ all ${result.counts.stopped} live peer(s) confirmed stopped`
                : `⚠ ${unconfirmed} of ${result.targeted.length} peer(s) NOT confirmed stopped`}
            </div>
            <ul className="estop-list">
              {Object.entries(result.stopped).map(([peer, info]) => (
                <li key={peer} className={info.state}>
                  <b>{peer}</b>
                  <span>{info.state === 'stopped' ? 'stopped'
                    : info.state === 'no_answer' ? 'no answer — may still be moving'
                    : `refused: ${typeof info.detail === 'string' ? info.detail : JSON.stringify(info.detail)}`}</span>
                </li>
              ))}
            </ul>
            {result.stale_skipped.length > 0 && (
              <p className="hint">
                skipped (no heartbeat, cannot be reached): <code>{result.stale_skipped.join(', ')}</code>
              </p>
            )}
            {/* `No live peers were on the mesh.` was read one second after someone hit stop while watching an arm move — and it is a claim about the ROOM made from this dashboard's snapshot. */}
            {result.targeted.length === 0 && (() => {
              const reach = estopNothingTargeted({ staleSkipped: result.stale_skipped })
              return (
                <div className="result bad" role="alert">
                  <b>{reach.headline}</b>
                  <div>{reach.detail}</div>
                </div>
              )
            })()}

            {/* `peers refuse all commands until resumed` was rendered from the ISSUER's latch, which
                is set unconditionally — so it read identically whether the stop reached every peer
                or none. What the peers said decides the sentence now. */}
            {result.lockout_engaged && (() => {
              const claim = signedRailClaim({
                lockoutEngaged: result.lockout_engaged,
                issuer: result.signed_rail?.issuer,
                responsesReceived: result.signed_rail?.responses_received,
                peersNotStopped: result.signed_rail?.peers_not_stopped,
              })
              return (
              <div className="resume-box">
                <div className="result bad" role="alert">
                  <b>🔒 {claim?.headline}</b>
                  <div>{claim?.detail}</div>
                </div>
                <div className="resume-row">
                  <input
                    type="password"
                    placeholder="operator override code" aria-label="operator override code"
                    value={code}
                    onChange={e => setCode(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && resume()}
                    disabled={resuming}
                  />
                  <button className="btn go" onClick={resume} disabled={resuming || !code.trim()}>
                    {resuming ? '…' : 'resume fleet'}
                  </button>
                </div>
                {resumeMsg && <div className="hint">{resumeMsg}</div>}
                <p className="hint">
                  The code is verified locally and an HMAC proof is broadcast — the code itself
                  never crosses the wire. Set <code>STRANDS_MESH_OVERRIDE_CODE</code> identically
                  on every peer.
                </p>
              </div>
              )
            })()}
            {result.signed_rail && !result.signed_rail.signed && (
              <p className="hint warn">
                ⚠ signed rail unavailable ({result.signed_rail.error}) — only per-peer stops were
                sent; no fleet lockout is in place.
              </p>
            )}

            <div className="sheet-actions">
              {!result.all_stopped && <button className="btn danger" onClick={fire}>send again</button>}
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/** The EstopResult shape for a stop that had no fleet to target: the sim rail alone answered. */
function simOnlyResult(_lockout: SimLockout): EstopResult {
  return {
    targeted: [], stale_skipped: [], counts: { stopped: 0, not_stopped: 0, no_answer: 0 },
    all_stopped: true, stopped: {}, lockout_engaged: false,
  }
}
