/** What we believe about this robot's task. */
import { useEffect, useRef, useState } from 'react'
import { deriveTaskFlags, nextPhase, reportedTaskStatus } from './taskPhase'
import { interpretRun, interpretStop } from './taskResponse'
import type { Outcome } from './taskResponse'
import type { TaskPhase } from './taskPhase'
import type { Peer, StopResult } from '../types'
import { HttpError, post } from './endpoints'
import { findConsent, type ConsentNeed } from './consent'
import { runFailure, stopFailure } from './taskOutcome'
import type { RunBody } from '../components/RunForm'

/** What we believe about this robot's task. */
export type { TaskPhase } from './taskPhase'

// The verdict shape lives with the code that decides it, so the two cannot drift apart.
export type { Outcome } from './taskResponse'

/**
 * Run/stop for one peer, shared by the card and the detail view so both report the *same*
 * verdict.
 */
export function useTask(peer: Peer) {
  const [phase, setPhase] = useState<TaskPhase>('idle')
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [twinBusy, setTwinBusy] = useState(false)
  const [consent, setConsent] = useState<ConsentNeed | null>(null)
  const lastBody = useRef<RunBody | null>(null)
  const mounted = useRef(true)
  useEffect(() => () => { mounted.current = false }, [])

  // Phase logic lives in ./taskPhase as a decision table (tested there; in this body it needed a
  // rendered card against a live peer).
  const reported = reportedTaskStatus(peer)
  const { running, busy } = deriveTaskFlags({ phase, reported, twinBusy })

  useEffect(() => {
    const next = nextPhase(phase, reported)
    if (next) setPhase(next)
  }, [reported])   // eslint-disable-line react-hooks/exhaustive-deps

  /** Bare message - only for requests with nothing physical behind them. */
  const fail = (e: unknown): Outcome => {
    if (e instanceof HttpError) {
      return { ok: false, text: e.message, detail: e.status ? `HTTP ${e.status}` : 'unreachable' }
    }
    return { ok: false, text: e instanceof Error ? e.message : String(e) }
  }

  /**
   * A thrown run/stop is NOT proof the robot was left alone: a rejected fetch covers "never left
   * this machine" and "dispatched, then lost the answer", and a 5xx means the handler ran.
   * lib/taskOutcome states which world it is and what to do about it - see that file for why
   * "failed" was the dangerous word.
   */
  const physicalFail = (e: unknown, kind: 'run' | 'stop'): Outcome => {
    const f = { status: e instanceof HttpError ? e.status : 0, message: e instanceof Error ? e.message : String(e) }
    const v = kind === 'run' ? runFailure(f) : stopFailure(f)
    return { ok: false, text: v.text, detail: v.detail, ambiguous: v.ambiguous }
  }

  const run = async (body: RunBody) => {
    setPhase('starting'); setOutcome(null); setConsent(null)
    lastBody.current = body
    try {
      const res = await post<{ ok: boolean; result: any; routed_to?: string; mirrored_to_twin?: boolean }>(
        `/api/robots/${encodeURIComponent(peer.peer_id)}/task`,
        /**
         * The confirmation marker: this call is only reachable through the run form, whose ▶ passes
         * the RunConfirm dialog first, so the browser can honestly say a human confirmed.
         */
        { ...body, confirmed: true },
      )
      if (!mounted.current) return
      // ./taskResponse decides what the response SAYS (tested there).
      const v = interpretRun(res)
      setOutcome(v.outcome)
      setPhase(v.phase)
      if (!v.outcome.ok) setConsent(findConsent(res))
    } catch (e) {
      if (!mounted.current) return
      setOutcome(physicalFail(e, 'run')); setPhase('failed')
      // A 4xx carries the same needs_consent in its body - a validation refusal
      // must be as answerable as a peer's refusal.
      if (e instanceof HttpError) setConsent(findConsent(e.body))
    }
  }

  /** Re-send the exact request that was refused (after a grant). */
  const retryLast = async () => {
    setConsent(null)
    if (lastBody.current) await run(lastBody.current)
  }

  const stop = async () => {
    setPhase('stopping'); setOutcome(null)
    try {
      const res = await post<StopResult>(`/api/robots/${encodeURIComponent(peer.peer_id)}/stop`)
      if (!mounted.current) return
      // stopped / not_stopped / no_answer - never a bare "stopped" on silence.
      const v = interpretStop(res)
      setOutcome(v.outcome)
      setPhase(v.phase)
    } catch (e) {
      if (!mounted.current) return
      setOutcome(physicalFail(e, 'stop')); setPhase('failed')
    }
  }

  const toggleTwin = async () => {
    setTwinBusy(true)
    try {
      await post(`/api/robots/${encodeURIComponent(peer.peer_id)}/twin`, {})
    } catch (e) {
      setOutcome(fail(e))
    } finally {
      if (mounted.current) setTwinBusy(false)
    }
  }

  return {
    phase, outcome, running, busy, twinBusy, run, stop, toggleTwin, setOutcome,
    consent, clearConsent: () => setConsent(null), retryLast,
  }
}
