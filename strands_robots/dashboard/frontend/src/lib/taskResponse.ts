/** Reading a run/stop RESPONSE — as pure functions. */
import type { StopResult } from '../types'
import type { TaskPhase } from './taskPhase'

/**
 * Reading a run/stop RESPONSE — as pure functions. lib/taskOutcome already covers the THROWN
 * cases (a rejected fetch, a 5xx: whether the arm may be moving).
 */

export interface Outcome {
  ok: boolean
  text: string
  detail?: string
  ambiguous?: boolean
}

/** Never throws and never returns undefined: this string is only ever shown to a human. */
function describe(value: unknown, cap = 300): string {
  if (value === undefined) return 'no result payload'
  try {
    return String(JSON.stringify(value) ?? String(value)).slice(0, cap)
  } catch {
    return String(value).slice(0, cap)
  }
}

/** The error message a peer's response carries, at either nesting depth, or undefined. */
export function errorInResult(result: any): string | undefined {
  const err = result?.error ?? result?.result?.error
  if (err === undefined || err === null || err === '') return undefined
  return String(err)
}

export interface RunResponse {
  ok?: boolean
  result?: any
  routed_to?: string
  mirrored_to_twin?: boolean
}

/** Did the run actually start? A RESPONSE IS NOT A CONFIRMATION. */
export function interpretRun(res: RunResponse | undefined): { outcome: Outcome; phase: TaskPhase } {
  const err = errorInResult(res?.result)
  if (res?.ok && !err) {
    const via = res.routed_to ? ` via ${res.routed_to}` : ''
    const twin = res.mirrored_to_twin ? ' + twin' : ''
    return { outcome: { ok: true, text: `running${via}${twin}` }, phase: 'running' }
  }
  return {
    outcome: {
      ok: false,
      // The peer's own words beat "refused" — and when the envelope said ok while the payload
      // carried an error, that error IS the news.
      text: err ?? 'refused',
      detail: describe(res?.result),
    },
    phase: 'failed',
  }
}

/**
 * Did the stop actually stop it? stopped / not_stopped / no_answer — never a bare "stopped" on
 * silence.
 */
export function interpretStop(res: StopResult | undefined): { outcome: Outcome; phase: TaskPhase } {
  const state = res?.state
  if (state === 'stopped') return { outcome: { ok: true, text: 'stopped' }, phase: 'idle' }
  if (state === 'no_answer') {
    return {
      outcome: { ok: false, text: 'no answer — robot may still be moving', ambiguous: true },
      phase: 'failed',
    }
  }
  // not_stopped, or a state this bundle does not recognise (an older/newer server). Both mean the
  // same thing to the person standing next to the arm: it was not stopped.
  const detail = typeof res?.detail === 'string' ? res.detail : describe(res?.detail ?? {})
  return { outcome: { ok: false, text: `not stopped: ${detail}` }, phase: 'failed' }
}
