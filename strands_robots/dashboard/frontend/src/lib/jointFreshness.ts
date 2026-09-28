/**
 * How old are the numbers on a joint strip? The strip renders `state.joints` as bold values
 * and filled bars.
 */

export type JointFreshness = 'live' | 'lagging' | 'frozen' | 'unknown'

export interface JointAgeNote {
  level: JointFreshness
  /** null when there is nothing worth saying (fresh, or never received). */
  text: string | null
  /** true when the values must not be presented as the arm's position now. */
  dim: boolean
}

export const LAGGING_MS = 2_000
export const FROZEN_MS = 10_000

function human(ms: number): string {
  const s = ms / 1000
  return s >= 10 ? `${Math.round(s)}s` : `${s.toFixed(1)}s`
}

export function jointAgeNote(ageMs: number | null | undefined): JointAgeNote {
  if (ageMs === null || ageMs === undefined || !Number.isFinite(ageMs)) {
    return { level: 'unknown', text: null, dim: false }
  }
  const age = Math.max(0, ageMs)
  if (age < LAGGING_MS) return { level: 'live', text: null, dim: false }
  if (age < FROZEN_MS) {
    return {
      level: 'lagging',
      text: `⚠ these numbers are ${human(age)} old — the state stream is lagging`,
      dim: false,
    }
  }
  return {
    level: 'frozen',
    // Name the failure AND the wrong conclusion, because the wrong conclusion is
    // the one a hand acts on.
    text: `⚠ frozen ${human(age)} ago — this is the last frame received, not where the arm is now`,
    dim: true,
  }
}
