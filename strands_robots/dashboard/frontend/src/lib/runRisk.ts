/**
 * Is pressing ▶ about to move METAL, or pixels? JOURNEYS.md #3: with a policy selected, typing
 * a sentence enabled ▶, and the 4th click moved a real arm with zero confirmation — no dialog,
 * no mention of the word "physical" anywhere in the app.
 */
import type { Presence } from '../types'

export type RunRisk = {
  /** True when the run is expected to drive physical hardware. */
  physical: boolean
  /** Short reason, shown to the operator so the judgment is auditable. */
  reason: string
  /** The hardware's own name for itself, when it gave one. */
  device: string | null
}

/**
 * Errs toward "physical". A peer whose nature we cannot establish gets the confirm sheet: a
 * needless dialog costs one click, a missing one costs a collision.
 */
export function runRisk(presence?: Presence | null): RunRisk {
  const hw = typeof presence?.hw === 'string' ? presence.hw.trim() : ''
  const type = String(presence?.robot_type ?? '').toLowerCase()

  if (type === 'sim') {
    return { physical: false, reason: 'simulated robot — nothing physical moves', device: null }
  }
  if (hw && !/^(sim|mock|fake|mujoco)/i.test(hw)) {
    return { physical: true, reason: `real hardware attached (${hw})`, device: hw }
  }
  if (hw) {
    return { physical: false, reason: `simulated backend (${hw})`, device: hw }
  }
  if (presence?.connected === false) {
    // Online peer, hardware disconnected: the run will fail rather than move.
    // Still not treated as safe — it may reconnect between judgment and click.
    return { physical: true, reason: 'hardware is not connected right now', device: null }
  }
  return { physical: true, reason: 'this peer did not say whether it is real', device: null }
}
