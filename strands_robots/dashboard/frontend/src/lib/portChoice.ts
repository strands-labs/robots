/** the chosen servo bus can go stale between picking it and pressing spawn. */
export interface PortChoiceInput {
  /** The path currently held in the form. '' = nothing picked yet. */
  chosen: string
  /** Ports the most recent scan reported. */
  known: string[]
  /** Ports a live managed child is holding. */
  claimed: string[]
  /** False while the first scan is still in flight — absence is not evidence yet. */
  scanned?: boolean
}

export type PortChoice =
  | { kind: 'empty' }
  /** Not yet knowable: no scan has landed, so "not in the list" means nothing. */
  | { kind: 'unknown'; detail: string }
  | { kind: 'ok'; port: string }
  | { kind: 'vanished'; port: string; detail: string; remedy: string }
  | { kind: 'claimed'; port: string; detail: string; remedy: string }

export function portChoice(input: PortChoiceInput): PortChoice {
  const chosen = (input.chosen ?? '').trim()
  if (!chosen) return { kind: 'empty' }

  const scanned = input.scanned !== false
  const known = input.known ?? []
  const claimed = input.claimed ?? []

  // Claimed outranks vanished: a bus held by a running robot is a fact about the fleet, while a path
  // missing from a scan can also just be a scan that raced the device.
  if (claimed.includes(chosen)) {
    return {
      kind: 'claimed',
      port: chosen,
      detail:
        `${chosen} is now held by a robot that is already running — two owners on one servo bus is the ` +
        `"Port is in use!" collision, and it usually shows up as an arm that starts and immediately dies`,
      remedy: 'pick another bus, or despawn the robot holding this one',
    }
  }

  if (!scanned || known.length === 0) {
    return {
      kind: 'unknown',
      detail: `${chosen} has not been confirmed by a scan yet`,
    }
  }

  if (!known.includes(chosen)) {
    return {
      kind: 'vanished',
      port: chosen,
      detail:
        `${chosen} is no longer on this machine — the board was unplugged, or it came back under a ` +
        `different /dev path, which these arms do on every reconnect. The picker shows blank because ` +
        `nothing matches, but this path is still what would be opened.`,
      remedy: 'rescan, then pick the bus again',
    }
  }

  return { kind: 'ok', port: chosen }
}

/** Would spawning with this choice be a mistake we can already see? */
export function blocksSpawn(c: PortChoice): boolean {
  return c.kind === 'vanished' || c.kind === 'claimed'
}
