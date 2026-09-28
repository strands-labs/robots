/** What the "Servo boards" list may say when it has nothing to show. */

export type BoardListEmpty =
  | { kind: 'scanning'; message: string }
  | { kind: 'unscanned'; message: string }
  | { kind: 'detected'; message: string }

export function boardListEmptyLine(opts: { scanned: boolean; error?: string | null }): BoardListEmpty {
  const err = (opts.error ?? '').trim()
  if (!opts.scanned) {
    if (err) {
      return {
        kind: 'unscanned',
        // Never "no boards": nothing was asked successfully, so nothing is known.
        message: `the device scan failed (${err}) — this list is empty because nothing answered, not because nothing is plugged in`,
      }
    }
    return { kind: 'scanning', message: 'scanning USB for servo boards…' }
  }
  // The scan answered and listed none: now the claim is about the hardware, and it says
  // what was actually looked for so an operator can tell it apart from a wiring fault.
  const suffix = err ? ` (the last refresh also reported: ${err})` : ''
  return {
    kind: 'detected',
    message: `no servo board detected — nothing on USB enumerated as a serial bus${suffix}`,
  }
}

/** The shared half of the rule: what to say when the scan itself has not spoken. */
function unanswered(what: string, error?: string | null): BoardListEmpty {
  const err = (error ?? '').trim()
  if (err) {
    return {
      kind: 'unscanned',
      message: `the device scan failed (${err}) — this list is empty because nothing answered, not because there ${what}`,
    }
  }
  return { kind: 'scanning', message: 'scanning this machine…' }
}

/**
 * `Managed robots (0) — None.` came from `doc?.managed ?? {}`, so a failed or pending
 * /api/devices reported zero children — while children were running, publishing to the mesh
 * and holding serial ports.
 */
export function managedListEmptyLine(opts: { scanned: boolean; error?: string | null }): BoardListEmpty {
  if (!opts.scanned) return unanswered('are none', opts.error)
  return {
    kind: 'detected',
    message:
      'None. Spawning one starts a child process that joins the mesh as its own peer — ' +
      'use it for a MuJoCo sim, or to drive a real arm from this machine.',
  }
}

/**
 * `No cameras probed — plug one in and rescan.` came from `doc?.cameras ?? []`: an instruction
 * (go plug in hardware) derived from a request that may never have answered.
 */
export function cameraGridEmptyLine(opts: { scanned: boolean; error?: string | null }): BoardListEmpty {
  if (!opts.scanned) return unanswered('are no cameras', opts.error)
  return {
    kind: 'detected',
    message: 'No camera index answered a probe — plug one in, or rescan if you just did.',
  }
}

/**
 * The terse form, for the `Detected hardware` key/value rows at the foot of the same drawer —
 * the third place this defect lived.
 */
export function hardwareSummaryValue(opts: {
  scanned: boolean
  error?: string | null
  items: string[]
  /** what to append after `none` once the scan HAS answered, e.g. how the device would appear */
  emptyNote: string
}): string {
  const items = opts.items.filter(Boolean)
  if (items.length > 0) return items.join(', ')
  const err = (opts.error ?? '').trim()
  if (!opts.scanned) return err ? `unknown — the scan failed (${err})` : 'unknown — still scanning'
  return `none ${opts.emptyNote}`.trim()
}
