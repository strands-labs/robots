/** The devices screen's `rescan` button, made honest. */

export interface ScanPort { device?: string | null }
export interface ScanCamera { index?: number | null; name_hint?: string | null }
export interface ScanShape {
  serial_ports?: ScanPort[] | null
  cameras?: ScanCamera[] | null
  camera_problem?: { kind?: string; message?: string } | null
}

export type RescanOutcome =
  | { ok: true; after: ScanShape | null }
  | { ok: false; error?: string | null }

export interface RescanReport {
  /** ok = it ran and told us something; warn = it ran and found nothing; bad = it did not run. */
  tone: 'ok' | 'warn' | 'bad'
  text: string
  /** true only when the list on screen is NOT the result of this scan. */
  stale: boolean
}

const MAX_NAMED = 3

function ports(s: ScanShape | null | undefined): string[] {
  const list = Array.isArray(s?.serial_ports) ? s!.serial_ports! : []
  return list.map(p => String(p?.device ?? '').trim()).filter(Boolean)
}

function cameras(s: ScanShape | null | undefined): string[] {
  const list = Array.isArray(s?.cameras) ? s!.cameras! : []
  return list
    .map(c => (c?.index == null ? '' : `index ${c.index}`))
    .filter(Boolean)
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`
}

/** "+1 serial port (/dev/tty.usbmodem1)" — names them, but never a wall of text. */
function delta(kind: string, added: string[], removed: string[]): string[] {
  const out: string[] = []
  for (const [sign, list] of [['+', added], ['−', removed]] as const) {
    if (!list.length) continue
    const shown = list.slice(0, MAX_NAMED).join(', ')
    const more = list.length > MAX_NAMED ? ` +${list.length - MAX_NAMED} more` : ''
    out.push(`${sign}${plural(list.length, kind)} (${shown}${more})`)
  }
  return out
}

function ageWords(beforeAtMs?: number | null, nowMs?: number | null): string {
  if (!beforeAtMs || !nowMs || nowMs < beforeAtMs) return ''
  const secs = Math.round((nowMs - beforeAtMs) / 1000)
  if (secs < 5) return ''
  if (secs < 90) return ` (${secs}s old)`
  return ` (${Math.round(secs / 60)}min old)`
}

/**
 * An identity for "what hardware this payload lists", so the component can tell when a later
 * background poll has made its rescan verdict describe a screen that no longer exists.
 */
export function hardwareKey(s: ScanShape | null | undefined): string {
  return `${ports(s).slice().sort().join('|')}#${cameras(s).slice().sort().join('|')}`
}

export function rescanReport(
  before: ScanShape | null,
  outcome: RescanOutcome,
  opts?: { beforeAtMs?: number | null; nowMs?: number | null },
): RescanReport {
  // (3) It did not run.
  if (!outcome.ok) {
    const why = String(outcome.error ?? '').trim() || 'the request failed'
    if (!before) {
      return { tone: 'bad', stale: true, text: `⚠ rescan failed: ${why} — nothing has been scanned yet, so the lists below are empty for that reason, not because this machine has no hardware.` }
    }
    const age = ageWords(opts?.beforeAtMs, opts?.nowMs)
    return {
      tone: 'bad',
      stale: true,
      text: `⚠ rescan failed: ${why} — the lists below are the PREVIOUS scan${age}, not what is plugged in now.`,
    }
  }

  const after = outcome.after
  const pAfter = ports(after)
  const cAfter = cameras(after)

  // (2) It ran and found nothing.
  if (!pAfter.length && !cAfter.length) {
    const blocked = after?.camera_problem?.kind
      ? ' Cameras are blocked on this machine, so the camera count says nothing about what is connected — see the camera notice below.'
      : ''
    return {
      tone: 'warn',
      stale: false,
      text: `scan completed: this machine reports no serial ports and no cameras.${blocked} If an arm is plugged in, check the cable and its power, then rescan.`,
    }
  }

  // The FIRST successful scan has nothing to compare against: reporting its
  // contents as "+2 serial ports appeared" would describe an event that never
  // happened. It found what was already there.
  if (before == null) {
    return {
      tone: 'ok',
      stale: false,
      text: `scan completed — found: ${plural(pAfter.length, 'serial port')}, ${plural(cAfter.length, 'camera')}.`,
    }
  }

  const pBefore = ports(before)
  const cBefore = cameras(before)
  const parts = [
    ...delta('serial port', pAfter.filter(d => !pBefore.includes(d)), pBefore.filter(d => !pAfter.includes(d))),
    ...delta('camera', cAfter.filter(d => !cBefore.includes(d)), cBefore.filter(d => !cAfter.includes(d))),
  ]

  // (1) It ran and nothing changed. The counts are the proof the click landed —
  // without them an unchanged list is indistinguishable from a dead button.
  if (!parts.length) {
    return {
      tone: 'ok',
      stale: false,
      text: `scan completed — unchanged: ${plural(pAfter.length, 'serial port')}, ${plural(cAfter.length, 'camera')}.`,
    }
  }

  return {
    tone: 'ok',
    stale: false,
    text: `scan completed: ${parts.join(', ')} — now ${plural(pAfter.length, 'serial port')}, ${plural(cAfter.length, 'camera')}.`,
  }
}
