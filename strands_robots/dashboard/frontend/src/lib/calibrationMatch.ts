/**
 * Does the typed calibration id exist on this machine? The spawn form's "Calibration id" is
 * free text with a prose warning under it - "must match the one used by lerobot-calibrate, or
 * the joint limits will be wrong" - and nothing ever checked.
 */

import type { CalibrationEntry } from './calibration'

export type CalibrationVerdict = {
  kind: 'none' | 'match' | 'suggest' | 'unknown' | 'unchecked'
  /** what to show the operator, already in plain words */
  note: string
  /** an id worth one tap, when we are confident enough to name one */
  suggestion?: string
  /** true when the consequence is wrong joint limits on a real arm */
  warn: boolean
}

/**
 * Case and surrounding whitespace are typing noise: lerobot's own id lookup is a filename
 * match, so `Follower_Arm` finds `follower_arm` on a case-insensitive filesystem, but
 * `follower-arm` is a DIFFERENT FILE and must never read as a match - saying "matches" there
 * is the same lie in a new place.
 */
const norm = (s: string) => s.trim().toLowerCase()
/** Looser, for SUGGESTING only: separators are where the typo lives. */
const loose = (s: string) => norm(s).replace(/[\s_-]+/g, '')

/**
 * Which SIDE of the pair a calibration was recorded for. lerobot stores them in
 * `<root>/robots/<type>/<id>.json` and `<root>/teleoperators/<type>/<id>.json`, and a robot in
 * real mode can only load the first: it is `robots/` or it is not loadable, whatever the name
 * says.
 */
const isRobotSide = (e: CalibrationEntry): boolean => {
  const t = norm(e.deviceType || '')
  return !t || t.startsWith('robot')
}

/** Entries whose model plausibly belongs to the selected robot family. */
function familyMatches(entry: CalibrationEntry, family: string): boolean {
  const f = loose(family)
  const m = loose(entry.model || '')
  if (!f || !m) return true // nothing to contradict
  // 'so101' should accept 'so101_follower' / 'so101_leader', and vice versa.
  return m.includes(f) || f.includes(m)
}

export function calibrationVerdict(
  typed: string,
  entries: CalibrationEntry[] | null | undefined,
  family = '',
): CalibrationVerdict {
  const id = (typed ?? '').trim()

  if (entries === null || entries === undefined) {
    // The list has not arrived (or the API failed). Silence is honest here: a
    // guess would either accuse a correct id or bless a wrong one.
    return {
      kind: 'unchecked',
      warn: false,
      note: id
        ? 'the calibration files could not be read, so this id was not checked'
        : '',
    }
  }

  const known = entries.filter(e => (e.id || '').trim() !== '')

  if (!id) {
    return {
      kind: 'none',
      warn: true,
      note:
        'no id: the arm starts uncalibrated and reports raw servo counts, so its joint limits will be wrong' +
        (known.length ? ` — calibrations on this machine: ${known.map(e => e.id).join(', ')}` : ''),
    }
  }

  const pick = (from: CalibrationEntry[]) =>
    from.find(e => e.id === id) ?? from.find(e => norm(e.id) === norm(id))
  const robotSide = known.filter(isRobotSide)
  const exact = pick(robotSide) ?? pick(known)
  if (exact && !isRobotSide(exact)) {
    const usable = robotSide.filter(e => familyMatches(e, family)).map(e => e.id)
    return {
      kind: 'match',
      warn: true,
      // No one-tap suggestion here on purpose: choosing WHICH of the robot-side files this arm
      // should load is a decision about a physical arm's limits, not a typo fix.
      note:
        `${exact.id} was calibrated as a teleoperator (${exact.deviceType}/${exact.model}), and a ` +
        'robot in real mode loads robots/{type}/{id}.json — lerobot will refuse with "has no ' +
        'calibration registered" and the arm will report presence with no joints' +
        (usable.length
          ? ` — ids calibrated as robots here: ${usable.join(', ')}`
          : ' — nothing on this machine is calibrated as a robot for this family yet'),
    }
  }
  if (exact) {
    if (exact.unreadable) {
      return {
        kind: 'match',
        warn: true,
        note: `${exact.id} exists but its file could not be read — lerobot will fail to load it`,
      }
    }
    const wrongFamily = family && !familyMatches(exact, family)
    if (wrongFamily) {
      return {
        kind: 'match',
        warn: true,
        // A real mismatch: same name, different robot. Wrong limits, silently.
        note: `${exact.id} was calibrated for ${exact.model}, not ${family} — the joint limits would come from the wrong arm`,
      }
    }
    const detail = [exact.model, exact.motors ? `${exact.motors} motors` : ''].filter(Boolean).join(', ')
    return {
      kind: 'match',
      warn: false,
      note: detail ? `matches ${exact.id} (${detail})` : `matches ${exact.id}`,
    }
  }

  // A near miss is the case that actually happens: underscore vs hyphen, case,
  // a trailing space. Name it so the operator does not have to spot it.
  const near = known.find(e => loose(e.id) === loose(id))
    ?? known.find(e => loose(e.id).startsWith(loose(id)) || loose(id).startsWith(loose(e.id)))
  if (near) {
    return {
      kind: 'suggest',
      warn: true,
      suggestion: near.id,
      note: `no calibration named "${id}" — did you mean ${near.id}?`,
    }
  }

  if (!known.length) {
    return {
      kind: 'unknown',
      warn: true,
      note: `no calibration files exist on this machine, so "${id}" will not load — run lerobot-calibrate first, or spawn now and calibrate after`,
    }
  }

  return {
    kind: 'unknown',
    warn: true,
    note: `no calibration named "${id}" on this machine (found: ${known.map(e => e.id).join(', ')}) — the arm would report raw servo counts`,
  }
}
