/** The one-line account of what a servo board was last spawned as, and the trap inside it. */
import { calibrationVerdict } from './calibrationMatch'
import type { CalibrationEntry } from './calibration'

export interface RememberedSpawn {
  peer_id: string
  robot_name?: string | null
  mode?: string | null
  cameras: string[]
  robot_id?: string
  saved_at?: number | null
}

/** Does a name claim the OTHER role? Only the two words count — an index like "arm-2" is not evidence. */
export function nameClaimsOtherRole(name: string | null | undefined, role: string): boolean {
  const n = (name ?? '').trim().toLowerCase()
  const r = (role ?? '').trim().toLowerCase()
  if (!n || (r !== 'follower' && r !== 'leader')) return false
  const other = r === 'follower' ? 'leader' : 'follower'
  return n.includes(other) && !n.includes(r)
}

export interface RememberedLine {
  /** "so101-arm-1 — so101, real, cameras top + wrist" */
  summary: string
  /** the calibration id, when one is remembered */
  calibrationId?: string
  warning?: string
  /**
   * present when the remembered calibration ID would not load: the respawn is not merely
   * mislabelled, it will reproduce a known failure.
   */
  idProblem?: string
}

export function rememberedLine(
  r: RememberedSpawn | null | undefined,
  facts: {
    role?: string | null
    role_volts?: number | null
    /** the machine's calibration files; undefined/null = not read, and then nothing is claimed */
    calibrations?: CalibrationEntry[] | null
  } = {},
): RememberedLine | null {
  if (!r || !r.peer_id) return null
  const role = (facts.role ?? '').trim().toLowerCase()
  const bits: string[] = []
  if (r.robot_name) bits.push(String(r.robot_name))
  if (r.mode) bits.push(String(r.mode))
  // Camera NAMES: the saved indices are what macOS renumbers between reboots, so printing them here
  // would be the confidently-stale kind of detail this dashboard keeps deleting.
  bits.push(r.cameras.length ? `cameras ${r.cameras.join(' + ')}` : 'no cameras')
  const line: RememberedLine = {
    summary: `${r.peer_id}${bits.length ? ' — ' + bits.join(', ') : ''}`,
  }
  if (r.robot_id) line.calibrationId = r.robot_id

  const badId = nameClaimsOtherRole(r.robot_id, role)
  const badPeer = nameClaimsOtherRole(r.peer_id, role)
  if (badId || badPeer) {
    const volts = facts.role_volts != null ? `${facts.role_volts}V` : 'its measured voltage'
    const named = badId ? `the calibration id is named "${r.robot_id}"` : `this peer is named "${r.peer_id}"`
    line.warning =
      `${named} while this bus measures ${volts} = ${role}. The measurement is the fact and the ` +
      `name is what is wrong — reuse the memory anyway (the calibration file lives under that id), ` +
      `but do not let the name convince you this is the other arm`
  }

  // Whether the remembered id can actually be LOADED, asked of the same rule the spawn form uses
  // so one id cannot get two verdicts on two screens.
  if (r.robot_id && facts.calibrations != null) {
    const v = calibrationVerdict(r.robot_id, facts.calibrations, r.robot_name ?? '')
    if (v.warn) {
      line.idProblem = `${v.note} — spawning this memory as it stands repeats that failure`
    }
  }
  return line
}
