/**
 * WHY a sensor row reads as it does — because "this robot has no lidar", "its lidar has not
 * spoken yet" and "its lidar stopped" are three different facts and only one of them is anybody's
 * problem.
 *
 * Two rails, the same shape jointAbsence uses:
 *
 *   1. DECLARATION. `presence.topics` is the peer's own list of the sensor topics it publishes
 *      (mesh.core builds it from the providers it found on the robot: `_pose`/`_slam_pose`/
 *      `_odom_pose` -> 'pose', `_imu` -> 'imu', `_battery` -> 'health', and so on). So a topic
 *      that is declared and silent is a REAL finding, and one that was never declared is simply
 *      not this robot's business.
 *   2. ARRIVAL. Whether a reading came, and how long ago.
 *
 * The rails disagree in the direction that matters: without the first, a declared-but-silent
 * lidar reads identically to a robot that has no lidar, which is the same false equivalence the
 * joint strip made when it called a live arm "no joint data on this peer".
 *
 * One measured detail shapes the whole strip: mesh.core appends 'health' to `topics`
 * UNCONDITIONALLY, because `_read_health` reads host stats (cpu_load, disk_free_gb, mem_pct,
 * uptime_s) that every machine has. So every peer declares and publishes health, an arm included
 * -- measured on a joints-only arm: `topics: ['health']`, and a payload carrying 86.7 GB free.
 * That is worth a line on an arm's card rather than noise: it is the disk a recording lands on.
 */
import { agoText } from './cameraFreshness'

/** Readings older than this are not "current" any more. Matches STATE_QUIET_S in jointAbsence:
 *  two different quiet windows would let a pose row contradict a joint row on one card. */
export const SENSOR_QUIET_S = 10

/** The sensor topics the mesh bridge forwards, in the order an operator reads them. The spellings
 *  are the ones `presence.topics` uses, so a declaration can be matched without translation. */
export const SENSOR_KINDS = ['health', 'pose', 'odom', 'imu', 'lidar'] as const
export type SensorKind = (typeof SENSOR_KINDS)[number]

/** A payload as the SDK wrote it. Only `t` is read here; the rest is the caller's to render. */
export interface SensorReading {
  t?: number | null
  [key: string]: unknown
}

export type SensorTone =
  /** arriving now */
  | 'live'
  /** it was arriving and stopped — a complaint, and it says how long */
  | 'stale'
  /** the peer says it publishes this and none has arrived yet — neutral, not a fault */
  | 'waiting'
  /** never declared, never seen: for most robots simply the truth */
  | 'absent'

export interface SensorVerdict {
  kind: SensorKind
  tone: SensorTone
  /** the row's sentence */
  text: string
  /** seconds since the reading, when there is a reading */
  ageS: number | null
}

/** Which of our sensor kinds this peer's presence document declares. */
export function declaredKinds(topics: string[] | null | undefined): SensorKind[] {
  if (!Array.isArray(topics)) return []
  return SENSOR_KINDS.filter(k => topics.includes(k))
}

/**
 * Read one sensor topic against both rails.
 *
 * Args:
 *   kind: Which topic this verdict is about.
 *   reading: The payload the bridge last filed for it, if any.
 *   nowS: Wall-clock seconds, so the caller owns the clock.
 *   declared: Whether `presence.topics` names this topic. An older peer sends no `topics` at
 *     all, in which case the caller passes false and absence stays the neutral verdict — never
 *     an accusation built on a field the robot did not send.
 *
 * Returns:
 *   A verdict whose tone is 'absent' or 'waiting' when nothing has arrived. Neither is a fault.
 */
export function sensorVerdict(
  kind: SensorKind,
  reading: SensorReading | null | undefined,
  nowS: number,
  declared = false,
): SensorVerdict {
  if (reading == null) {
    // The interesting case, and the one that would otherwise read as "no lidar": the peer says
    // it publishes this and nothing has come.
    if (declared) {
      return { kind, tone: 'waiting', text: 'declared, waiting for the first reading', ageS: null }
    }
    return { kind, tone: 'absent', text: 'not published by this robot', ageS: null }
  }
  const t = reading.t
  // A reading with no usable timestamp still PROVES the topic exists: it arrived.
  if (typeof t !== 'number' || !Number.isFinite(t) || t <= 0) {
    return { kind, tone: 'live', text: 'arriving (no timestamp in the payload)', ageS: null }
  }
  const ageS = nowS - t
  // Clock skew: a peer whose clock runs ahead must not be reported as quiet for -4s.
  if (ageS < 0) return { kind, tone: 'live', text: 'arriving', ageS: 0 }
  if (ageS <= SENSOR_QUIET_S) return { kind, tone: 'live', text: 'arriving', ageS }
  // How stale, not just "stale": the number is what tells an operator whether to wait or look.
  return { kind, tone: 'stale', text: `last reading ${agoText(ageS)}`, ageS }
}

/**
 * Which rows to draw: everything the peer declared, plus anything that arrived without being
 * declared (an older peer sends no `topics`, and a reading is still proof).
 */
export function rowsToShow(
  topics: string[] | null | undefined,
  sensors: Partial<Record<SensorKind, SensorReading | null | undefined>> | null | undefined,
): SensorKind[] {
  const declared = new Set(declaredKinds(topics))
  return SENSOR_KINDS.filter(k => declared.has(k) || sensors?.[k] != null)
}

/**
 * The strip's own one-line summary, or null when there is nothing to say at all.
 *
 * A peer with nothing declared and nothing arriving renders NOTHING rather than five rows saying
 * so: a permanent "no lidar" line on every card trains an operator to stop reading the strip.
 */
export function stripSummary(verdicts: SensorVerdict[]): { text: string; tone: SensorTone } | null {
  const shown = verdicts.filter(v => v.tone !== 'absent')
  if (shown.length === 0) return null
  const stale = shown.filter(v => v.tone === 'stale')
  if (stale.length > 0) {
    return { text: `${stale.map(v => v.kind).join(', ')} went quiet`, tone: 'stale' }
  }
  const live = shown.filter(v => v.tone === 'live')
  if (live.length === 0) {
    return { text: `${shown.map(v => v.kind).join(', ')} declared, nothing yet`, tone: 'waiting' }
  }
  const waiting = shown.length - live.length
  const tail = waiting > 0 ? `, ${waiting} not started` : ''
  return { text: `${live.length} sensor${live.length === 1 ? '' : 's'} arriving${tail}`, tone: 'live' }
}
