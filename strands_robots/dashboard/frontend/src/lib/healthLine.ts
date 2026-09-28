/**
 * The one line an operator reads off a robot's health topic, and WHY it is one line.
 *
 * `SensorLoops._read_health` publishes whatever it could read — battery, temps, cpu_load,
 * disk_free_gb, mem_pct, uptime_s — and any of them may be missing on any robot. Rendering all
 * seven as labelled numbers makes the card a table nobody scans; the operator's question is
 * "is anything wrong with this robot", so the verdict names the worst finding and the rest stays
 * available as detail.
 *
 * On thresholds: the disk floors are the dashboard's OWN (disk_headroom.CRITICAL_MB / TIGHT_MB),
 * reused so a health line and the recording pre-flight cannot disagree about the same volume.
 * The battery floor has no upstream owner, so it is a display choice named once here rather than
 * a derived fact — and it only ever fires while DISCHARGING, because 8% on the charger is a
 * robot that is fine.
 */

/** Free space below this is "not enough to finish a recording" (disk_headroom.CRITICAL_MB). */
export const DISK_CRITICAL_GB = 2
/** Below this is "fine for a short session, tight for a long one" (disk_headroom.TIGHT_MB). */
export const DISK_TIGHT_GB = 12
/** A display threshold, not a derived one: see the module note. */
export const BATTERY_LOW_PCT = 10

export interface HealthReading {
  battery_pct?: number | null
  charging?: boolean | null
  temps?: Record<string, number> | null
  cpu_load?: number | null
  disk_free_gb?: number | null
  mem_pct?: number | null
  uptime_s?: number | null
  [key: string]: unknown
}

export interface HealthLine {
  /** the sentence */
  text: string
  /** 'ok' = nothing to do, 'attention' = a fact worth acting on, 'none' = nothing was reported */
  tone: 'ok' | 'attention' | 'none'
  /** the other readings, for a tooltip — never the whole sentence */
  detail: string | null
}

/** A finite number, or null. Guards every read: the SDK can publish a null battery. */
function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/** "3.5h" / "12m" — uptime in the brackets agoText uses, so the card reads consistently. */
export function uptimeText(seconds: number | null | undefined): string | null {
  const s = num(seconds)
  if (s === null || s < 0) return null
  if (s < 90) return `${Math.round(s)}s`
  if (s < 5400) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

/** The hottest reported sensor, as `[name, celsius]`, or null when none was reported. */
export function hottest(temps: HealthReading['temps']): [string, number] | null {
  if (!temps || typeof temps !== 'object') return null
  let best: [string, number] | null = null
  for (const [name, raw] of Object.entries(temps)) {
    const c = num(raw)
    if (c === null) continue
    if (best === null || c > best[1]) best = [name, c]
  }
  return best
}

/**
 * Read a health payload into one line.
 *
 * Args:
 *   health: The payload the bridge filed under `health`, if any.
 *
 * Returns:
 *   A line whose tone is 'none' when the robot published nothing readable — which is not a
 *   fault, and must not render as one.
 */
export function healthLine(health: HealthReading | null | undefined): HealthLine {
  if (health == null) return { text: 'no health topic on this robot', tone: 'none', detail: null }

  const battery = num(health.battery_pct)
  const charging = health.charging === true
  const disk = num(health.disk_free_gb)
  const mem = num(health.mem_pct)
  const load = num(health.cpu_load)
  const hot = hottest(health.temps)
  const up = uptimeText(health.uptime_s)

  // Everything that was reported, for the tooltip. Built before the verdict so the detail is the
  // same whichever finding wins.
  const extras: string[] = []
  if (battery !== null) extras.push(`battery ${battery.toFixed(0)}%${charging ? ' (charging)' : ''}`)
  if (disk !== null) extras.push(`${disk.toFixed(1)} GB free`)
  if (mem !== null) extras.push(`memory ${mem.toFixed(0)}%`)
  if (load !== null) extras.push(`load ${load.toFixed(2)}`)
  if (hot) extras.push(`${hot[0]} ${hot[1].toFixed(0)}C`)
  if (up) extras.push(`up ${up}`)
  const detail = extras.length > 0 ? extras.join(' \u00b7 ') : null

  // A payload can arrive carrying only a peer_id and a timestamp: `_read_health` sets has_data
  // for a battery dict with neither `pct` nor `percentage`, so battery_pct lands as null.
  if (detail === null) {
    return { text: 'health topic is arriving, but reported no readings', tone: 'none', detail: null }
  }

  // Worst finding wins, and only findings that are unambiguous get 'attention'.
  if (battery !== null && !charging && battery <= BATTERY_LOW_PCT) {
    return { text: `battery ${battery.toFixed(0)}% and discharging`, tone: 'attention', detail }
  }
  if (disk !== null && disk < DISK_CRITICAL_GB) {
    return {
      text: `only ${disk.toFixed(1)} GB free - not enough to finish a recording`,
      tone: 'attention',
      detail,
    }
  }
  if (disk !== null && disk < DISK_TIGHT_GB) {
    return {
      text: `${disk.toFixed(1)} GB free - tight for a long session`,
      tone: 'attention',
      detail,
    }
  }

  // Nothing is wrong, so the line reports the reading an operator asks for first.
  if (battery !== null) {
    return { text: `battery ${battery.toFixed(0)}%${charging ? ', charging' : ''}`, tone: 'ok', detail }
  }
  return { text: extras[0], tone: 'ok', detail }
}
