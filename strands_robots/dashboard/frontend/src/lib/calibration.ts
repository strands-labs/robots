/**
 * Parsers for the calibration API's text payloads. `GET /api/calibration` and
 * `/api/calibration/{name}` both answer `{status, text}` where `text` is the rich-markdown
 * block `lerobot_calibrate` prints for a human.
 */

export type CalibrationEntry = {
  /** 'robots' | 'teleoperators' — lowercased for the API's device_type param. */
  deviceType: string
  /** e.g. 'so101_follower' */
  model: string
  /** the calibration id, e.g. 'follower_arm' */
  id: string
  modified?: string
  sizeKb?: number
  motors?: number
  /** true when the tool could not read the file behind this id */
  unreadable: boolean
  /** set when the id ITSELF is the footprint of a bug — see idProblem */
  problem?: string
}

export type CalibrationList = {
  location?: string
  entries: CalibrationEntry[]
}

export type MotorCalibration = {
  name: string
  id?: string
  driveMode?: string
  homingOffset?: string
  rangeMin?: string
  rangeMax?: string
}

export type CalibrationDetail = {
  title?: string
  path?: string
  modified?: string
  size?: string
  motors: MotorCalibration[]
}

/** Strip the markdown emphasis the tool wraps almost every value in. */
function plain(s: string): string {
  return s.replace(/[*`]/g, '').trim()
}

export function idProblem(id: string): string | undefined {
  const bare = id.trim()
  if (!bare) return 'this calibration has no id — the file name is empty'
  // The stringified null of every language that could have written this: python None, JS
  // null/undefined, and the literal word. Case-insensitive because `none` is equally a bug here.
  if (/^(none|null|undefined|nan)$/i.test(bare)) {
    return `"${bare}" is a missing value that reached a file name — something was spawned without a `
      + 'robot id, so these joint limits belong to that accident, not to an arm. Recalibrate under a '
      + 'real id and delete this file.'
  }
  return undefined
}

/** Parse the `list` action's text into flat rows. */
export function parseCalibrationList(text: string): CalibrationList {
  const out: CalibrationList = { entries: [] }
  if (!text) return out

  let deviceType = ''
  let model = ''

  for (const raw of text.split('\n')) {
    const line = raw.trimEnd()

    const loc = /^Location:\s*(.+)$/.exec(line.trim())
    if (loc) { out.location = plain(loc[1]); continue }

    const type = /^##\s+(.+)$/.exec(line)
    if (type && !line.startsWith('###')) {
      // The heading is title-cased for display ('Teleoperators'); the API
      // parameter is not, so normalise here rather than at every call site.
      deviceType = plain(type[1]).toLowerCase()
      model = ''
      continue
    }

    const mod = /^###\s+(.+?)(?:\s+\(\d+\s+calibrations?\))?\s*$/.exec(line)
    if (mod) { model = plain(mod[1]); continue }

    const item = /^\s*-\s+`([^`]+)`\s*(?:\*\((.*)\)\*)?\s*$/.exec(line)
    if (item) {
      const id = item[1]
      const meta = (item[2] ?? '').trim()
      const entry: CalibrationEntry = {
        deviceType, model, id,
        unreadable: /error reading file/i.test(meta),
      }
      const bad = idProblem(id)
      if (bad) entry.problem = bad
      if (meta && !entry.unreadable) {
        const when = /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})/.exec(meta)
        if (when) entry.modified = when[1]
        const size = /([\d.]+)\s*KB/i.exec(meta)
        if (size) entry.sizeKb = Number(size[1])
        const motors = /(\d+)\s*motors?/i.exec(meta)
        if (motors) entry.motors = Number(motors[1])
      }
      out.entries.push(entry)
    }
  }

  return out
}

/** Parse the `view` action's text into a per-motor table. */
export function parseCalibrationDetail(text: string): CalibrationDetail {
  const out: CalibrationDetail = { motors: [] }
  if (!text) return out

  let current: MotorCalibration | null = null

  for (const raw of text.split('\n')) {
    const line = raw.trim()

    const title = /^\*\*Calibration Details:\s*(.+?)\*\*$/.exec(line)
    if (title) { out.title = plain(title[1]); continue }

    const path = /^\*\*Path:\*\*\s*(.+)$/.exec(line)
    if (path) { out.path = plain(path[1]); continue }

    const modified = /^\*\*Modified:\*\*\s*(.+)$/.exec(line)
    if (modified) { out.modified = plain(modified[1]); continue }

    const size = /^\*\*Size:\*\*\s*(.+)$/.exec(line)
    if (size) { out.size = plain(size[1]); continue }

    const motor = /^###\s+(.+)$/.exec(line)
    if (motor) {
      current = { name: plain(motor[1]) }
      out.motors.push(current)
      continue
    }

    if (!current) continue

    const field = /^-\s+\*\*([^:]+):\*\*\s*(.*)$/.exec(line)
    if (!field) continue
    const key = field[1].trim().toLowerCase()
    const value = plain(field[2])
    if (key === 'id') current.id = value
    else if (key === 'drive mode') current.driveMode = value
    else if (key === 'homing offset') current.homingOffset = value
    else if (key === 'range') {
      // "700 to 3400" — keep the halves apart so the table can align them.
      const range = /^(.*?)\s+to\s+(.*)$/.exec(value)
      if (range) { current.rangeMin = range[1].trim(); current.rangeMax = range[2].trim() }
      else current.rangeMin = value
    }
  }

  return out
}
