/** the camera reconfigure editor's pure half. */
export interface CamRow {
  name: string
  /** OpenCV index ("0") or a device path ("/dev/video1") — as typed. */
  indexOrPath: string
  fps: string
  width: string
  height: string
}

export interface CamConfig {
  [name: string]: { index_or_path: number | string; fps?: number; width?: number; height?: number }
}

export function rowsFromConfig(config: CamConfig | null | undefined): CamRow[] {
  return Object.entries(config ?? {}).map(([name, c]) => ({
    name,
    indexOrPath: String(c.index_or_path ?? ''),
    fps: c.fps != null ? String(c.fps) : '',
    width: c.width != null ? String(c.width) : '',
    height: c.height != null ? String(c.height) : '',
  }))
}

/** "0" is an index, "/dev/video1" (or anything non-numeric) is a path. */
export function parseIndexOrPath(raw: string): number | string | null {
  const t = raw.trim()
  if (!t) return null
  if (/^\d+$/.test(t)) return Number(t)
  return t
}

const BOUNDS: Record<'fps' | 'width' | 'height', [number, number]> = {
  fps: [1, 240],
  width: [16, 7680],
  height: [16, 4320],
}

function intField(row: CamRow, field: 'fps' | 'width' | 'height'): { value?: number; error?: string } {
  const raw = row[field].trim()
  if (!raw) return {}
  if (!/^\d+$/.test(raw)) return { error: `${row.name || '(unnamed)'}: ${field} must be a whole number` }
  const v = Number(raw)
  const [lo, hi] = BOUNDS[field]
  if (v < lo || v > hi) return { error: `${row.name || '(unnamed)'}: ${field}=${v} is outside ${lo}..${hi}` }
  return { value: v }
}

/**
 * Rows -> backend payload. `{ cameras: null }` when every row was removed — detaching all
 * cameras is a legal, deliberate config, not an error.
 */
export function configFromRows(rows: CamRow[]): { cameras: CamConfig | null; error?: string } {
  const live = rows.filter(r => r.name.trim() || r.indexOrPath.trim() || r.fps.trim() || r.width.trim() || r.height.trim())
  if (live.length === 0) return { cameras: null }
  const out: CamConfig = {}
  for (const row of live) {
    const name = row.name.trim()
    if (!name) return { cameras: null, error: 'every camera needs a name (top / wrist / main…)' }
    if (out[name]) return { cameras: null, error: `two cameras are both named "${name}"` }
    const iop = parseIndexOrPath(row.indexOrPath)
    if (iop === null) return { cameras: null, error: `${name}: needs an index (0, 1…) or a device path` }
    const entry: CamConfig[string] = { index_or_path: iop }
    for (const field of ['fps', 'width', 'height'] as const) {
      const { value, error } = intField(row, field)
      if (error) return { cameras: null, error }
      if (value != null) entry[field] = value
    }
    // Width and height describe one rectangle: half of it is a typo, and the
    // driver "helpfully" filling the other half hides which half was meant.
    if ((entry.width == null) !== (entry.height == null)) {
      return { cameras: null, error: `${name}: give both width and height, or neither (driver default)` }
    }
    out[name] = entry
  }
  return { cameras: out }
}

/**
 * Which row (and field) the sheet should focus when opened FROM a camera on the robot
 * surface: an add lands on the new row's name, a named camera lands on its fps.
 */
export function focusTarget(
  rows: CamRow[], focusCam: string | null, adding: boolean,
): { index: number; field: 'name' | 'fps' } | null {
  if (adding) return rows.length ? { index: rows.length - 1, field: 'name' } : null
  if (!focusCam) return null
  const index = rows.findIndex(r => r.name === focusCam)
  return index >= 0 ? { index, field: 'fps' } : null
}

/** What the confirm sheet must say — the cost is a respawn, named plainly. */
export function applySummary(rows: CamRow[], peerId: string): string {
  const { cameras } = configFromRows(rows)
  const n = cameras ? Object.keys(cameras).length : 0
  const what = n === 0 ? 'detach every camera from' : `apply ${n} camera${n === 1 ? '' : 's'} to`
  return `This will ${what} ${peerId} by restarting it — its streams (and any running task) stop during the restart.`
}

/** The default publish rate every spawned child inherits (device_manager's
 *  `STRANDS_MESH_CAMERA_HZ` setdefault). Named here so the sheet can be honest
 *  even before /api/config has answered. */
export const DEFAULT_MESH_CAMERA_HZ = 5

/**
 * What the fps field actually buys — the sentence that stops a false bug report. `fps` is the
 * CAMERA's capture rate, handed to lerobot's OpenCVCameraConfig.
 */
export function previewRateNote(fps: number | null | undefined, cameraHz?: number | null): string | null {
  const publish = cameraHz == null || !Number.isFinite(cameraHz) || cameraHz <= 0
    ? DEFAULT_MESH_CAMERA_HZ
    : cameraHz
  const capture = fps == null || !Number.isFinite(fps) || fps <= 0 ? null : fps
  if (capture == null || publish >= capture) return null
  return `${capture} fps is the camera's capture rate — this dashboard receives ${publish}/s, `
    + `the mesh publish rate (Settings › mesh › camera_hz). The recording on disk is not capped by it.`
}
