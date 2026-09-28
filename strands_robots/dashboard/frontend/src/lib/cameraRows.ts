/**
 * The multi-camera half of the spawn form: rows of {name, index} the operator composed,
 * judged into the exact `cameras` mapping /api/devices/spawn expects — or a refusal in
 * words. One rule owns this so the form cannot invent a second, slightly different one.
 *
 * The shapes that already bit us, pinned here:
 * - each entry must be a MAPPING ({index_or_path: N, ...}) — a bare int is the live
 *   ValueError "Camera 'main' config must be a mapping ... got int: 3";
 * - names become lerobot config keys, so they must be identifier-shaped;
 * - the same index twice is one physical camera with two capture threads — it fails at
 *   spawn with a claim error, so it is refused before the button.
 */

export interface CameraRow {
  /** the config key the operator chose: main, wrist, top… */
  name: string
  /** camera index as typed/selected; '' = row not filled in yet */
  index: string
}

export interface CameraShared {
  fps?: number | null
  width?: number | null
  height?: number | null
}

export interface CamerasField {
  /** the spawn payload — null when no row is filled in (spawn without cameras) */
  value: Record<string, Record<string, number>> | null
  /** why this cannot be sent, for the operator; null when it is fine */
  problem: string | null
  /** sendable but suspicious — said out loud, never blocking (two rows on one index) */
  note: string | null
}

const NAME_SHAPE = /^[A-Za-z][A-Za-z0-9_]*$/
/** The dashboard's own bookkeeping keys (camera_liveness.ANNOTATION_KEYS). validate_cameras
 *  only checks these as OPTION keys, so the form must refuse them as NAMES itself. */
const RESERVED_NAMES = new Set(['device_name'])

export function camerasField(rows: CameraRow[], shared: CameraShared = {}): CamerasField {
  const filled = (rows ?? []).filter(r => (r.index ?? '').trim() !== '')
  if (filled.length === 0) return { value: null, problem: null, note: null }

  const seenNames = new Set<string>()
  const seenIndices = new Map<number, string>()
  const value: Record<string, Record<string, number>> = {}
  let note: string | null = null

  for (const row of filled) {
    const name = (row.name ?? '').trim()
    if (!name) {
      return { value: null, problem: 'every selected camera needs a name — main, wrist, top…', note: null }
    }
    if (!NAME_SHAPE.test(name)) {
      return {
        value: null,
        problem: `“${name}” cannot be a camera name — letters, digits and _ only, starting with a letter`,
        note: null,
      }
    }
    const key = name.toLowerCase()
    if (RESERVED_NAMES.has(key)) {
      return {
        value: null,
        problem: `“${name}” is the dashboard's own bookkeeping key, not a camera name — pick main, wrist, top…`,
        note: null,
      }
    }
    if (seenNames.has(key)) {
      return { value: null, problem: `two cameras named “${name}” — the second would overwrite the first`, note: null }
    }
    seenNames.add(key)

    const index = Number(row.index)
    if (!Number.isInteger(index) || index < 0) {
      return { value: null, problem: `“${row.index}” is not a camera index`, note: null }
    }
    const claimant = seenIndices.get(index)
    if (claimant !== undefined) {
      // A warning, not a block: sharing an index is almost always a mistake (the second
      // open usually fails at spawn), but the operator may know something we don't.
      note = `index ${index} is claimed by both “${claimant}” and “${name}” — the second open usually fails at spawn`
    }
    seenIndices.set(index, name)

    value[name] = {
      index_or_path: index,
      ...(shared.fps ? { fps: shared.fps } : {}),
      ...(shared.width ? { width: shared.width } : {}),
      ...(shared.height ? { height: shared.height } : {}),
    }
  }
  return { value, problem: null, note }
}
