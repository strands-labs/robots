/** When the SERVER says it is pacing this tile, what should the tile do about it? */

export interface PacingNotice {
  /** the frame rate the server said it is serving us at, when it named one */
  fps: number | null
  /** what the tile shows, in the operator's words */
  note: string
}

/** Extract the number out of the server's sentence ("...pacing it at 2 fps until...") */
export function pacedFps(message: string): number | null {
  const m = /at\s+([0-9]+(?:\.[0-9]+)?)\s*fps/i.exec(message)
  if (!m) return null
  const v = Number(m[1])
  return Number.isFinite(v) && v > 0 ? v : null
}

/** A `camera_error` payload -> a pacing notice, or null when it is a REAL error. */
export function pacingFromNotice(ev: unknown): PacingNotice | null {
  if (!ev || typeof ev !== 'object') return null
  const e = ev as { type?: unknown; throttled?: unknown; error?: unknown }
  if (e.type !== 'camera_error' || e.throttled !== true) return null
  const message = typeof e.error === 'string' ? e.error : ''
  const fps = pacedFps(message)
  return {
    fps,
    note: fps === null
      ? 'the server is pacing this stream while it settles'
      : `paced by the server at ${fps} fps — the link was being saturated`,
  }
}

/**
 * The rate this tile should ask for next, given what it knows about itself and what the server
 * has told it.
 */
export function nextRequestedFps(own: number | null, paced: number | null): number | null {
  const caps = [own, paced].filter((v): v is number => typeof v === 'number' && v > 0)
  return caps.length ? Math.min(...caps) : null
}
