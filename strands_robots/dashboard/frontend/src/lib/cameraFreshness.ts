/**
 * Which of a robot's cameras have STOPPED publishing — asked before the operator commits an
 * hour of their life to a recording.
 */

/** Captures older than this mean the camera stopped, not that it is between frames. */
export const CAMERA_STOPPED_AGE_S = 120

export interface CameraMeta { t?: number | null }

export interface StoppedCamera { camera: string; ageS: number }

/** Seconds since this camera's last capture, or null when unknowable. */
export function captureAge(meta: CameraMeta | undefined | null, nowS: number): number | null {
  const t = meta?.t
  if (t == null || typeof t !== 'number' || !Number.isFinite(t) || t <= 0) return null
  const age = nowS - t
  if (age < 0) return null // clock skew, not a very fresh frame
  return age
}

/**
 * The cameras we have evidence have stopped, oldest first (the worst offender leads the
 * sentence the operator reads).
 */
export function stoppedCameras(
  cameras: Record<string, CameraMeta> | undefined | null,
  nowS: number,
  maxAgeS: number = CAMERA_STOPPED_AGE_S,
): StoppedCamera[] {
  const out: StoppedCamera[] = []
  for (const [camera, meta] of Object.entries(cameras ?? {})) {
    const ageS = captureAge(meta, nowS)
    if (ageS !== null && ageS > maxAgeS) out.push({ camera, ageS })
  }
  return out.sort((a, b) => b.ageS - a.ageS)
}

/** A duration with its own "ago" ALREADY IN IT — `12.3h ago`. */
export function agoText(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s ago`
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`
  return `${(seconds / 3600).toFixed(1)}h ago`
}

/** The warning shown next to the follower picker, or null when there is nothing to say. */
export function cameraWarning(
  stopped: StoppedCamera[],
  opts: { peerId?: string } = {},
): string | null {
  if (stopped.length === 0) return null
  const which = stopped.map(c => `${c.camera} (last frame ${agoText(c.ageS)})`).join(', ')
  const who = opts.peerId ? `${opts.peerId}: ` : ''
  const plural = stopped.length > 1 ? 'cameras have' : 'camera has'
  return (
    `${who}${stopped.length} ${plural} stopped publishing — ${which}. ` +
    `Recording now writes episodes with a frozen or missing image stream, ` +
    `which you would only notice at training time.`
  )
}

/**
 * The card's own line about cameras that have stopped — a pure function so the wording is
 * testable, and so the ONE mistake this line has already made cannot come back: agoText()
 * carries its own "ago", and the first version of this note appended another one.
 */
export function deadCameraNote(stopped: StoppedCamera[], totalCameras: number): string | null {
  if (stopped.length === 0 || totalCameras === 0) return null
  const which = stopped.map(c => `${c.camera} ${agoText(c.ageS)}`).join(', ')
  const noun = totalCameras > 1 ? 'cameras' : 'camera'
  return `${stopped.length} of ${totalCameras} ${noun} stopped — ${which}`
}
