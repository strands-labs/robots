/** What a camera tile is actually showing. */
export type CamKind =
  | 'connecting' | 'live' | 'stalled' | 'silent' | 'waiting'
  | 'busy' | 'unauthorized' | 'error' | 'closed' | 'retrying'

export interface CamStatus {
  kind: CamKind
  /** short headline, e.g. "stalled" */
  title: string
  /** the fact behind it, e.g. "last frame 4s ago" */
  detail: string
  /** frames arrived recently: the image is worth showing at full strength */
  live: boolean
  /** an old frame is on screen and must be visibly marked as stale */
  frozen: boolean
}

/** No frame for this long and the picture is no longer the present. */
export const STALL_MS = 2500

/** macOS/Linux capture backends all have their own way of saying "taken". */
const BUSY = /(in use|busy|already open|cannot open|could not open|-11852|EBUSY|Resource temporarily unavailable)/i
const DENIED = /(unauthorized|not permitted|permission|denied|forbidden|TCC)/i

/** No frame for this long and the peer's own last frame is history, not a warm-up. */
export const PUBLISH_FRESH_MS = 15_000

/** How old the peer's own CAPTURE time may be while pixels still count as the present. */
export const CAPTURE_STALE_MS = 10_000

/** A duration a human reads at a glance. */
export function ageText(ms: number): string {
  if (!isFinite(ms) || ms < 0) return 'unknown'
  if (ms < 1000) return '<1s'
  if (ms < 90_000) return `${Math.round(ms / 1000)}s`
  if (ms < 5_400_000) return `${Math.round(ms / 60_000)}m`
  if (ms < 172_800_000) return `${(ms / 3_600_000).toFixed(1).replace(/\.0$/, '')}h`
  return `${Math.round(ms / 86_400_000)}d`
}

/**
 * `publishedAt` reaches us from the mesh snapshot, where python writes unix SECONDS, while
 * `now` is `Date.now()` in milliseconds.
 */
export function publishedAtMs(publishedAt?: number): number | undefined {
  if (publishedAt === undefined || publishedAt === null || !isFinite(publishedAt)) return undefined
  if (publishedAt <= 0) return undefined
  return publishedAt < 1e12 ? publishedAt * 1000 : publishedAt
}

export function classifyCamera(input: {
  now: number
  conn: 'connecting' | 'open' | 'closed'
  frames: number
  lastFrameAt?: number
  error?: string | null
  /** the peer says it has published frames (from the presence snapshot) */
  publishedAt?: number
  /** when a reconnect is scheduled, and which attempt it will be */
  retryInMs?: number
  attempt?: number
  stallMs?: number
  publishFreshMs?: number
  captureStaleMs?: number
}): CamStatus {
  const { now, conn, frames, lastFrameAt, error, publishedAt } = input
  const stallMs = input.stallMs ?? STALL_MS
  const age = lastFrameAt === undefined ? undefined : now - lastFrameAt
  const hadFrames = frames > 0 && age !== undefined
  const secs = (ms: number) => (ms < 1000 ? '<1s' : `${Math.round(ms / 1000)}s`)
  const pubMs = publishedAtMs(publishedAt)
  const pubAge = pubMs === undefined ? undefined : now - pubMs

  if (error && DENIED.test(error)) {
    return { kind: 'unauthorized', title: 'not permitted', detail: 'this session may not read camera frames', live: false, frozen: hadFrames }
  }
  if (error && BUSY.test(error)) {
    return { kind: 'busy', title: 'camera busy', detail: 'another app is holding this device', live: false, frozen: hadFrames }
  }
  if (error) {
    return { kind: 'error', title: 'no image', detail: error, live: false, frozen: hadFrames }
  }
  // Two independent clocks on the same picture: when it ARRIVED here, and when the peer says it
  // was CAPTURED.
  const captureAge = pubAge !== undefined && pubAge >= 0 ? pubAge : undefined
  const captureStale =
    captureAge !== undefined && captureAge > (input.captureStaleMs ?? CAPTURE_STALE_MS)

  // A stall outranks a healthy socket: the connection being fine is exactly
  // what makes a frozen frame convincing.
  if (hadFrames && (age! > stallMs || captureStale)) {
    // Whichever fact explains the staleness leads.
    const detail = captureStale && age! <= stallMs
      ? `the peer says it captured this ${ageText(captureAge!)} ago - it arrived here ${ageText(age!)} ago as a replay of its last frame, not a new one`
      : captureStale
        ? `last frame ${ageText(age!)} ago, and the peer says it captured it ${ageText(captureAge!)} ago`
        : `last frame ${ageText(age!)} ago`
    return {
      kind: 'stalled',
      title: captureStale && age! <= stallMs ? 'stale frame' : 'stalled',
      detail,
      live: false,
      frozen: true,
    }
  }
  if (hadFrames) return { kind: 'live', title: 'live', detail: '', live: true, frozen: false }
  if (conn === 'closed') {
    if (input.retryInMs !== undefined) {
      return {
        kind: 'retrying', title: 'reconnecting',
        detail: `in ${secs(input.retryInMs)}${input.attempt ? ` (attempt ${input.attempt})` : ''}`,
        live: false, frozen: false,
      }
    }
    return { kind: 'closed', title: 'disconnected', detail: 'stream closed', live: false, frozen: false }
  }
  if (conn === 'open') {
    if (pubAge !== undefined && pubAge > (input.publishFreshMs ?? PUBLISH_FRESH_MS)) {
      return {
        kind: 'silent', title: 'no frames',
        // Says WHERE it stopped: the stream is fine, the camera at the other end
        // is not, so the next step is that robot's log rather than this page.
        detail: `the peer's last frame is ${ageText(pubAge)} old - the camera stopped there, not in transit`,
        live: false, frozen: false,
      }
    }
    return pubAge !== undefined
      ? { kind: 'waiting', title: 'waiting', detail: `peer published ${ageText(pubAge)} ago, none arrived here yet`, live: false, frozen: false }
      : { kind: 'silent', title: 'no frames', detail: 'stream open, camera sending nothing', live: false, frozen: false }
  }
  return { kind: 'connecting', title: 'connecting', detail: 'opening the stream', live: false, frozen: false }
}
