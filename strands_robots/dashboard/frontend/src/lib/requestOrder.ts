/**
 * Which answer is allowed to reach the screen. Three places in this dashboard fire a request
 * per keystroke or per tick, and a response is not ordered with respect to its siblings.
 */

/** The newest request wins: only the answer whose seq is still the latest speaks. */
export function isLatestRequest(seq: number, latest: number): boolean {
  return seq === latest
}

/**
 * For per-key state (one job's status, one camera's frame): apply an answer only if it is
 * newer than what is already displayed.
 */
export function newerThanApplied(seq: number, applied: number | undefined): boolean {
  return applied === undefined || seq > applied
}
