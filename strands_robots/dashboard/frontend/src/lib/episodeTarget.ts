/** How many episodes the operator actually asked for — and whether we understood them. */
export interface EpisodeTarget {
  /** the number to send, only meaningful when `problem` is null */
  value: number
  /** why this input cannot be used, in words for the operator — null when it is fine */
  problem: string | null
  /** a change we DID make and must therefore admit (e.g. 3.7 → 3) */
  note: string | null
}

export const EPISODE_MAX = 500

export function episodeTarget(raw: string): EpisodeTarget {
  const text = (raw ?? '').trim()
  if (!text) return { value: 0, problem: 'how many episodes? enter a number', note: null }
  // Number('') is 0 and Number('12 ') is 12; both are handled above.
  const n = Number(text)
  if (!Number.isFinite(n)) return { value: 0, problem: `“${text}” is not a number`, note: null }
  if (n <= 0) {
    return { value: 0, problem: n === 0 ? 'zero episodes would record nothing' : 'that is a negative number of episodes', note: null }
  }
  if (n > EPISODE_MAX) {
    return { value: 0, problem: `${n} episodes is more than this screen will start (max ${EPISODE_MAX}) — record in batches`, note: null }
  }
  if (!Number.isInteger(n)) {
    const floored = Math.floor(n)
    return { value: floored, problem: null, note: `recording ${floored} episodes — ${text} is not a whole number` }
  }
  return { value: n, problem: null, note: null }
}
