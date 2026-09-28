/**
 * Which query the dataset list on screen actually belongs to. The picker searches the Hub as
 * you type, and the copy under it is confident: "nothing here or on the Hub matches 'so10'".
 */

export interface DatasetHintInput {
  /** what is in the box right now */
  query: string
  shownQuery: string | null
  /** how many rows are visible */
  count: number
  /** the Hub half's failure, if it failed */
  problem?: string | null
  /** hf auth block from the API, when it says we are anonymous */
  anonymous?: boolean
  authDetail?: string | null
}

export interface DatasetHint {
  /** the main sentence under the picker, or null when the list speaks for itself */
  text: string | null
  tone: 'info' | 'warn' | 'pending'
  /** the extra "public results only" line, or null */
  auth: string | null
}

export function datasetHint(input: DatasetHintInput): DatasetHint {
  const q = input.query.trim()
  const shown = input.shownQuery === null ? null : input.shownQuery.trim()
  const stale = shown !== q

  // Anonymous Hub search is worth saying whenever the user is asking the Hub
  // something, because it changes what "no match" MEANS (gated repos are hidden).
  const auth = input.anonymous && q
    ? `Hub results are public only${input.authDetail ? ` (${input.authDetail})` : ''} — a private or gated dataset will look like “no match”.`
    : null

  if (input.problem && !stale) {
    return { text: `⚠ ${input.problem}`, tone: 'warn', auth }
  }
  // Results (or a verdict) that belong to a different question may not answer
  // this one. Say which state we are in instead of borrowing the old answer.
  if (stale) {
    return {
      text: q
        ? `searching for “${q}”…${input.count > 0 && shown ? ` — the rows below are still the results for “${shown}”` : ''}`
        : null,
      tone: 'pending',
      auth,
    }
  }
  if (input.count === 0 && q) {
    return {
      text: `nothing here or on the Hub matches “${q}” — the Hub answered, it simply has no match.`,
      tone: 'info',
      auth,
    }
  }
  return { text: null, tone: 'info', auth }
}

/**
 * Should this response be applied to the screen? The last request WINS, not the last response
 * — invisible in testing, because a fast local search always answers before a slow Hub one.
 */
export { isLatestRequest as isCurrentResponse } from './requestOrder'
