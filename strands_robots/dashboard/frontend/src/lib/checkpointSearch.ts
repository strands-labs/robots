/**
 * What the checkpoint type-ahead may claim about an empty result — and which answer it is
 * allowed to show at all.
 */

export interface EmptyNoteInput {
  /** What the operator typed (may be empty: focus triggers a search). */
  query: string
  /** Backend's reason the Hub half did not answer, if any. */
  hubProblem?: string | null
}

/** The honest sentence for "the search returned nothing". */
export function emptyNote({ query, hubProblem }: EmptyNoteInput): string {
  const q = String(query ?? '').trim()
  const named = q ? `“${q}”` : 'that'
  const problem = String(hubProblem ?? '').trim()
  if (problem) {
    // Scope the claim to what was actually consulted, and repeat WHY here: this
    // line is the one being read, and the warning above it may have scrolled.
    return q
      ? `no checkpoint already on this machine matches ${named} — and the Hub was not searched (${problem}), so this is not "it does not exist".`
      : `nothing in this machine's local cache — and the Hub was not searched (${problem}), so the catalogue is unknown from here.`
  }
  return q
    ? `no checkpoints match ${named} (local cache + Hub).`
    : 'type part of a checkpoint name — local cache and the Hub are both searched.'
}

/** True when a response may be rendered: only the newest request speaks. */
export { isLatestRequest as isCurrent } from './requestOrder'
