/** What the Jobs list may claim about a history it might not have. */

export interface JobsLedgerInput {
  /** how many job rows the API returned */
  count: number
  /** the API's sentence about the LEDGER file, if it could not be read */
  problem?: string | null
}

export interface JobsLedgerVerdict {
  /** the line to render, or null when the list needs no explanation */
  text: string | null
  tone: 'info' | 'warn'
  /** true when rows may be missing — the caller must not present the list as complete */
  partial: boolean
}

export function jobsLedgerNotice({ count, problem }: JobsLedgerInput): JobsLedgerVerdict {
  const trimmed = (problem ?? '').trim()
  if (trimmed) {
    // The API's sentence already says what happened, where the bad file went and that running jobs
    // are unaffected.
    const lead = count > 0
      ? 'Some earlier runs may be missing from this list'
      : 'This list is empty because the history could not be read, not because nothing ran'
    return { text: `⚠ ${lead} — ${trimmed}`, tone: 'warn', partial: true }
  }
  if (count === 0) {
    return { text: 'No training jobs yet.', tone: 'info', partial: false }
  }
  return { text: null, tone: 'info', partial: false }
}
