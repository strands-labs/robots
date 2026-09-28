/**
 * Newest training run first — and, more importantly, the five that get POLLED. TrainingTab
 * reversed the ledger (`jobs.slice().reverse()`) and then polled `jobs.slice(0, 5)`.
 */

export interface OrderableJob {
  /** epoch seconds recorded at submit; absent on rows written before the field existed */
  submitted_at?: number
}

/** True when the value is a usable epoch-second timestamp. */
function knownTime(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v > 0
}

/**
 * Newest first. Rows that cannot say when they started keep their (reversed) file order and
 * follow the ones that can.
 */
export function orderJobsNewestFirst<T extends OrderableJob>(jobs: readonly T[]): T[] {
  const timed: T[] = []
  const untimed: T[] = []
  for (const j of jobs) (knownTime(j?.submitted_at) ? timed : untimed).push(j)
  // Sort is stable in every engine we target, so two runs submitted within the same
  // second keep the order the ledger recorded them in rather than swapping around on
  // each refresh.
  timed.sort((a, b) => (b.submitted_at as number) - (a.submitted_at as number))
  untimed.reverse()
  return [...timed, ...untimed]
}
