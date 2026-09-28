/** episode labels on a dataset row — and the honest reason when there is nothing to show. */
export type LabelRow = {
  episode_index: number
  verdict: 'success' | 'failure' | null
  steps?: number | null
  quality: string | null
  failure_mode: string | null
  note: string | null
  disputes_verdict: boolean
  model: string | null
  annotatable: boolean
}

export type LabelView = {
  benchmark?: string | null
  episodes: LabelRow[]
  total_episodes?: number | null
  with_verdict: number
  labelled: number
  disputed: number
  can_annotate: boolean
  why: string
  sidecar_error?: string | null
}

/** Whether the labels panel can be opened for this row, and why not when it cannot. */
export function labelsGate(row: { root?: string | null; recording?: boolean } | null | undefined):
  { ok: boolean; reason: string } {
  if (!row) return { ok: false, reason: 'no dataset selected' }
  if (!row.root) {
    return { ok: false, reason: 'labels live in a sidecar next to the dataset on disk — download this Hub dataset first' }
  }
  if (row.recording) {
    // Not a refusal about permissions: mid-recording there is genuinely nothing judged yet, and the
    // sidecar is written after the verdicts are recorded.
    return { ok: false, reason: 'this dataset is being recorded right now — episodes are judged after the session' }
  }
  return { ok: true, reason: 'show the deterministic verdicts and judge annotations for each episode' }
}

/** One line for the panel header: what is known, in the server's own words when it matters. */
export function labelSummary(view: LabelView | null | undefined, error?: string | null):
  { text: string; tone: 'plain' | 'warn' } {
  if (error) return { text: `Labels could not be read — ${error}`, tone: 'warn' }
  if (!view) return { text: 'Reading labels…', tone: 'plain' }
  if (view.sidecar_error) return { text: `Labels may be damaged — ${view.why}`, tone: 'warn' }
  if (!view.episodes.length || !view.can_annotate) {
    // The server's sentence says which of four situations this is; a count cannot.
    return { text: view.why, tone: 'plain' }
  }
  const parts = [
    `${view.labelled}/${view.with_verdict} judged`,
    view.total_episodes != null ? `${view.total_episodes} episodes recorded` : null,
    view.benchmark ? `benchmark ${view.benchmark}` : null,
    view.disputed ? `${view.disputed} disputing the verdict` : null,
  ].filter(Boolean)
  return { text: parts.join(' · '), tone: view.disputed ? 'warn' : 'plain' }
}

/** How one episode reads in the list. Never invents a verdict for an unjudged episode. */
export function labelRowLine(row: LabelRow): { badge: string; detail: string; muted: boolean } {
  if (!row.annotatable) {
    return {
      badge: '—',
      detail: 'no deterministic verdict, so it cannot be annotated',
      muted: true,
    }
  }
  const badge = row.verdict === 'success' ? '✓' : row.verdict === 'failure' ? '✗' : '—'
  const bits = [
    row.quality ? `quality ${row.quality}` : 'awaiting a quality grade',
    row.failure_mode || null,
    row.disputes_verdict ? 'judge disputes this verdict' : null,
    row.note || null,
    row.model ? `by ${row.model}` : null,
  ].filter(Boolean)
  return { badge, detail: bits.join(' · '), muted: !row.quality }
}
