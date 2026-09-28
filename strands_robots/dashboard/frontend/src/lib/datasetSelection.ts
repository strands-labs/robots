/** R6: turning a picked dataset row into the ONE field training should receive. */

export interface DatasetRow {
  root?: string
  repo_id: string
  local?: boolean
  downloads?: number | null
  total_episodes?: number
  /** frames actually written; 0 alongside 0 episodes is the abandoned-session shape. */
  total_frames?: number | null
  robot_type?: string
  fps?: number
  usable?: boolean
  reason?: string
  problem?: string
  recording?: boolean
}

export interface DatasetSelection {
  dataset_root: string
  dataset_repo_id: string
  /** true when the trainer will download it before the first step */
  fromHub: boolean
  /** what the plan sentence should call it, or '' when nothing is picked */
  label: string
}

const EMPTY: DatasetSelection = { dataset_root: '', dataset_repo_id: '', fromHub: false, label: '' }

/** Stable option key for either kind — a Hub row has no path to key on. */
export function datasetKey(d: DatasetRow): string {
  return d.root ?? `hub:${d.repo_id}`
}

/** Resolve a <select> value back to the fields training accepts. */
export function selectDataset(rows: DatasetRow[], key: string): DatasetSelection {
  if (!key) return EMPTY
  const d = rows.find(r => datasetKey(r) === key)
  if (!d) return EMPTY
  if (d.root) {
    return {
      dataset_root: d.root,
      dataset_repo_id: '',
      fromHub: false,
      label: `${d.repo_id}${d.total_episodes ? ` (${d.total_episodes} eps)` : ''}`,
    }
  }
  return {
    dataset_root: '',
    dataset_repo_id: d.repo_id,
    fromHub: true,
    label: `${d.repo_id} — downloaded from the Hub at start`,
  }
}

/** The <select> value for the current form state (round-trips selectDataset). */
export function selectionKey(sel: { dataset_root: string; dataset_repo_id: string }): string {
  if (sel.dataset_root) return sel.dataset_root
  return sel.dataset_repo_id ? `hub:${sel.dataset_repo_id}` : ''
}

/** Can this row be replayed here? */
function noEpisodes(d: DatasetRow): boolean {
  if (d.usable !== undefined) return false
  return d.total_episodes === 0 || d.total_frames === 0
}

/** The sentence for a row whose own count says it is empty — the server's wording, minus the part only it can know. */
const EMPTY_ROW = '0 episodes. meta/info.json is written when a recording session OPENS, before the first '
  + 'episode is captured, so a directory like this is what an abandoned session leaves behind — not a dataset. '
  + 'Record into it, or delete it.'

export function replayable(d: DatasetRow): { ok: boolean; reason: string } {
  if (!d.root) return { ok: false, reason: 'on the Hub, not on this machine — training downloads it; replay needs it local' }
  if (d.usable === false) return { ok: false, reason: d.problem ?? 'this dataset has no episodes to replay' }
  if (noEpisodes(d)) return { ok: false, reason: EMPTY_ROW }
  // Deliberately says no episode NUMBER: the number lives in the box next to the button now, and
  // this sentence is composed with episodeChoice's in the tooltip — two claims about the index
  // would contradict.
  return { ok: true, reason: 'Replay in a live mesh sim — appears in the fleet grid' }
}

export function datasetMark(d: DatasetRow): { glyph: string; kind: 'recording' | 'problem' | 'ok' } {
  if (d.recording) return { glyph: '⏺ ', kind: 'recording' }
  if (d.usable === false) return { glyph: '⚠ ', kind: 'problem' }
  // Same evidence, same glyph: a row the buttons will refuse must not look normal in the list.
  if (noEpisodes(d)) return { glyph: '⚠ ', kind: 'problem' }
  return { glyph: '', kind: 'ok' }
}

export function trainable(d: DatasetRow | null): { ok: boolean; reason: string } {
  if (!d) return { ok: true, reason: '' }
  if (d.usable === false) return { ok: false, reason: d.problem ?? 'this dataset has no episodes to train on' }
  if (noEpisodes(d)) return { ok: false, reason: EMPTY_ROW }
  return { ok: true, reason: '' }
}

/** The picked row, or null when the form's selection is not in the current list. */
export function selectedRow(rows: DatasetRow[], sel: { dataset_root: string; dataset_repo_id: string }): DatasetRow | null {
  return rows.find(r => datasetKey(r) === selectionKey(sel)) ?? null
}

/**
 * Which episode a replay click should ask for — and whether the number the operator typed
 * exists.
 */
export function episodeChoice(
  d: DatasetRow,
  requested?: number | string | null,
): { ok: boolean; episode: number; reason: string; countKnown: boolean } {
  const total = typeof d.total_episodes === 'number' && Number.isFinite(d.total_episodes) ? d.total_episodes : null
  const countKnown = total !== null && total > 0
  const raw = typeof requested === 'string' ? requested.trim() : requested
  if (raw === undefined || raw === null || raw === '') {
    return { ok: true, episode: 0, reason: countKnown ? `episode 0 of ${total}` : 'episode 0', countKnown }
  }
  const n = Number(raw)
  if (!Number.isInteger(n)) {
    return { ok: false, episode: 0, reason: `“${raw}” is not a whole episode number`, countKnown }
  }
  if (n < 0) return { ok: false, episode: 0, reason: 'episode numbers start at 0', countKnown }
  if (total !== null && total <= 0) {
    return { ok: false, episode: 0, reason: 'this dataset records no episodes yet', countKnown }
  }
  if (total !== null && n >= total) {
    return {
      ok: false,
      episode: 0,
      reason: `this dataset has ${total} episode${total === 1 ? '' : 's'}, numbered 0–${total - 1}`,
      countKnown,
    }
  }
  return {
    ok: true,
    episode: n,
    reason: countKnown ? `episode ${n} of ${total}` : `episode ${n} (this server does not report a count)`,
    countKnown,
  }
}
