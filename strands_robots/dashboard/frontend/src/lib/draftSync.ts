/** The subset of settings that are plain text drafts, keyed as the drawer names them. */
export type Drafts = Record<string, string>

export interface SyncResult {
  /** What each field should hold after the reload. */
  next: Drafts
  /** Fields whose draft was preserved because the operator had touched it. */
  kept: string[]
  /** Touched fields whose server value ALSO changed — the human has to decide. */
  conflicts: string[]
  /** Fields quietly updated from the server (untouched by the operator). */
  adopted: string[]
}

/**
 * Merge a fresh server snapshot into the current drafts. @param current what the fields hold
 * right now @param lastServer the snapshot the fields were seeded from (what "untouched"
 * means) @param nextServer the snapshot that just arrived
 */
export function syncDrafts(current: Drafts, lastServer: Drafts, nextServer: Drafts): SyncResult {
  const next: Drafts = { ...current }
  const kept: string[] = []
  const conflicts: string[] = []
  const adopted: string[] = []

  for (const key of Object.keys(nextServer)) {
    const server = nextServer[key]
    const seeded = lastServer[key]
    const draft = current[key]

    // Nothing was ever seeded (first load): take the server's value.
    if (seeded === undefined) {
      next[key] = server
      if (draft !== undefined && draft !== server) kept.push(key), (next[key] = draft)
      else adopted.push(key)
      continue
    }

    const touched = draft !== undefined && draft !== seeded
    if (!touched) {
      if (server !== draft) adopted.push(key)
      next[key] = server
      continue
    }

    // Touched: the draft survives. Only the reason changes.
    next[key] = draft as string
    // The server may have moved TO the draft — that is this very save landing, not a conflict.
    if (server === draft) adopted.push(key)
    else if (server !== seeded) conflicts.push(key)
    else kept.push(key)
  }

  return { next, kept, conflicts, adopted }
}

/** Which fields differ from the server snapshot — the "unsaved" markers and the close guard. */
export function dirtyFields(current: Drafts, server: Drafts): string[] {
  return Object.keys(server)
    .filter(k => current[k] !== undefined && current[k] !== server[k])
    .sort()
}

/** Human sentence for the close guard. Empty string when there is nothing to lose. */
export function unsavedSummary(dirty: string[], labels: Record<string, string> = {}): string {
  if (!dirty.length) return ''
  const names = dirty.map(k => labels[k] ?? k)
  const list =
    names.length === 1
      ? names[0]
      : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
  return `Unsaved ${names.length === 1 ? 'change' : 'changes'} to ${list}`
}
