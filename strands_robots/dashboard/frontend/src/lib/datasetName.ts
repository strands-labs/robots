/** Is the dataset name the operator is typing already taken? */
export type KnownDataset = { repo_id?: string; root?: string; total_episodes?: number; local?: boolean }

/** Local rows only: a Hub dataset of the same name is not what a local recording would collide with. */
function localNames(known: KnownDataset[]): Map<string, number | undefined> {
  const out = new Map<string, number | undefined>()
  for (const d of known) {
    if (d.local === false) continue
    const id = (d.repo_id ?? '').trim()
    if (id) out.set(id, typeof d.total_episodes === 'number' ? d.total_episodes : undefined)
  }
  return out
}

/** The next free name in the "-N" family: cubes -> cubes-2 -> cubes-3. */
export function freeVariant(name: string, known: KnownDataset[]): string | null {
  const taken = localNames(known)
  const base = name.trim()
  if (!base) return null
  const m = base.match(/^(.*?)-(\d+)$/)
  const stem = m ? m[1] : base
  let n = m ? Number(m[2]) : 1
  for (let i = 0; i < 200; i += 1) {
    n += 1
    const candidate = `${stem}-${n}`
    if (!taken.has(candidate)) return candidate
  }
  return null
}

/** What to say about this name, or null to say nothing. */
export function nameVerdict(
  name: string,
  known: KnownDataset[] | null,
): { message: string; suggestion: string | null } | null {
  const id = (name ?? '').trim()
  if (!id || !known) return null
  const taken = localNames(known)
  if (!taken.has(id)) return null
  const episodes = taken.get(id)
  const what = typeof episodes === 'number' && episodes > 0
    // The number is the whole point: 40 episodes is an afternoon of hand-guiding, and the operator
    // is the only one who knows whether this is the same take or a name they forgot they used.
    ? `already exists with ${episodes} episode(s)`
    : 'already exists on disk (no episodes recorded — probably an interrupted session)'
  return {
    message: `“${id}” ${what}. Recording refuses to reuse a dataset directory, so this will be turned away when you press start.`,
    suggestion: freeVariant(id, known),
  }
}
