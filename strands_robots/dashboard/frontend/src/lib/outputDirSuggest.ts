/**
 * The one hand-typed field on training's golden path is the output dir — a path the
 * operator must invent. This suggests one from the dataset they just picked, offered
 * as a one-click fill (never silently written: the field stays theirs, and the
 * existing output-dir verdict rail re-judges whatever lands in it).
 */

/** the name a human would recognise: last path segment, or the repo's own name */
function datasetName(sel: { dataset_root?: string; dataset_repo_id?: string }): string | null {
  const src = (sel.dataset_root ?? '').trim() || (sel.dataset_repo_id ?? '').trim()
  if (!src) return null
  const seg = src.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? ''
  // keep it a safe directory name; a dataset named only punctuation suggests nothing
  const clean = seg.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[_.]+|[_.]+$/g, '')
  return clean || null
}

export function suggestOutputDir(
  sel: { dataset_root?: string; dataset_repo_id?: string },
  currentValue: string,
): string | null {
  // the field is the operator's once they typed into it — never fight their text
  if ((currentValue ?? '').trim()) return null
  const name = datasetName(sel)
  if (!name) return null
  return `/tmp/train_${name}`
}
