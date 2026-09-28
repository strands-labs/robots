/**
 * The record form's golden path is "pick arm → type task → record" — but the dataset
 * name is a second hand-invented field standing in the way. The task sentence already
 * says what is being taught, so a name is derived from it and offered as one click.
 * Same law as outputDirSuggest: never offered over the operator's own text, and the
 * existing nameVerdict rail re-judges whatever lands in the field.
 */
import { freeVariant, type KnownDataset } from './datasetName'

/** the task sentence as a dataset-name slug: first few meaningful words, kebab-cased */
function slugFromTask(task: string): string | null {
  const words = (task ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]+/g, ' ')
    .split(/[\s_]+/)
    .filter(Boolean)
  if (words.length === 0) return null
  // enough words to be recognisable, few enough to stay a name
  return words.slice(0, 4).join('-')
}

export function suggestDatasetName(
  task: string,
  currentValue: string,
  known: KnownDataset[] | null,
): string | null {
  // the field is the operator's once they typed into it
  if ((currentValue ?? '').trim()) return null
  const slug = slugFromTask(task)
  if (!slug) return null
  // taken locally? offer the next free "-N" variant instead of a collision
  const taken = (known ?? []).some(d => d.local !== false && (d.repo_id ?? '').trim() === slug)
  if (!taken) return slug
  return freeVariant(slug, known ?? [])
}
