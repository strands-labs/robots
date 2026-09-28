/** val_episodes: hold out the LAST N episodes as a validation set. */

export interface Holdout {
  /** what to put in the submit body, or null to omit the key entirely */
  send: number | null
  /** why this value cannot be sent, or null */
  problem: string | null
  /** the sentence under the input: what WILL happen with this value */
  say: string
}

const NO_SPLIT =
  'empty: trains on every episode and reports training loss only — a validation split is what tells learning from memorising'

export function holdout(raw: string, episodeCount?: number | null): Holdout {
  const text = (raw ?? '').trim()
  if (!text) return { send: null, problem: null, say: NO_SPLIT }

  const n = Number(text)
  if (!Number.isFinite(n)) {
    return { send: null, problem: `“${text}” is not a number`, say: NO_SPLIT }
  }
  if (n <= 0) {
    return {
      send: null,
      problem: `a holdout of ${n} reserves no episodes at all — leave it empty to train on every episode`,
      say: NO_SPLIT,
    }
  }
  if (!Number.isInteger(n)) {
    return {
      send: null,
      problem: `${text} would reserve ${Math.ceil(n)} episodes, not ${Math.floor(n)} — enter a whole number`,
      say: NO_SPLIT,
    }
  }

  const known = typeof episodeCount === 'number' && Number.isFinite(episodeCount) && episodeCount > 0
  if (known && n >= (episodeCount as number)) {
    return {
      send: null,
      problem: `this dataset has ${episodeCount} episodes — holding out ${n} leaves ${Math.max(0, (episodeCount as number) - n)} to train on`,
      say: NO_SPLIT,
    }
  }

  const rest = known ? `${(episodeCount as number) - n} to train on, ` : ''
  return {
    send: n,
    problem: null,
    say: `holds out the last ${n} episode${n === 1 ? '' : 's'} for validation (${rest}so an eval loss is logged alongside the training loss)`,
  }
}
