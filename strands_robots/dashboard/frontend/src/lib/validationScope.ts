/**
 * What a "✓ resolves" verdict actually applies to. The validate button POSTs the provider plus
 * the config CURRENTLY in the form and shows "✓ lerobot_local resolves" in green.
 */

export interface ValidatedInput {
  provider: string
  config: Record<string, unknown>
}

export interface ValidationScope {
  /** the verdict still describes what is in the form */
  applies: boolean
  /** which keys moved since it was taken ('provider' included when it changed) */
  changed: string[]
  /** sentence for the banner, '' while the verdict still applies */
  note: string
}

/** Stable rendering of one value, so key order and number formatting cannot fake a change. */
function norm(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') {
    try {
      return JSON.stringify(v, Object.keys(v as object).sort())
    } catch {
      return String(v)
    }
  }
  return String(v)
}

export function changedKeys(a: Record<string, unknown>, b: Record<string, unknown>): string[] {
  const keys = new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})])
  const out: string[] = []
  for (const k of keys) {
    // An absent key and an empty one are the same input to the backend, which
    // skips blanks - so clearing an already-empty box is not a change.
    if (norm(a?.[k]) !== norm(b?.[k])) out.push(k)
  }
  return out.sort()
}

export function validationScope(
  validated: ValidatedInput | null | undefined,
  current: ValidatedInput,
): ValidationScope {
  if (!validated) return { applies: true, changed: [], note: '' }
  const changed = changedKeys(validated.config ?? {}, current.config ?? {})
  if (validated.provider !== current.provider) changed.unshift('provider')
  if (changed.length === 0) return { applies: true, changed: [], note: '' }
  const shown = changed.slice(0, 3).join(', ')
  const more = changed.length > 3 ? ` +${changed.length - 3} more` : ''
  return {
    applies: false,
    changed,
    // Names WHAT moved: "something changed" would send them hunting.
    note: `this verdict was taken before ${shown}${more} changed — it does not describe the form as it is now. Validate again.`,
  }
}
