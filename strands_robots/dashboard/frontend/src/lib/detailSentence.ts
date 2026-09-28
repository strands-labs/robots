/** Longest a list of alternatives may run before it starts counting instead. */
const MAX_LISTED = 6

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

/** One alternative as a person would name it: a string, or an object's own name field. */
function nameOf(item: unknown): string | null {
  if (typeof item === 'string') return item.trim() || null
  if (typeof item === 'number' || typeof item === 'boolean') return String(item)
  if (isPlainObject(item)) {
    for (const k of ['name', 'peer_id', 'id', 'device_name', 'dataset', 'path']) {
      const v = item[k]
      if (typeof v === 'string' && v.trim()) return v.trim()
    }
  }
  return null
}

function listPhrase(key: string, items: readonly unknown[]): string | null {
  const names = items.map(nameOf).filter((n): n is string => !!n)
  const label = key.replace(/_/g, ' ')
  if (names.length === 0) {
    // Objects nobody can name: say how many there are rather than dumping them. The count is the
    // actionable part ("it exists 3 times"), and the shapes are still in HttpError.body for a report.
    return items.length ? `${label}: ${items.length}` : null
  }
  const shown = names.slice(0, MAX_LISTED).join(', ')
  return names.length > MAX_LISTED
    ? `${label}: ${shown} and ${names.length - MAX_LISTED} more`
    : `${label}: ${shown}`
}

/**
 * A FastAPI validation error is a LIST of {loc, msg}: "body -> fps" is a field, not prose.
 * Named by its last path segment, which is the field the operator actually typed into.
 */
function validationSentence(items: readonly unknown[]): string | null {
  const parts: string[] = []
  for (const item of items) {
    if (!isPlainObject(item)) return null
    const msg = typeof item.msg === 'string' ? item.msg : null
    if (!msg) return null
    const loc = Array.isArray(item.loc) ? item.loc.filter(x => typeof x === 'string' && x !== 'body') : []
    const field = loc.length ? String(loc[loc.length - 1]) : null
    parts.push(field ? `${field}: ${msg}` : msg)
  }
  return parts.length ? parts.join('; ') : null
}

/** The detail of an error response as one readable sentence. */
export function detailSentence(detail: unknown): string {
  if (typeof detail === 'string') return detail.trim()
  if (typeof detail === 'number' || typeof detail === 'boolean') return String(detail)
  if (detail === null || detail === undefined) return ''

  if (Array.isArray(detail)) {
    const strings = detail.map(nameOf).filter((n): n is string => !!n)
    if (strings.length === detail.length && strings.length) return strings.join('; ')
    return validationSentence(detail) ?? JSON.stringify(detail)
  }

  if (!isPlainObject(detail)) return JSON.stringify(detail)

  const head = ['error', 'message', 'reason']
    .map(k => detail[k])
    .find(v => typeof v === 'string' && v.trim()) as string | undefined
  // `detail.detail` is the SPECIFIC cause under a general error (auth's rp_id refusal writes both).
  const because = typeof detail.detail === 'string' && detail.detail.trim() ? detail.detail.trim() : null
  const hint = typeof detail.hint === 'string' && detail.hint.trim() ? detail.hint.trim() : null
  if (!head && !because && !hint) return JSON.stringify(detail)

  const lists: string[] = []
  for (const [k, v] of Object.entries(detail)) {
    if (k === 'error' || k === 'message' || k === 'reason' || k === 'detail' || k === 'hint') continue
    if (Array.isArray(v)) {
      const phrase = listPhrase(k, v)
      if (phrase) lists.push(phrase)
    }
  }

  let text = [head?.trim(), because].filter(Boolean).join(' — ')
  if (hint) text = text ? `${text} — ${hint}` : hint
  if (lists.length) text += ` (${lists.join('; ')})`
  return text
}
