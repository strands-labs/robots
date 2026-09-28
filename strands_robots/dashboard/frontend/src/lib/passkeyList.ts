/** The keys to this dashboard, visible and revocable. */
import { agoText } from './cameraFreshness'

export interface Credential { id: string; name?: string | null; created?: number | string | null }

export interface PasskeyRow {
  id: string
  label: string
  /** '' when the server never recorded a creation time — silence, not a guessed date. */
  when: string
  revocable: boolean
  /** Why not, when it is not. Shown as the disabled button's reason. */
  reason: string
}

/** Seconds → the same phrasing the camera tiles use, or '' if there is nothing to say. */
function whenText(created: Credential['created'], nowMs: number): string {
  if (created === null || created === undefined || created === '') return ''
  const n = typeof created === 'number' ? created : Number(created)
  if (!Number.isFinite(n) || n <= 0) return ''
  // The store writes epoch SECONDS; a value that looks like milliseconds is still meant as a date,
  // and rendering it as "57000 years ago" would be the dashboard's mistake, not the store's.
  const ms = n > 1e12 ? n : n * 1000
  const secs = (nowMs - ms) / 1000
  if (secs < 0) return 'just now'
  return `added ${agoText(secs)}`
}

export function passkeyRows(creds: Credential[] | null | undefined, nowMs = Date.now()): PasskeyRow[] {
  const list = Array.isArray(creds) ? creds.filter(c => c && typeof c.id === 'string' && c.id) : []
  const last = list.length <= 1
  return list.map(c => ({
    id: c.id,
    label: (c.name ?? '').trim() || 'passkey',
    when: whenText(c.created, nowMs),
    revocable: !last,
    reason: last
      ? 'this is the only key to this dashboard — enroll another device first, or removing it would re-open the setup flow to anyone who can reach this page'
      : '',
  }))
}

/** What the operator is told when a revoke fails. */
export function revokeRefusal(status: number, detail: string): string {
  const text = (detail || '').trim()
  if (status === 409) return text || 'cannot remove the last passkey — enroll another first'
  if (status === 404) return 'that passkey is already gone — reload to see the current list'
  if (status === 401 || status === 403) return 'your session expired — reload and sign in again'
  return text || 'could not remove that passkey'
}

/**
 * The sentence above the list. An EMPTY list means auth is not protecting this dashboard at
 * all, which is a different and much louder fact than "you have no extra keys".
 */
export function passkeySummary(rows: PasskeyRow[], authRequired: boolean): string {
  if (rows.length === 0) {
    return authRequired
      ? 'no passkey is enrolled, yet this dashboard demands one — nobody can sign in until a device is enrolled'
      : 'no passkey is enrolled: anyone who can reach this page can use this dashboard'
  }
  if (rows.length === 1) return '1 device can sign in to this dashboard'
  return `${rows.length} devices can sign in to this dashboard`
}
