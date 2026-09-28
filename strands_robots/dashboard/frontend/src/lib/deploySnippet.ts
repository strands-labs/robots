/** A filename safe on every OS the operator might carry this to, derived from the server's own. */
export function safeFilename(name: string | null | undefined, peerId?: string | null): string {
  const raw = (name ?? '').trim() || `${(peerId ?? 'robot').trim() || 'robot'}.py`
  // Peer ids legitimately contain `__` (parent__child) and may contain a slash or a colon from a
  // hand-written config; a slash in a download name silently becomes a DIRECTORY the browser
  // will not create, so the download fails with no message at all.
  const base = raw.replace(/[/\\:*?"<>|\s]+/g, '-').replace(/^[.-]+/, '')
  const stem = base.endsWith('.py') ? base.slice(0, -3) : base
  return `${stem || 'robot'}.py`
}

/**
 * What the operator is told when the snippet cannot be produced. The server's 422 for a serial
 * with no remembered profile is the common case and it has a remedy, so it must not read as a
 * bug.
 */
export function snippetRefusal(status: number, detail: string): string {
  const text = (detail || '').trim()
  if (status === 404 || /no profile remembered/i.test(text)) {
    return 'nothing is remembered for this board yet — spawn it once and the dashboard can write the file'
  }
  if (status === 422) return text || 'the saved payload is not enough to write a snippet'
  if (status === 401 || status === 403) return 'your session expired — reload and sign in again'
  return text || 'could not write the snippet'
}

/** The server's own note when the snippet went out with NO hub address, else null. */
export function hubAddressMissing(code: string | null | undefined): string | null {
  if (!code) return null
  // Presence of the real line is the proof an address made it in - looking for the note alone would
  // keep prompting after a successful override, since the note stays for other reasons.
  if (/^\s*os\.environ\.setdefault\("ZENOH_CONNECT"/m.test(code)) return null
  const note = code.match(/^#\s*NOTE:\s*(.+?)\.?\s*$/m)
  return note
    ? note[1]
    : 'this file carries no hub address, so the peer it starts will connect to nothing'
}

export interface HubHostDraft {
  /** the address to send, or null when it cannot be one */
  host: string | null
  /** why it was rejected — '' when accepted */
  why: string
}

/** Tidy what the operator typed into something sendable, or say why it is not an address. */
export function cleanHubHost(input: string | null | undefined): HubHostDraft {
  const raw = (input ?? '').trim()
  if (!raw) return { host: null, why: 'type the address this dashboard is reachable at from the other machine' }
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) {
    return { host: null, why: 'paste a host, not a URL — no http:// and no path' }
  }
  const host = raw.replace(/\/+$/, '')
  if (/\s/.test(host)) return { host: null, why: 'an address has no spaces in it' }
  // host or host:port — a bare port, an empty port and a non-numeric port are all typos worth
  // catching here, because the snippet would otherwise be generated around them silently.
  const m = host.match(/^([^:]+)(?::([^:]*))?$/)
  if (!m || !m[1]) return { host: null, why: 'that is not a host name or IP address' }
  if (m[2] !== undefined && !/^\d{1,5}$/.test(m[2])) {
    return { host: null, why: 'the part after ":" must be a port number' }
  }
  return { host, why: '' }
}
