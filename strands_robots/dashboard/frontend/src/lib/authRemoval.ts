/** what removing the dashboard's auth token actually exposes, said before it happens. */
export interface AuthRemovalFacts {
  /** location.hostname of the page doing the removing. */
  host: string
  /** security.cors_origins as configured ('*' = any). */
  corsOrigins?: string | null
  /** How many peers the fleet currently shows, robots included. */
  peerCount?: number
}

export interface AuthRemovalWarning {
  /** 'exposed' when this very page proves off-box reachability. */
  severity: 'exposed' | 'local'
  /** The consequence, first line first. Every line is a fact, not an adjective. */
  lines: string[]
  /** Label for the button that goes through with it. */
  confirmLabel: string
}

const LOCAL = new Set(['localhost', '127.0.0.1', '::1', '[::1]', '0.0.0.0', ''])

export function isLocalHost(host: string): boolean {
  return LOCAL.has((host || '').trim().toLowerCase())
}

export function authRemovalWarning(facts: AuthRemovalFacts): AuthRemovalWarning {
  const host = (facts.host || '').trim()
  const remote = !isLocalHost(host)
  const lines: string[] = []

  if (remote) {
    // Not a guess: the page making this request arrived over that host.
    lines.push(
      `You are viewing this dashboard as ${host}, not localhost — so it is reachable from outside ` +
      `this machine, and removing the token opens it to everyone who can reach that address.`,
    )
  } else {
    lines.push(
      'Anyone who can reach this port gets full control — including from another machine on this ' +
      'network, or through any tunnel that forwards it.',
    )
  }

  const n = facts.peerCount ?? 0
  lines.push(
    n > 0
      ? `${n} robot${n === 1 ? '' : 's'} on this fleet can then be commanded, and motors moved, ` +
        'without a token.'
      : 'Any robot that joins this fleet can then be commanded, and its motors moved, without a token.',
  )

  if ((facts.corsOrigins ?? '').trim() === '*') {
    lines.push('CORS is set to * , so a web page on any site can call this API from a browser too.')
  }

  // The way back matters: an operator who knows the undo is one field away decides faster and is
  // less likely to leave it off "for now".
  lines.push('Setting a token again re-locks it immediately — this is reversible, but not automatic.')

  return {
    severity: remote ? 'exposed' : 'local',
    lines,
    confirmLabel: remote ? 'yes — leave it open to the network' : 'yes — remove the token',
  }
}
