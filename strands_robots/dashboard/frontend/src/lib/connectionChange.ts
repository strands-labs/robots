/** re-pointing the backend used to send the OLD host's token to the NEW host, silently. */
export interface ConnectionChange {
  /** the base the app is talking to now ('' = the origin that served the page) */
  currentBase: string
  /** the token currently held for it */
  currentToken: string
  /** what the operator typed, already normalised where possible ('' = this origin) */
  nextBase: string
  /** what is in the token field now */
  nextToken: string
  /** hostname of the page itself, for judging "this origin" and clear-text risk */
  pageHost?: string
}

export type ConnectionVerdict =
  | { kind: 'ok' }
  | { kind: 'unparseable'; detail: string }
  | {
      /** a token minted for one host is about to be sent to another */
      kind: 'token_follows_host'
      detail: string
      fromHost: string
      toHost: string
      /** the choice that keeps the secret where it belongs */
      alternative: string
    }
  | {
      /** the token would cross the network in clear text */
      kind: 'cleartext_token'
      detail: string
      toHost: string
      alternative: string
    }

const LOCAL = new Set(['localhost', '127.0.0.1', '::1', '[::1]', '0.0.0.0'])

function hostOf(base: string, pageHost = ''): string {
  const v = (base ?? '').trim()
  if (!v) return (pageHost || '').toLowerCase() // empty means "the origin that served this page"
  try {
    return new URL(/^[a-z]+:\/\//i.test(v) ? v : `http://${v}`).host.toLowerCase()
  } catch {
    return ''
  }
}

function isLocal(host: string): boolean {
  return LOCAL.has(host.replace(/:\d+$/, ''))
}

/**
 * Judge a pending "connect & reload". Only ONE thing is ever escalated: the token moving to a
 * host it was not given for.
 */
export function connectionChange(c: ConnectionChange): ConnectionVerdict {
  const nextRaw = (c.nextBase ?? '').trim()
  const nextToken = (c.nextToken ?? '').trim()
  const currentToken = (c.currentToken ?? '').trim()

  // A base that cannot be parsed is refused with the reason, BEFORE the reload: afterwards every
  // request fails against a URL the operator can no longer see, which reads like a dead backend.
  if (nextRaw && hostOf(nextRaw, c.pageHost) === '') {
    return {
      kind: 'unparseable',
      detail:
        `"${nextRaw}" is not an address this browser can dial. Use host:port or a full URL ` +
        `(https://robot.lan:8090); leave it empty to talk to the origin that served this page`,
    }
  }

  const from = hostOf(c.currentBase, c.pageHost)
  const to = hostOf(nextRaw, c.pageHost)
  const carryingOldToken = nextToken !== '' && nextToken === currentToken

  if (carryingOldToken && from !== to) {
    return {
      kind: 'token_follows_host',
      fromHost: from || '(this origin)',
      toHost: to || '(this origin)',
      detail:
        `The token in this browser was given for ${from || 'this origin'}, and connecting to ` +
        `${to || 'this origin'} will send it there — a credential for one machine handed to another. ` +
        `If that address is a typo or not the robot you think it is, the secret is gone.`,
      alternative: 'connect without a token',
    }
  }

  // Clear text is only worth raising when the token is actually leaving this machine: http to
  // localhost never touches a wire.
  if (nextToken !== '' && to && !isLocal(to)) {
    const scheme = /^https:\/\//i.test(nextRaw) ? 'https' : /^[a-z]+:\/\//i.test(nextRaw) ? 'http' : 'http'
    if (scheme === 'http') {
      return {
        kind: 'cleartext_token',
        toHost: to,
        detail:
          `http://${to} is not encrypted, so this token crosses the network in clear text — anyone ` +
          `on the path can read it and use it to move motors. https:// keeps it private.`,
        alternative: 'connect without a token',
      }
    }
  }

  return { kind: 'ok' }
}

/** Does this verdict need the operator's explicit go-ahead before connecting? */
export function needsConfirm(v: ConnectionVerdict): boolean {
  return v.kind === 'token_follows_host' || v.kind === 'cleartext_token'
}
