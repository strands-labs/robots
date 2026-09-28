/** Should this page offer the local address instead of the tunnel? */

export type HintBody = {
  same_network?: boolean | null
  client_ip?: string | null
  lan_urls?: string[]
  why?: string
}

export type HintVerdict =
  | { show: false; reason: string }
  | { show: true; url: string; text: string; reason: string }

/** localStorage key: dismissal is remembered PER URL, not globally. */
export const DISMISS_KEY = 'lanHintDismissed'

export function lanHintVerdict(args: {
  body: HintBody | null
  origin: string
  dismissed: string[]
}): HintVerdict {
  const { body, origin, dismissed } = args
  if (!body) return { show: false, reason: 'no answer from the server (old build, or offline)' }
  // `null` means the server could not decide (IPv4 behind NAT). Silence, never a guess.
  if (body.same_network !== true) {
    return { show: false, reason: body.same_network === false ? 'viewer is on another network' : 'server could not tell' }
  }
  const urls = (body.lan_urls || []).filter(u => typeof u === 'string' && u.startsWith('http://'))
  if (!urls.length) return { show: false, reason: 'local, but the server named no address to offer' }

  // Already there: the whole point is to move OFF the tunnel, so a page loaded from the LAN
  // address must never be told to go to the LAN address.
  const here = urls.find(u => sameOrigin(u, origin))
  if (here) return { show: false, reason: 'this page is already served from the local address' }

  const url = urls[0]
  if (dismissed.includes(url)) return { show: false, reason: 'dismissed for this address' }
  return {
    show: true,
    url,
    text: `You are on the same network as this dashboard. ${url} skips the trip out to the internet and back — camera streams stall on that round trip.`,
    reason: 'local viewer coming in over the tunnel',
  }
}

/** Origin comparison that cannot throw on a malformed candidate. */
export function sameOrigin(a: string, b: string): boolean {
  try {
    const ua = new URL(a)
    const ub = new URL(b)
    return ua.hostname === ub.hostname && (ua.port || '80') === (ub.port || '80')
  } catch {
    return false
  }
}

/**
 * The href the hint actually navigates to, given the handoff answer. The LAN address is
 * plain http (no WebAuthn), so a minted token rides in the URL and AuthGate absorbs it;
 * anything else — old server (404), auth off (token:null), refusal — degrades to the
 * plain link, which is exactly today's behavior, never a broken one.
 */
export function handoffHref(url: string, res: { token?: string | null } | null | undefined): string {
  const token = res && typeof res.token === 'string' && res.token.trim() ? res.token.trim() : null
  if (!token) return url
  try {
    const u = new URL(url)
    u.searchParams.set('token', token)
    return u.toString()
  } catch {
    return url // a malformed candidate must not become a malformed navigation
  }
}

export function readDismissed(store: Pick<Storage, 'getItem'> | null): string[] {
  try {
    const raw = store?.getItem(DISMISS_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.filter(x => typeof x === 'string') : []
  } catch {
    return []  // a corrupt entry must not hide a working hint forever
  }
}
