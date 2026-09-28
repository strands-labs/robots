/** Read-only view of what this arm’s teleop is actually doing. */
export type TeleopTone = 'ok' | 'warn' | 'idle'
export interface TeleopView {
  tone: TeleopTone
  headline: string
  detail?: string | null
  /** Present when the server's refusal is the widenable safety envelope: the operator is one consent
   *  away from working teleop, and ConsentSettings already renders this kind. Never widen it here. */
  consentKind?: 'teleop_degree_units'
  /** Is anything on the wire right now — the one fact a stop button should be enabled by. */
  streaming: boolean
}

/** `refusing` outranks everything, exactly as the server's own ordering does. */
const TONE: Record<string, TeleopTone> = {
  refusing: 'warn', unrouted: 'warn', silent: 'warn', stopped: 'idle', following: 'ok',
}

export function teleopView(payload: unknown): TeleopView | null {
  const health = (payload as { health?: unknown } | null | undefined)?.health
  if (!health || typeof health !== 'object') return null            // unasked, unreachable, or an old server
  const h = health as {
    receivers?: Record<string, { state?: string; headline?: string; detail?: string | null; refusal?: unknown }>
    publishers?: Record<string, { state?: string; headline?: string; detail?: string | null }>
    worst?: { state?: string; headline?: string; detail?: string | null; refusal?: unknown } | null
  }
  const receivers = h.receivers ?? {}
  const publishers = h.publishers ?? {}
  const pubs = Object.entries(publishers)
  const live = pubs.filter(([, p]) => p.state === 'publishing')

  const worst = h.worst ?? Object.values(receivers)[0] ?? null
  if (!worst && pubs.length === 0) {
    return { tone: 'idle', headline: 'no teleop on this arm', streaming: false,
      detail: 'it is neither following another arm nor publishing its own joints' }
  }
  if (!worst) {
    // Publishing only: this is a LEADER.
    const [name, p] = live[0] ?? pubs[0]
    return { tone: live.length ? 'ok' : 'idle', headline: `publishing ${name}: ${p.headline ?? p.state ?? 'unknown'}`,
      detail: p.detail ?? null, streaming: live.length > 0 }
  }
  const view: TeleopView = {
    tone: TONE[worst.state ?? ''] ?? 'warn',
    headline: worst.headline ?? worst.state ?? 'teleop state unknown',
    detail: worst.detail ?? null,
    // A receiver that is following counts as traffic too: frames are being applied to a real arm.
    streaming: live.length > 0 || worst.state === 'following' || worst.state === 'refusing',
  }
  if (worst.refusal) view.consentKind = 'teleop_degree_units'
  return view
}

export function stopVerdict(after: TeleopView | null): { ok: boolean; line: string } {
  if (!after) {
    return { ok: false, line: 'stop was sent, but the arm did not answer when asked again — nothing confirms it landed' }
  }
  if (after.streaming) {
    return { ok: false, line: `stop was sent, but frames are STILL on the wire: ${after.headline}` }
  }
  return { ok: true, line: `teleop stopped — ${after.headline}` }
}

export function startVerdict(after: TeleopView | null): { ok: boolean; line: string } {
  if (!after) {
    return { ok: false, line: 'start was sent, but the arm did not answer when asked again — nothing confirms frames are flowing' }
  }
  if (after.consentKind) {
    return { ok: false, line: `started, but every frame is being REFUSED: ${after.headline} — the bound is widened at settings › consent › ${after.consentKind}, deliberately and by you` }
  }
  if (after.streaming) return { ok: true, line: `teleop live — ${after.headline}` }
  return { ok: false, line: 'start was sent, but nothing is on the wire yet — a follower can take up to 45s to declare its subscriber, so ask again before assuming it failed' }
}
