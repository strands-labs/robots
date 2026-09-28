/**
 * What an e-stop may claim when it targeted NOTHING. The sheet said `No live peers were on the
 * mesh.` — the most dangerous sentence in this dashboard, because it is read one second after
 * someone hit the stop button while watching an arm move.
 */

export type EstopReach = { headline: string; detail: string; cutPower: boolean }

export function estopNothingTargeted(opts: { staleSkipped?: string[] }): EstopReach {
  const stale = (opts.staleSkipped ?? []).filter(Boolean)
  if (stale.length > 0) {
    return {
      headline: 'nothing was stopped — every peer was unreachable',
      detail:
        `no stop was delivered: ${stale.join(', ')} had no heartbeat, so ${stale.length === 1 ? 'it was' : 'they were'} ` +
        `skipped rather than stopped. A peer with no heartbeat can still be moving — losing telemetry is not ` +
        `stopping. If anything is moving, cut power at the supply.`,
      cutPower: true,
    }
  }
  return {
    headline: 'nothing was stopped — this view had no live peer to target',
    detail:
      'no stop was delivered, because this dashboard sees no live peer. That is a statement about this ' +
      'fleet view, not about the room: a robot started outside this dashboard, one whose presence has not ' +
      'arrived yet, or one already dropped as stale is invisible here and can still be moving. If anything ' +
      'is moving, cut power at the supply.',
    cutPower: true,
  }
}

/**
 * What the SIGNED rail may claim about the fleet. `lockout_engaged` is the issuer's own latch and
 * `Mesh.emergency_stop` sets it unconditionally, so it is true whether the stop reached every peer
 * or none — it may not be rendered as `peers refuse all commands until resumed`, which is a claim
 * about the room. The peer half lives in `responses_received` (REPLIES, not confirmed stops) and
 * `peers_not_stopped` (responders that reported they did NOT stop), and this decides between them.
 *
 * Returns null when no lockout is engaged, i.e. there is nothing to claim.
 */
export type SignedRailClaim = { headline: string; detail: string; cutPower: boolean }

export function signedRailClaim(opts: {
  lockoutEngaged?: boolean
  issuer?: string | null
  responsesReceived?: number | null
  peersNotStopped?: unknown[] | null
}): SignedRailClaim | null {
  if (opts.lockoutEngaged !== true) return null

  const by = typeof opts.issuer === 'string' && opts.issuer.trim() ? ` (signed by ${opts.issuer.trim()})` : ''
  const latched = `fleet LOCKOUT engaged${by}`
  const refused = 'A peer that received it refuses all commands until you resume with the override code.'
  const acks = typeof opts.responsesReceived === 'number' ? opts.responsesReceived : null
  // Coerce AFTER discarding blanks: String(null) is 'null', which is truthy, and would have
  // rendered a peer named "null" from a malformed payload.
  const notStopped = (opts.peersNotStopped ?? []).map(v => (v == null ? '' : String(v).trim())).filter(Boolean)

  // Worst first: a peer that answered "I did not stop" is a robot that may still be executing, and
  // it is the one case where the lockout being engaged is the least useful fact on screen.
  if (notStopped.length > 0) {
    return {
      headline: `${latched} — but ${notStopped.length} peer${notStopped.length === 1 ? '' : 's'} reported NOT stopping`,
      detail:
        `${notStopped.join(', ')} answered that ${notStopped.length === 1 ? 'it did' : 'they did'} not stop, so ` +
        `${notStopped.length === 1 ? 'that robot' : 'those robots'} may still be executing. The lockout stops the ` +
        `NEXT command; it does not halt motion already underway. If anything is moving, cut power at the supply.`,
      cutPower: true,
    }
  }

  if (acks === 0) {
    return {
      headline: `${latched} — but NO peer acknowledged`,
      detail:
        'the lockout is latched on this rail, and no peer replied to say it received the stop. Nothing here ' +
        'shows a robot was halted: a peer that never answered is not a peer that stopped. If anything is ' +
        'moving, cut power at the supply.',
      cutPower: true,
    }
  }

  if (acks === null) {
    // A server older than the accounting fields. Absent is not zero, and it is not "everyone" either.
    return {
      headline: `${latched} — peer acknowledgements unknown`,
      detail: `this server does not report which peers replied, so how far the stop reached cannot be shown here. ${refused}`,
      cutPower: false,
    }
  }

  return {
    headline: `${latched} — ${acks} peer${acks === 1 ? '' : 's'} acknowledged, none reported a failure to stop`,
    detail: `${refused} A peer that never replied is not counted here, so it is not covered by that count.`,
    cutPower: false,
  }
}
