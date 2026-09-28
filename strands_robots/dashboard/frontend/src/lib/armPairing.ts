/** Which arm is the leader and which is the follower — decided from what was MEASURED off each servo bus, not from what the peer is called. */
export interface RoleCandidate {
  peer_id: string
  role?: string | null
  role_volts?: number | null
}

export interface ArmPair {
  leader: string
  follower: string
  /** How each slot was decided — shown to the operator, never inferred by them. */
  basis: 'measured' | 'named' | 'none'
  note?: string
}

const NAMED_LEADER = /(^|[^a-z])leader([^a-z]|$)/i
const NAMED_FOLLOWER = /(^|[^a-z])follower([^a-z]|$)/i

export function measured(candidates: RoleCandidate[], role: string): RoleCandidate[] {
  return candidates.filter(c => c.role === role)
}

/**
 * Pick the pair. Measurement wins; an explicit name is a weak second; an index in a name
 * ("arm-2") is NOT evidence and is never used.
 */
export function pairArms(candidates: RoleCandidate[]): ArmPair {
  const leaders = measured(candidates, 'leader')
  const followers = measured(candidates, 'follower')

  // The happy path: the hardware answered for both slots.
  if (leaders.length === 1 && followers.length === 1) {
    return { leader: leaders[0].peer_id, follower: followers[0].peer_id, basis: 'measured' }
  }

  // Exactly one side is UNAMBIGUOUS. Fill that slot and only that slot: guessing the other from
  // a name is how the pair got inverted in the first place.
  if (leaders.length === 1 && followers.length !== 1) {
    return {
      leader: leaders[0].peer_id, follower: '', basis: 'measured',
      note: followers.length === 0
        ? `${leaders[0].peer_id} measured as the leader; no arm has measured as a 12V follower yet — ` +
          `pick the follower yourself, or measure it on the devices screen`
        : `${leaders[0].peer_id} measured as the leader, but ${followers.length} arms measured as ` +
          `followers — which one it drives is your call, so check the volts next to each name`,
    }
  }
  if (followers.length === 1 && leaders.length !== 1) {
    return {
      leader: '', follower: followers[0].peer_id, basis: 'measured',
      note: leaders.length === 0
        ? `${followers[0].peer_id} measured as the follower; no arm has measured as a 7.4V leader yet ` +
          `(an unpowered arm reads 5.5V on the USB rail) — pick the leader yourself, or measure it ` +
          `on the devices screen`
        : `${followers[0].peer_id} measured as the follower, but ${leaders.length} arms measured as ` +
          `leaders — which one drives is your call, so check the volts next to each name`,
    }
  }

  // Both sides ambiguous, or one side ambiguous and the other silent. A real situation (two followers
  // on the bench) and not the dashboard's job to break the tie silently.
  if (leaders.length > 1 || followers.length > 1) {
    const parts: string[] = []
    if (leaders.length > 1) parts.push(`${leaders.length} arms measured as the leader`)
    if (followers.length > 1) {
      parts.push(parts.length
        ? `${followers.length} as the follower`
        : `${followers.length} arms measured as the follower`)
    }
    return {
      leader: '', follower: '', basis: 'none',
      note: `${parts.join(' and ')}, so which one drives is your call — ` +
            `check the volts next to each name`,
    }
  }

  const named = {
    leader: candidates.find(c => NAMED_LEADER.test(c.peer_id))?.peer_id ?? '',
    follower: candidates.find(c => NAMED_FOLLOWER.test(c.peer_id))?.peer_id ?? '',
  }
  if (named.leader && named.follower && named.leader !== named.follower) {
    return {
      ...named, basis: 'named',
      note: 'paired from the peer names — nobody has measured these buses, so the names are ' +
            'being taken at their word',
    }
  }
  // Only ONE name states a role.
  if (named.leader !== named.follower && (named.leader || named.follower)) {
    const stated = named.leader ? 'leader' : 'follower'
    return {
      leader: named.leader, follower: named.follower, basis: 'named',
      note: `only ${named.leader || named.follower} states a role in its name, so it fills the ` +
            `${stated} slot — nobody has measured these buses, and the other arm is left to you ` +
            `(the leader is the lighter 7.4V arm)`,
    }
  }

  return {
    leader: '', follower: '', basis: 'none',
    note: candidates.length
      ? 'no arm has been measured, and the names do not say which is which — the leader is the ' +
        'lighter 7.4V arm, or measure both on the devices screen'
      : 'no arms on the mesh',
  }
}

/** The label an option carries, so the volts travel with the name. */
export function roleLabel(c: RoleCandidate): string {
  if (!c.role) return `${c.peer_id} — role not measured`
  const v = typeof c.role_volts === 'number' ? ` · ${c.role_volts}V` : ''
  return `${c.peer_id} — ${c.role}${v}`
}

/**
 * The contradiction check: the operator has chosen a slot that the hardware says is the other
 * thing.
 */
export function contradiction(
  candidates: RoleCandidate[], slot: 'leader' | 'follower', chosen: string,
): string | null {
  if (!chosen) return null
  const c = candidates.find(x => x.peer_id === chosen)
  if (!c?.role) return null
  if (c.role === slot) return null
  const v = typeof c.role_volts === 'number' ? `${c.role_volts}V` : 'its bus'
  if (c.role === 'unpowered') {
    return `${chosen} read ${v} — that is the USB logic rail, so its power supply is off. ` +
           `It can report positions but it cannot hold or mirror anything.`
  }
  if (c.role === 'mixed') {
    return `${chosen} reported inconsistent voltages across its servos — that is a wiring fault, ` +
           `not a role. Fix it before recording an episode with it.`
  }
  return `${chosen} measured ${v} — it is the ${c.role}, not the ${slot}. ` +
         (slot === 'leader'
           ? 'You would be hand-moving a torqued 12V arm while the 7.4V one tries to mirror it.'
           : 'The arm being recorded should be the 12V one that mirrors your hand.')
}
