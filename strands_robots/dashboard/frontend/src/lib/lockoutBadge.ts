/** What the LOCKOUT verdict from the fleet snapshot should look like on screen. */
export type LockoutVerdict = {
  state?: string | null
  reason?: string | null
  since?: number | null
  by?: string | null
} | null | undefined

export type LockoutBadge = {
  /** Badge text, or null to render nothing at all. */
  label: string | null
  /** 'locked' = loud, 'doubt' = quiet dashed marker. */
  tone: 'locked' | 'doubt' | null
  title: string
}

const AGO = (since: number | null | undefined, now: number): string => {
  if (!since) return ''
  const s = Math.max(0, Math.round(now / 1000 - since))
  if (s < 90) return ` ${s}s ago`
  if (s < 5400) return ` ${Math.round(s / 60)}m ago`
  return ` ${Math.round(s / 3600)}h ago`
}

export function lockoutBadge(v: LockoutVerdict, now = Date.now()): LockoutBadge {
  const state = v?.state ?? null
  if (state === 'locked') {
    const who = v?.by ? ` by ${v.by}` : ''
    return {
      label: 'e-stop locked',
      tone: 'locked',
      title:
        `This robot is refusing every command except status${who}${AGO(v?.since, now)}.\n` +
        (v?.reason ? `${v.reason}.\n` : '') +
        'Clearing it needs the operator override code (Safety > resume). Resuming moves nothing:\n' +
        'it only stops commands being refused.',
    }
  }
  if (state === 'unknown' && v?.since) {
    // A resume was broadcast, or this peer appeared after the e-stop. Both mean the
    // dashboard genuinely cannot tell — and saying so is the point.
    return {
      label: 'e-stop?',
      tone: 'doubt',
      title:
        `The e-stop state of this robot is unknown${AGO(v?.since, now)}.\n` +
        (v?.reason ? `${v.reason}.\n` : '') +
        'It will read as clear again as soon as it accepts a command a lockout would refuse.',
    }
  }
  // 'clear', a bare 'unknown' with nothing behind it, or a server too old to send the
  // field: render nothing. A green badge for "clear" would be one more thing to trust.
  return { label: null, tone: null, title: '' }
}

/** Fleet-wide line for the banner, or null when there is nothing to say. */
export function lockoutBanner(
  peers: { peer_id: string; lockout?: LockoutVerdict }[],
): { text: string; severity: 'bad' | 'warn' } | null {
  const locked = peers.filter(p => p.lockout?.state === 'locked')
  if (locked.length) {
    const by = locked.map(p => p.lockout?.by).find(Boolean)
    const who = by ? ` by ${by}` : ''
    const names = locked.map(p => p.peer_id).join(', ')
    return {
      severity: 'bad',
      text:
        locked.length === 1
          ? `${names} is e-stop locked${who} — it refuses every command except status. Safety > resume needs the override code.`
          : `${locked.length} robots are e-stop locked${who} (${names}) — they refuse every command except status. Safety > resume needs the override code.`,
    }
  }
  const doubt = peers.filter(p => p.lockout?.state === 'unknown' && p.lockout?.since)
  if (doubt.length) {
    return {
      severity: 'warn',
      text:
        `E-stop state unknown for ${doubt.length === 1 ? doubt[0].peer_id : `${doubt.length} robots`}: ` +
        'a resume was broadcast, but each robot verifies the override code itself, so this is not proof any of them cleared.',
    }
  }
  return null
}
