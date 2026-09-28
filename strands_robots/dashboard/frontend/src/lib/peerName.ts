/** What should this robot be CALLED? */

/** Exactly `_PEER_ID_RE` in strands_robots/dashboard/device_manager.py. Kept in sync by a test. */
export const PEER_NAME_RE = /^[A-Za-z0-9._:-]{1,64}$/

export type PeerNameVerdict = {
  /** What to send as `peer_id`: null means "omit it, let the server generate one". */
  value: string | null
  /** A reason to refuse the submit, or null. */
  problem: string | null
  /** Something true worth saying that is not a refusal. */
  note: string | null
  /** A legal name derived from what they typed, when we can offer one. */
  suggestion: string | null
}

/** Turn anything typed into something the key space accepts, or null if nothing survives. */
export function sanitizePeerName(raw: string): string | null {
  const s = (raw ?? '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^A-Za-z0-9._:-]/g, '')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
  return s === '' ? null : s
}

/** The next free `-N` in a family, so a second left arm becomes left-arm-2 rather than a collision. */
function freeVariant(name: string, taken: Set<string>): string | null {
  const m = name.match(/^(.*?)-(\d+)$/)
  const stem = m ? m[1] : name
  let n = m ? Number(m[2]) : 1
  for (let i = 0; i < 100; i += 1) {
    n += 1
    const candidate = `${stem}-${n}`.slice(0, 64)
    if (!taken.has(candidate)) return candidate
  }
  return null
}

/**
 * Judge the Name field. @param raw What the operator typed. @param opts.existing Peer ids that
 * already exist (live children AND remembered profiles): a collision is the server's 409,
 * reported here while nothing has been started yet.
 */
export function peerNameField(
  raw: string,
  opts: { existing?: readonly string[]; robotName?: string; mode?: string } = {},
): PeerNameVerdict {
  const taken = new Set((opts.existing ?? []).filter(Boolean))
  const typed = (raw ?? '').trim()

  if (typed === '') {
    const family = (opts.robotName ?? '').trim() || 'robot'
    const mode = (opts.mode ?? '').trim() || 'sim'
    return {
      value: null,
      problem: null,
      note: `unnamed: the server will call it ${family}-${mode}-{clock} — name it now if you want to recognise it later, a peer cannot be renamed while it runs`,
      suggestion: null,
    }
  }

  if (typed.length > 64) {
    return {
      value: null,
      problem: `that name is ${typed.length} characters; a peer id must be 1-64`,
      note: null,
      suggestion: sanitizePeerName(typed),
    }
  }

  if (!PEER_NAME_RE.test(typed)) {
    const clean = sanitizePeerName(typed)
    return {
      value: null,
      problem:
        'a name becomes a zenoh key segment, so only letters, digits and . _ : - are allowed' +
        (/[*/?#]/.test(typed) ? " — '*' and '/' there rewrite the fleet's key space instead of naming a peer" : ''),
      note: null,
      suggestion: clean && clean !== typed ? clean : null,
    }
  }

  if (taken.has(typed)) {
    return {
      value: null,
      problem: `${typed} already exists — the server refuses a duplicate peer id rather than shadowing the one that is running`,
      note: null,
      suggestion: freeVariant(typed, taken),
    }
  }

  return { value: typed, problem: null, note: null, suggestion: null }
}
