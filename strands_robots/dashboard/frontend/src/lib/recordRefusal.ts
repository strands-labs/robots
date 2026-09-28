/** A refused recording the operator can actually answer, on the screen. */

/** The overrides `/api/record/open` accepts, and what ticking one admits to. */
export interface RecordOverride {
  /** The request field to send. Exactly as the backend spells it. */
  flag: 'ignore_dead_cameras' | 'ignore_missing_cameras' | 'ignore_camera_identity'
  /** First-person, present tense: what the operator is claiming by ticking. */
  label: string
  /** What the dataset will carry if the operator is wrong — never hidden. */
  cost: string
}

const OVERRIDES: RecordOverride[] = [
  {
    flag: 'ignore_dead_cameras',
    label: 'I know this camera is stale — record without waiting for it',
    cost: 'the episodes may carry a frozen image, or none, for that view',
  },
  {
    flag: 'ignore_missing_cameras',
    label: 'record without the camera this machine cannot see',
    cost: 'that view is simply absent from every episode',
  },
  {
    flag: 'ignore_camera_identity',
    label: 'my cameras really are at these indices — record with them as they stand',
    cost: 'if the numbering did shift, every episode records the WRONG view while looking healthy',
  },
]

/**
 * The override a refusal is offering, or null when it offers none. @param message The 409
 * detail, exactly as the server wrote it.
 */
export function overrideOffered(message: unknown): RecordOverride | null {
  if (typeof message !== 'string' || !message) return null
  const named = OVERRIDES.filter(o => message.includes(o.flag))
  return named.length === 1 ? named[0] : null
}

/** The three flags, in the order `/api/record/open` checks them: dead, then missing, then identity. */
export type RecordOverrideFlag = RecordOverride['flag']
const FLAGS: RecordOverrideFlag[] = OVERRIDES.map(o => o.flag)

/** Every admission the operator has made about THIS attempt sequence, not just the last one. */
export function nextAcknowledged(
  prev: readonly string[],
  offered: RecordOverride | null,
  acknowledged: boolean,
): RecordOverrideFlag[] {
  const kept = FLAGS.filter(f => prev.includes(f))
  if (!offered || !acknowledged || kept.includes(offered.flag)) return kept
  return FLAGS.filter(f => kept.includes(f) || f === offered.flag)
}

/** Those admissions as request fields. Nothing else can get in: the allowlist is the safety property. */
export function overrideBodyFlags(flags: readonly string[]): Record<string, true> {
  const body: Record<string, true> = {}
  for (const f of FLAGS) if (flags.includes(f)) body[f] = true
  return body
}
