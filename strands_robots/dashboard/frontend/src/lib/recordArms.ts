/**
 * Say BEFORE the click that a chosen arm cannot report where it is. The backend now refuses
 * this (record_joints.refusal -> 409, non-continuable: a missing camera view is a degraded
 * dataset, positions that cannot be read are an empty one).
 */

import type { Peer } from '../types'
import { jointAbsence } from './jointAbsence'

/** Older than this and "no joints" is not evidence about now (record_joints.MAX_AGE_S). */
export const MAX_AGE_S = 30

/** How many joints this peer's snapshot reports, or null when there is nothing to read. */
function jointCount(peer: Peer | null | undefined): number | null {
  const joints = peer?.state?.joints
  if (peer?.state == null) return null
  if (joints == null) return 0
  if (Array.isArray(joints)) return joints.length
  if (typeof joints === 'object') return Object.keys(joints).length
  return null  // a shape we do not understand is not evidence of absence
}

function ageS(peer: Peer | null | undefined, nowS: number): number | null {
  const seen = peer?.last_seen
  if (typeof seen !== 'number' || !Number.isFinite(seen) || seen <= 0) return null
  return Math.max(0, nowS - seen)
}

/**
 * Why this arm cannot be recorded from, or null to proceed. @param slot which side of the pair
 * it was chosen for — the consequence differs: the follower's positions are the dataset's
 * observations, the leader's are its actions.
 */
export function armJointWarning(
  peer: Peer | null | undefined,
  { slot, nowS }: { slot: 'leader' | 'follower'; nowS: number },
): string | null {
  const count = jointCount(peer)
  if (count === null || count > 0) return null
  const age = ageS(peer, nowS)
  if (age === null || age > MAX_AGE_S) return null
  const note = jointAbsence({
    state: peer?.state, presence: peer?.presence, problem: peer?.joint_problem, nowS,
  })
  const why = [note.text, note.hint].filter(Boolean).join(' — ')
  return (
    `${slot} ${peer?.peer_id ?? ''} reports no joint positions, so the episodes would carry no ` +
    `${slot === 'leader' ? 'actions' : 'observations'} to learn from` +
    (why ? `: ${why}` : '') +
    '. The recording will be refused until this arm reads.'
  )
}
