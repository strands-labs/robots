/**
 * Mirror a REAL arm into its sim twin, 1:1 — the real arm as teleop SOURCE
 * (hand-moved), the twin's robot as FOLLOWER, at the mesh's stream rate.
 *
 * This module answers ONE question for the real arm's own screen: can this
 * arm be mirrored right now, and if not, what stands in the way. The teleop
 * rail itself already accepts a sim follower (the /teleop/receive route says
 * so in its own docstring); what was missing is the affordance on the REAL
 * arm's card — the operator had to know to open the twin's CHILD peer and
 * pick a leader there.
 */
import { isChildOf, type HostInput } from './armHosts'

export interface MirrorInput extends HostInput {
  /** presence.robot_type — 'sim' means pixels, anything else is treated as metal */
  robot_type?: string | null
  role?: string | null
  role_volts?: number | null
  role_source?: string | null
}

export interface MirrorPlan {
  /** the twin's robot peer that would follow — null when the mirror cannot start */
  follower: string | null
  /** hard preconditions: while any of these stand, the start button is not offered */
  blockers: string[]
  /** true but not disqualifying — what the operator should know before starting */
  notes: string[]
}

const isSim = (p: MirrorInput | undefined): boolean => p?.robot_type === 'sim'
const jointCount = (p: MirrorInput | undefined): number =>
  typeof p?.joints === 'number' && p.joints > 0 ? p.joints : 0

/**
 * The twin process for a real arm is named `<id>-twin`; the robot that can
 * actually follow is published UNDER it as `<id>-twin__<robot>` and is the
 * peer carrying joints (measured on this fleet: the twin process reports 0
 * joints and no cameras — the articulated state lives on the child).
 */
export function twinFollowerOf(peerId: string, peers: MirrorInput[] | null | undefined): {
  process: MirrorInput | null
  arm: MirrorInput | null
} {
  const list = (peers ?? []).filter(p => p && p.peer_id)
  const twinId = `${peerId}-twin`
  const process = list.find(p => p.peer_id === twinId) ?? null
  const arm = list.find(p => isChildOf(p.peer_id, twinId) && jointCount(p) > 0) ?? null
  return { process, arm }
}

/** Can `peerId` (a REAL arm) be mirrored into its sim twin right now? */
export function mirrorPlan(peerId: string, peers: MirrorInput[] | null | undefined): MirrorPlan {
  const list = (peers ?? []).filter(p => p && p.peer_id)
  const subject = list.find(p => p.peer_id === peerId)
  const blockers: string[] = []
  const notes: string[] = []

  if (!subject) {
    return { follower: null, blockers: [`${peerId} is not on the mesh`], notes }
  }
  if (isSim(subject)) {
    // Nothing to hand-move: the mirror is a rail FROM metal TO pixels.
    return { follower: null, blockers: ['this peer is already a simulation — the mirror follows a REAL arm'], notes }
  }
  if (!jointCount(subject)) {
    blockers.push('this arm reports no joints, so it has no position to publish (its log — devices › logs — says why)')
  }

  const { process, arm } = twinFollowerOf(peerId, list)
  if (!process) {
    blockers.push('no sim twin on the mesh — spawn it first (the twin button on this card), then mirror')
  } else if (!arm) {
    blockers.push(`${process.peer_id} is up but no robot under it reports joints yet — it may still be loading; ask again in a few seconds`)
  }

  // Hand-movability is a property of the ARM's wiring, not of this rail: a
  // leader-wired arm is free to move by hand; a follower under torque will
  // resist. Relaxing torque is a MOTION-ADJACENT change the dashboard will
  // not perform — the operator does it at the arm, deliberately.
  if (subject.role === 'leader' && subject.role_source === 'measured') {
    notes.push(`measured as a leader (${subject.role_volts ?? '?'}V) — hand-movable by design`)
  } else if (subject.role === 'follower' && subject.role_source === 'measured') {
    notes.push(
      `measured as a FOLLOWER (${subject.role_volts ?? '?'}V) — its motors may hold torque, so it will resist being hand-moved; ` +
      'relaxing torque is a physical change made at the arm, not from this screen'
    )
  } else {
    notes.push('role not measured — if the arm resists being hand-moved, its motors hold torque; relaxing it is done at the arm, not from this screen')
  }

  const fj = jointCount(arm ?? undefined)
  const lj = jointCount(subject)
  if (arm && fj && lj && fj !== lj) {
    notes.push(`this arm reports ${lj} joints and ${arm.peer_id} reports ${fj} — only the names they share can be mirrored`)
  }

  return { follower: blockers.length ? null : arm?.peer_id ?? null, blockers, notes }
}

/** One line for the screen. */
export function mirrorSentence(plan: MirrorPlan): string {
  if (plan.blockers.length) return `cannot mirror: ${plan.blockers.join('; ')}`
  return `hand-move this arm and ${plan.follower} copies it, joint for joint — nothing physical moves`
}
