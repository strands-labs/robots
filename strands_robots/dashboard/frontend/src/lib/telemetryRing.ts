/** The state-topic telemetry ring, as pure functions. */
import type { Peer } from '../types'

/** The state-topic telemetry ring, as pure functions. */

export const TELEMETRY_CAP = 120 // ~12 s at the 10 Hz state topic

export interface TelemetrySample { t: number; motion: number }

export interface RingAcc {
  /** newest last, capped at TELEMETRY_CAP */
  samples: TelemetrySample[]
  /** joint values of the previous frame, for the per-frame delta */
  prev: number[]
  /** has ANY frame carried joint positions? null before the first frame */
  jointsSeen: boolean | null
  /** the state timestamp already folded in, to ignore repeated frames */
  lastT: number | undefined
}

export const emptyRing = (): RingAcc => ({ samples: [], prev: [], jointsSeen: null, lastT: undefined })

/**
 * Joint positions out of a state payload, whatever shape the robot published: a bare number, a
 * [position, velocity] pair, or an object with .position.
 */
export function jointValues(peer: Pick<Peer, 'state'>): number[] {
  const joints = peer.state?.joints
  if (!joints) return []
  return Object.values(joints).map(v => {
    if (typeof v === 'number') return v
    if (Array.isArray(v)) return v[0] ?? 0
    return (v as { position?: number }).position ?? 0
  })
}

/** Mean absolute change per joint between two frames. */
export function motionBetween(prev: number[], values: number[]): number {
  // A frame whose joint COUNT changed cannot be differenced against the previous one; report
  // no motion rather than a spike manufactured from a reshaped vector.
  if (prev.length !== values.length || values.length === 0) return 0
  let motion = 0
  for (let i = 0; i < values.length; i++) motion += Math.abs(values[i] - prev[i])
  return motion / values.length
}

/** Fold one state frame into the ring. */
export function advance(acc: RingAcc, peer: Pick<Peer, 'state'>, nowS: number): RingAcc {
  const stateT = peer.state?.t
  if (stateT === undefined || stateT === acc.lastT) return acc
  const values = jointValues(peer)
  return {
    samples: [...acc.samples, { t: nowS, motion: motionBetween(acc.prev, values) }].slice(-TELEMETRY_CAP),
    prev: values,
    // Once joints have been seen they stay seen: an arm that drops a frame has not stopped being
    // an arm, and `jointsSeen: false` is authoritative enough downstream to suppress the whole
    // motion sentence.
    jointsSeen: (acc.jointsSeen ?? false) || values.length > 0,
    lastT: stateT,
  }
}

/** A gap in ARRIVALS longer than this starts a new episode. */
export const TELEMETRY_GAP_S = 5

/** The trailing run of samples with no dead gap in it — the only ones that describe NOW. */
export function recentRun(samples: TelemetrySample[], maxGapS = TELEMETRY_GAP_S): TelemetrySample[] {
  for (let i = samples.length - 1; i > 0; i--) {
    if (samples[i].t - samples[i - 1].t > maxGapS) return samples.slice(i)
  }
  return samples
}

export interface TelemetryView {
  samples: TelemetrySample[]
  hz: number
  /** joints changed recently; null until enough samples to judge */
  moving: boolean | null
  /** the peer publishes joint positions at all; null before any state sample */
  jointsSeen: boolean | null
  /** seconds since the newest state sample; null before the first one */
  stateAgeS: number | null
}

/** Derive the card's view of the ring. NO JOINTS MEANS NO OPINION. */
export function summarize(acc: RingAcc, nowS: number): TelemetryView {
  const { jointsSeen } = acc
  const samples = recentRun(acc.samples)
  const newest = acc.samples[acc.samples.length - 1]
  if (samples.length < 2) {
    return { samples, hz: 0, moving: null, stateAgeS: newest ? nowS - newest.t : null, jointsSeen }
  }
  const span = samples[samples.length - 1].t - samples[0].t
  const hz = span > 0 ? (samples.length - 1) / span : 0
  const peak = Math.max(...samples.map(s => s.motion), 1e-6)
  const moving = jointsSeen && samples.length >= 10
    ? samples.slice(-10).some(s => s.motion > peak * 0.05)
    : null
  return { samples, hz, moving, stateAgeS: nowS - samples[samples.length - 1].t, jointsSeen }
}
