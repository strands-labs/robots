/** The state-topic telemetry ring, as a hook. */
import { useEffect, useRef, useState } from 'react'
import type { Peer } from '../types'
import { advance, emptyRing, summarize, TELEMETRY_CAP } from './telemetryRing'
import type { RingAcc, TelemetrySample, TelemetryView } from './telemetryRing'

export type { TelemetrySample } from './telemetryRing'
export type Telemetry = TelemetryView
export { TELEMETRY_CAP }

/** The state-topic telemetry ring, as a hook. */
export function useTelemetry(peer: Peer): Telemetry {
  const acc = useRef<RingAcc>(emptyRing())
  const [, tick] = useState(0)

  const stateT = peer.state?.t

  useEffect(() => {
    const next = advance(acc.current, peer, Date.now() / 1000)
    // advance() returns the SAME object for a frame that carries nothing new, so this
    // cannot loop on a repeated timestamp.
    if (next === acc.current) return
    acc.current = next
    tick(n => n + 1)
  }, [stateT]) // eslint-disable-line react-hooks/exhaustive-deps

  return summarize(acc.current, Date.now() / 1000)
}
