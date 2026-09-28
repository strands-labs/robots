import type { Peer } from '../types'
import { healthLine } from '../lib/healthLine'
import {
  SENSOR_KINDS, declaredKinds, rowsToShow, sensorVerdict, stripSummary,
  type SensorKind, type SensorReading,
} from '../lib/sensorFreshness'

/** The one-line reading for a sensor chip: pose says WHERE, odom says how fast, and so on. */
function chipValue(kind: SensorKind, r: SensorReading): string | null {
  const n = (v: unknown): number | null =>
    typeof v === 'number' && Number.isFinite(v) ? v : null
  if (kind === 'pose') {
    const x = n(r.x), y = n(r.y)
    return x !== null && y !== null ? `${x.toFixed(2)}, ${y.toFixed(2)}` : null
  }
  if (kind === 'odom') {
    const vx = n(r.vx), wz = n(r.wz)
    if (vx === null && wz === null) return null
    return `${vx !== null ? `${vx.toFixed(2)} m/s` : ''}${vx !== null && wz !== null ? ' ' : ''}${
      wz !== null ? `${wz.toFixed(2)} rad/s` : ''}`
  }
  if (kind === 'imu') {
    const g = Array.isArray(r.gyro) ? (r.gyro as unknown[]).map(n).filter(v => v !== null) : []
    return g.length > 0 ? `gyro ${Math.max(...(g as number[]).map(Math.abs)).toFixed(3)}` : null
  }
  if (kind === 'lidar') {
    const min = n(r.min_range)
    return min !== null ? `min ${min.toFixed(2)} m` : null
  }
  return null
}

/**
 * SensorLoops readings for one peer, or nothing at all.
 *
 * The SDK has published pose/health/imu/odom/lidar since the sensor loops landed and this
 * dashboard rendered none of it, so a rover or a humanoid showed a name and a camera. Every rule
 * here lives in ../lib (sensorFreshness, healthLine) where it is tested; this component only
 * places what those return.
 *
 * Both of sensorFreshness's rails are fed from here: `presence.topics` is the peer's own
 * declaration of what it publishes, and the filed payloads are what arrived. That is what lets a
 * declared-but-silent lidar read differently from a robot that has no lidar.
 *
 * A peer with nothing declared and nothing arriving renders NOTHING rather than five rows saying
 * "absent" — the same choice TelemetryStrip makes when it has too few samples. Note that every
 * peer declares `health` (mesh.core appends it unconditionally), so an arm does get one line:
 * the host stats behind it include the free space a recording needs.
 */
export default function SensorStrip({ peer, nowS }: { peer: Peer; nowS?: number }) {
  const now = nowS ?? Date.now() / 1000
  // `lidar` holds two documents; its freshness is whichever arrived most recently.
  const lidarReading: SensorReading | null = (() => {
    const s = peer.lidar?.summary, st = peer.lidar?.state
    if (!s && !st) return null
    const ts = (r?: { t?: number }) => (typeof r?.t === 'number' ? r.t : -Infinity)
    return (ts(s) >= ts(st) ? s : st) as SensorReading
  })()

  const readings: Partial<Record<SensorKind, SensorReading | null>> = {
    health: (peer.health ?? null) as SensorReading | null,
    pose: (peer.pose ?? null) as SensorReading | null,
    odom: (peer.odom ?? null) as SensorReading | null,
    imu: (peer.imu ?? null) as SensorReading | null,
    lidar: lidarReading,
  }

  const topics = peer.presence?.topics
  const declared = new Set(declaredKinds(topics))
  const verdicts = SENSOR_KINDS.map(k => sensorVerdict(k, readings[k], now, declared.has(k)))
  const summary = stripSummary(verdicts)
  // Nothing declared and nothing arriving: say nothing. This is the honesty rule.
  if (summary === null) return null

  const shown = rowsToShow(topics, readings)
  const health = healthLine(readings.health)

  return (
    <div className="sensorstrip">
      {/* Health first: "is anything wrong with this robot" is the question, so the verdict
          leads and the individual readings follow. */}
      {readings.health && (
        <p
          className={`hint${health.tone === 'attention' ? ' warn' : health.tone === 'ok' ? ' ok' : ''}`}
          role="status"
          title={health.detail ?? undefined}
        >
          {health.text}
        </p>
      )}
      <div className="telemetry">
        {shown.map(kind => {
          const v = verdicts.find(x => x.kind === kind)!
          const reading = readings[kind]
          const value = reading ? chipValue(kind, reading) : null
          // The freshness sentence is the title, so a quiet sensor says HOW quiet on hover
          // while the row stays one line.
          return (
            <span
              key={kind}
              className={`metric${v.tone === 'stale' ? ' warn' : ''}`}
              title={`${kind}: ${v.text}`}
            >
              {kind}
              {value ? ` ${value}` : ''}
              {v.tone === 'stale' && v.ageS !== null ? ` (${v.ageS.toFixed(0)}s old)` : ''}
              {/* 'waiting' is neutral: the peer said it publishes this and has not yet. */}
              {v.tone === 'waiting' ? ' \u2014' : ''}
            </span>
          )
        })}
      </div>
    </div>
  )
}
