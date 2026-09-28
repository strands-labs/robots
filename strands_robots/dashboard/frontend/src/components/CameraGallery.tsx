import { useEffect, useRef, useState } from 'react'
import { cameraGridEmptyLine } from '../lib/boardList'
import { apiBlob } from '../lib/endpoints'

export interface CameraInfo {
  index: number
  width?: number
  height?: number
  fps?: number | null
  claimed_by?: string
  /** ready | in_use | assigned | blocked | unreadable | vanished | absent | unknown (server-side). */
  state?: string
  available?: boolean
  reason?: string
  remedy?: string
  name_hint?: string
  name_is_guess?: boolean
  geometry_from?: string
}

export interface CameraName { listing_index: number; name: string }

export interface CameraProblem { kind: string; message: string; remedy?: string; indices?: number[] }

/** One word per state, in the operator's vocabulary rather than the probe's. */
const STATE_LABEL: Record<string, string> = {
  ready: 'ready',
  in_use: 'in use',
  assigned: 'assigned, no frames',
  blocked: 'blocked by macOS',
  unreadable: 'not responding',
  absent: 'nothing here',
  /** Not 'nothing here': a camera WAS here. */
  vanished: 'gone since we saw it',
  unknown: 'not probed',
}

/** The devices screen's answer to "which camera is index N?". */
export default function CameraGallery(
  { cameras, names, problem, scanned = true, error = null }:
    { cameras: CameraInfo[]; names: CameraName[]; problem?: CameraProblem | null
      /** Whether /api/devices has ANSWERED. */
      scanned?: boolean; error?: string | null },
) {
  const [previews, setPreviews] = useState<Record<number, string>>({})
  const [errors, setErrors] = useState<Record<number, string>>({})
  const [loading, setLoading] = useState<Record<number, boolean>>({})
  const urlsRef = useRef<Record<number, string>>({})

  // Object URLs leak real memory until revoked; one place owns their lifetime.
  useEffect(() => () => {
    for (const url of Object.values(urlsRef.current)) URL.revokeObjectURL(url)
  }, [])

  const snap = async (index: number) => {
    setLoading(l => ({ ...l, [index]: true }))
    setErrors(({ [index]: _gone, ...rest }) => rest)
    try {
      const url = await apiBlob(`/api/devices/camera/${index}/preview`)
      if (urlsRef.current[index]) URL.revokeObjectURL(urlsRef.current[index])
      urlsRef.current[index] = url
      setPreviews(p => ({ ...p, [index]: url }))
    } catch (e: any) {
      setErrors(er => ({ ...er, [index]: e?.message ?? String(e) }))
    } finally {
      setLoading(l => ({ ...l, [index]: false }))
    }
  }

  return (
    <div>
      {/* Said once, loudly: when every camera fails for the same systemic reason, per-card reasons are correct and universally missed. */}
      {problem && (
        <div className="result bad camproblem">
          <b>⚠ {problem.message}</b>
          {problem.remedy ? <div>{problem.remedy}</div> : null}
        </div>
      )}
      <div className="camgrid">
        {cameras.length === 0 && (() => {
          const line = cameraGridEmptyLine({ scanned, error })
          return <p className="hint" role="status">{line.message}</p>
        })()}
        {cameras.map(c => (
          <div key={c.index} className={`camcard cam-${c.state ?? 'ready'}${c.claimed_by ? ' claimed' : ''}`}>
            <div className="camcard-head">
              <b>index {c.index}</b>
              {c.state && c.state !== 'ready' && (
                <span className={`campill campill-${c.state}`}>{STATE_LABEL[c.state] ?? c.state}</span>
              )}
              <span className="meta">
                {c.width ? `${c.width}×${c.height}` : ''}
                {c.fps ? ` @ ${c.fps}fps` : ''}
                {/* Never let remembered numbers pass as a fresh measurement. */}
                {c.geometry_from === 'remembered' && c.width ? ' (last seen)' : ''}
              </span>
            </div>
            {c.name_hint && (
              <div className="camname-hint" title="from the OS listing — the order is not OpenCV's, so treat it as a hint">
                probably <b>{c.name_hint}</b>{c.name_is_guess ? ' · snap a preview to be sure' : ''}
              </div>
            )}
            {c.claimed_by && c.state !== 'assigned' ? (
              <div className="camcard-body claimed-note">
                streaming for <b>{c.claimed_by}</b> — watch it on that robot's card
                {c.remedy ? <div className="hint">{c.remedy}</div> : null}
              </div>
            ) : c.state && c.state !== 'ready' ? (
              <div className="camcard-body cam-why">
                <div>{c.reason}</div>
                {c.remedy ? <div className="hint">→ {c.remedy}</div> : null}
                <button className="btn ghost campreview-btn" disabled={!!loading[c.index]}
                        onClick={() => void snap(c.index)}
                        title="opens the camera once, right now — the server refuses only if a robot is really streaming it">
                  {loading[c.index]
                    ? 'trying…'
                    : c.state === 'assigned'
                      ? '📷 identify it anyway'
                      /**
                       * For a vanished index the probe answers the question the remedy raises: macOS renumbers on
                       * removal, so what matters is not "is something there" but WHICH camera is there now.
                       */
                      : c.state === 'vanished' ? '📷 see which camera is here now' : 'try anyway'}
                </button>
                {previews[c.index] && (
                  <img src={previews[c.index]} alt={`camera index ${c.index} snapshot`}
                       onClick={() => void snap(c.index)} title="click to re-snap" />
                )}
                {errors[c.index] && <div className="camerr">⚠ {errors[c.index]}</div>}
              </div>
            ) : (
              <div className="camcard-body">
                {previews[c.index]
                  ? <img src={previews[c.index]} alt={`camera index ${c.index} snapshot`}
                         onClick={() => void snap(c.index)} title="click to re-snap" />
                  : (
                    <button className="btn ghost campreview-btn" disabled={!!loading[c.index]}
                            onClick={() => void snap(c.index)}>
                      {loading[c.index] ? 'snapping…' : '📷 snap a preview'}
                    </button>
                  )}
                {errors[c.index] && <div className="camerr">⚠ {errors[c.index]}</div>}
              </div>
            )}
          </div>
        ))}
      </div>
      {names.length > 0 && (
        <>
          <div className="camnames">
            {names.map(n => (
              <span key={n.listing_index} className="chip" title="position in the OS device listing">
                {n.name}
              </span>
            ))}
          </div>
          <p className="hint">
            Attached cameras by name, in OS listing order — which is <em>not</em> OpenCV index
            order. The snapshot is the identity: if you're unsure which index is which, look.
          </p>
        </>
      )}
    </div>
  )
}
