import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/endpoints'
import {
  parseCalibrationDetail, parseCalibrationList,
  type CalibrationDetail, type CalibrationEntry,
} from '../lib/calibration'

type ServerMotor = {
  name: string
  id?: number | null
  drive_mode?: number | null
  homing_offset?: number | null
  range_min?: number | null
  range_max?: number | null
}

type TextReply = {
  status?: string
  text?: string
  path?: string
  modified?: string
  motors?: ServerMotor[]
}

type Selected = {
  entry: CalibrationEntry
  loading: boolean
  detail?: CalibrationDetail
  /** the server's own words when the detail could not be shown */
  problem?: string
}

/** A number for display, or undefined so the row renders a dash rather than "null". */
const show = (v: number | null | undefined): string | undefined =>
  v === null || v === undefined ? undefined : String(v)

const label = (e: CalibrationEntry) => `${e.deviceType}/${e.model}/${e.id}`

/** The calibration files on this machine, read-only. */
export default function CalibrationSection() {
  const [entries, setEntries] = useState<CalibrationEntry[] | null>(null)
  const [location, setLocation] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sel, setSel] = useState<Selected | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await api<TextReply>('/api/calibration')
      const parsed = parseCalibrationList(r.text ?? '')
      // A tool-level failure still answers 200 with status:'error'; treating it
      // as an empty list would report "no calibrations" for what is really a
      // broken read.
      if (r.status && r.status !== 'success' && parsed.entries.length === 0) {
        setError(r.text || 'calibration list failed')
        setEntries([])
        return
      }
      setEntries(parsed.entries)
      setLocation(parsed.location ?? null)
      setError(null)
    } catch (e: any) {
      setError(e?.message ?? String(e))
      setEntries([])
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const select = async (entry: CalibrationEntry) => {
    if (sel && label(sel.entry) === label(entry)) { setSel(null); return }
    setSel({ entry, loading: true })
    // Both parameters are required to identify a calibration: a name alone is ambiguous
    // (leader_arm exists under three models here) and the endpoint answers 409 with the candidates
    // rather than guessing.
    const path = `/api/calibration/${encodeURIComponent(entry.id)}`
      + `?device_type=${encodeURIComponent(entry.deviceType)}`
      + `&device_model=${encodeURIComponent(entry.model)}`
    try {
      const r = await api<TextReply>(path)
      // The server sends the motors as data now. Markdown parsing stays as the
      // fallback for an older backend, but reading prose for numbers that exist
      // in a field is a bridge, not a plan.
      const detail = r.motors?.length
        ? { title: r.text?.split('\n')[0], path: r.path, modified: r.modified,
            motors: r.motors.map(m => ({
              name: m.name,
              id: show(m.id),
              driveMode: show(m.drive_mode),
              homingOffset: show(m.homing_offset),
              rangeMin: show(m.range_min),
              rangeMax: show(m.range_max),
            })) }
        : parseCalibrationDetail(r.text ?? '')
      if (r.status === 'success' && detail.motors.length > 0) {
        setSel({ entry, loading: false, detail })
      } else {
        setSel({ entry, loading: false, detail, problem: r.text || 'no detail returned' })
      }
    } catch (e: any) {
      setSel({ entry, loading: false, problem: e?.message ?? String(e) })
    }
  }

  const groups: { key: string; deviceType: string; model: string; rows: CalibrationEntry[] }[] = []
  for (const e of entries ?? []) {
    const key = `${e.deviceType}/${e.model}`
    const last = groups[groups.length - 1]
    if (last && last.key === key) last.rows.push(e)
    else groups.push({ key, deviceType: e.deviceType, model: e.model, rows: [e] })
  }

  return (
    <section>
      <h3>
        Calibration {entries ? <em>{entries.length}</em> : null}
        <button className="btn ghost" onClick={() => void load()}>reload</button>
      </h3>

      {error && <div className="result bad">⚠ {error}</div>}

      {entries === null && !error && <p className="hint">reading calibration files…</p>}

      {entries !== null && entries.length === 0 && !error && (
        <p className="hint">
          No calibration files on this machine{location ? <> under <code>{location}</code></> : null}.
          An arm with no calibration reports raw servo counts, so its joint limits will be wrong —
          run <code>lerobot-calibrate</code> in a terminal to create one.
        </p>
      )}

      {groups.map(g => (
        <div key={g.key}>
          <h4 className="calibgroup">
            <span className="mono">{g.model}</span>
            <span className="meta">{g.deviceType}</span>
          </h4>
          <ul className="devlist">
            {g.rows.map(e => {
              const open = !!sel && label(sel.entry) === label(e)
              return (
                <li key={label(e)} className={e.unreadable || e.problem ? 'dead' : ''}>
                  <b>{e.problem ? '⚠ ' : ''}{e.id}</b>
                  <span className="meta">
                    {e.problem
                      ? e.problem
                      : e.unreadable
                      ? 'file unreadable'
                      : [
                          e.motors !== undefined ? `${e.motors} motors` : null,
                          e.modified ?? null,
                          e.sizeKb !== undefined ? `${e.sizeKb.toFixed(1)}KB` : null,
                        ].filter(Boolean).join(' · ')}
                  </span>
                  <span className="devactions">
                    <button className="btn ghost" onClick={() => void select(e)}>
                      {open ? 'hide' : 'view'}
                    </button>
                  </span>

                  {open && sel && (
                    <div className="calibdetail">
                      {sel.loading && <p className="hint">loading {label(e)}…</p>}

                      {sel.detail && sel.detail.motors.length > 0 && (
                        <>
                          <table className="jointtable">
                            <thead>
                              <tr>
                                <th>motor</th><th>id</th><th>drive</th>
                                <th>homing offset</th><th>range</th>
                              </tr>
                            </thead>
                            <tbody>
                              {sel.detail.motors.map(m => (
                                <tr key={m.name}>
                                  <td>{m.name}</td>
                                  <td className="mono">{m.id ?? '—'}</td>
                                  <td className="mono">{m.driveMode ?? '—'}</td>
                                  <td className="mono">{m.homingOffset ?? '—'}</td>
                                  <td className="mono">
                                    {m.rangeMin !== undefined
                                      ? `${m.rangeMin}${m.rangeMax !== undefined ? ` … ${m.rangeMax}` : ''}`
                                      : '—'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {sel.detail.path && (
                            <p className="hint mono small">{sel.detail.path}</p>
                          )}
                        </>
                      )}

                      {!sel.loading && sel.problem && (
                        <div className="result bad">
                          <span>⚠ the backend could not return per-joint values</span>
                          <details>
                            <summary>details</summary>
                            <pre>{sel.problem}</pre>
                          </details>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      ))}

      {entries !== null && entries.length > 0 && (
        <p className="hint">
          Read-only. Values come from <code>lerobot-calibrate</code>, which moves the arm through
          its range, so it is a terminal job and not a dashboard button — this view only shows what
          it wrote{location ? <> under <code>{location}</code></> : null}. The <b>id</b> above is
          what the spawn form's <i>Calibration id</i> expects.
        </p>
      )}
    </section>
  )
}
