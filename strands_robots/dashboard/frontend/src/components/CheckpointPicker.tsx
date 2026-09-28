import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/endpoints'
import { emptyNote, isCurrent } from '../lib/checkpointSearch'

interface CheckpointRow {
  repo_id: string
  local: boolean
  downloads: number | null
  policy_type: string | null
  tags: string[]
}

interface HfAuth { authenticated: boolean; user: string | null; detail: string | null }

/** Type-ahead over LeRobot policy checkpoints for `pretrained_name_or_path`. */
export default function CheckpointPicker({ value, onPick, disabled }: {
  value: string
  onPick: (repoId: string, policyType: string | null) => void
  disabled?: boolean
}) {
  const [query, setQuery] = useState(value)
  const [rows, setRows] = useState<CheckpointRow[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [hubProblem, setHubProblem] = useState<string | null>(null)
  const [hfAuth, setHfAuth] = useState<HfAuth | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const debounce = useRef<ReturnType<typeof setTimeout>>()
  // The debounce cancels a pending TIMER, not an in-flight fetch: without a sequence, a slow
  // search for "act" can resolve after a fast one for "smolvla" and paint act's rows under the
  // newer query.
  const seq = useRef(0)
  // Which query the rows on screen belong to, so the empty note cannot describe
  // a different search than the one that produced it.
  const [shownQuery, setShownQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => { setQuery(value) }, [value])

  // close on outside click
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const searchNow = (q: string) => {
    clearTimeout(debounce.current)
    const mine = ++seq.current
    debounce.current = setTimeout(async () => {
      setLoading(true)
      try {
        const j = await api(`/api/checkpoints/search?q=${encodeURIComponent(q)}&limit=12`)
        if (!isCurrent(mine, seq.current)) return
        setRows(j.results ?? [])
        setHubProblem(j.hub_problem ?? null)
        setHfAuth(j.hf_auth ?? null)
        setFailed(null)
        setShownQuery(q)
        setOpen(true)
      } catch (e) {
        // the search endpoint itself failed (auth, network) - name it instead
        // of rendering the same silence as 'no matches'
        if (!isCurrent(mine, seq.current)) return
        setRows([])
        setFailed((e as any)?.message ?? String(e))
        setShownQuery(q)
        setOpen(true)
      } finally {
        // A superseded request must not switch the spinner off under a newer one.
        if (isCurrent(mine, seq.current)) setLoading(false)
      }
    }, 300)
  }

  const fmt = (n: number | null) =>
    n == null ? '' : n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)

  return (
    <div className="ckpt" ref={rootRef}>
      <input
        placeholder="search checkpoints… (e.g. smolvla, act so101)" aria-label="search checkpoints"
        value={query}
        onChange={e => { setQuery(e.target.value); onPick(e.target.value, null); searchNow(e.target.value) }}
        onFocus={() => { if (rows.length) setOpen(true); else searchNow(query) }}
        disabled={disabled}
      />
      {loading && <span className="ckpt-spin">…</span>}
      {open && (
        <div className="ckpt-menu">
          {failed && <div className="ckpt-note bad">✗ search failed: {failed}</div>}
          {/* When there are no rows the empty note carries this reason itself — two lines saying "the Hub is down" is one line the eye skips. */}
          {!failed && hubProblem && rows.length > 0 && <div className="ckpt-note warn">⚠ {hubProblem}</div>}
          {!failed && hfAuth && (
            <div className={`ckpt-note ${hfAuth.authenticated ? 'ok' : ''}`}>
              {hfAuth.authenticated
                ? `HF: signed in as ${hfAuth.user} — private + gated repos reachable`
                : `HF: anonymous — ${hfAuth.detail ?? 'public repos only'}`}
            </div>
          )}
          {rows.length === 0 && !failed && (
            // Scoped to what was actually consulted: with the Hub down, only the
            // local cache answered, and "no checkpoints match" would be a claim
            // about a catalogue nobody asked.
            <div className={hubProblem ? 'ckpt-note warn' : 'ckpt-note'}>
              {emptyNote({ query: shownQuery, hubProblem })}
            </div>
          )}
          {rows.map(r => (
            <button
              key={r.repo_id}
              className="ckpt-row"
              onMouseDown={e => e.preventDefault()}
              onClick={() => { onPick(r.repo_id, r.policy_type); setQuery(r.repo_id); setOpen(false) }}
            >
              <span className="ckpt-id">{r.repo_id}</span>
              <span className="ckpt-meta">
                {r.local && <b className="ckpt-local">local</b>}
                {r.policy_type && <em>{r.policy_type}</em>}
                {r.downloads != null && <span>↓{fmt(r.downloads)}</span>}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
