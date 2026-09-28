import { useEffect, useRef } from 'react'
import { HELP_TOPICS, DOC_LINKS, REPO_DOC_PATHS } from '../lib/helpTopics'

/** The help affordance — JOURNEYS #7. */
export default function HelpSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement | null>(null)
  useEffect(() => { if (open) closeRef.current?.focus() }, [open])
  if (!open) return null

  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div
        className="sheet help-sheet"
        role="dialog"
        aria-modal="true"
        aria-label="Help"
        onClick={e => e.stopPropagation()}
      >
        <div className="help-head">
          <h2>Help</h2>
          <button ref={closeRef} className="btn ghost" onClick={onClose} aria-label="Close help">✕</button>
        </div>

        {HELP_TOPICS.map(t => (
          <section className="help-topic" key={t.title}>
            <h3>{t.title}</h3>
            {t.lines.map((line, i) => <p key={i}>{line}</p>)}
          </section>
        ))}

        <section className="help-topic">
          <h3>Deeper reading</h3>
          <ul className="help-links">
            {DOC_LINKS.map(l => (
              <li key={l.url}>
                <a href={l.url} target="_blank" rel="noreferrer noopener">{l.label} ↗</a>
                <span className="hint"> — {l.note}</span>
              </li>
            ))}
          </ul>
          {/* Paths, not links: these pages are in the repo but not on the deployed site yet, and a 404 handed to a confused operator is worse than no link at all. */}
          <p className="hint">In this repository (not published yet):</p>
          <ul className="help-paths">
            {REPO_DOC_PATHS.map(p => <li key={p}><code>{p.split(' — ')[0]}</code> — {p.split(' — ')[1]}</li>)}
          </ul>
        </section>

        <p className="hint">Press <kbd>?</kbd> for this sheet, <kbd>.</kbd> for STOP ALL, <kbd>Esc</kbd> to close.</p>
      </div>
    </div>
  )
}
