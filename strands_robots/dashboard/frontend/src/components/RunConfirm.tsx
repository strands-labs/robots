/** The dialog between "I typed a sentence" and "the arm moves" (JOURNEYS #3). */
import { useEffect, useRef } from 'react'
import type { RunRisk } from '../lib/runRisk'

type Props = {
  peerId: string
  risk: RunRisk
  instruction: string
  provider: string
  /** Model path / checkpoint, when the policy names one. */
  model?: string | null
  durationS?: number
  onCancel: () => void
  onConfirm: () => void
}

export default function RunConfirm({
  peerId, risk, instruction, provider, model, durationS, onCancel, onConfirm,
}: Props) {
  const go = useRef<HTMLButtonElement>(null)

  // Focus lands on the safe reading position, not on "start" — the primary
  // action should never be one stray Enter away.
  useEffect(() => {
    go.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  return (
    <div className="sheet-backdrop" onClick={onCancel}>
      <div className="sheet run-confirm" onClick={e => e.stopPropagation()}
           role="dialog" aria-modal="true" aria-label="Confirm running a policy">
        <h2>⚠️ This will move {peerId}</h2>
        <p className="rc-lede">
          A physical arm is about to start moving on its own. Keep hands, cables and
          anything breakable out of its reach before you start.
        </p>

        <dl className="rc-facts">
          <div><dt>robot</dt><dd>{peerId} <span className="hint">({risk.reason})</span></dd></div>
          <div><dt>task</dt><dd>“{instruction}”</dd></div>
          <div><dt>policy</dt><dd>{provider}</dd></div>
          {model ? <div><dt>weights</dt><dd className="rc-mono">{model}</dd></div> : null}
          <div><dt>runs for</dt><dd>{durationS ? `${durationS}s, unless you stop it sooner` : 'until you stop it'}</dd></div>
        </dl>

        <p className="hint">
          To stop it: <b>■</b> on this card, <b>STOP ALL</b> in the corner, or press <kbd>.</kbd> anywhere.
        </p>

        <div className="sheet-actions">
          <button className="btn ghost" ref={go} onClick={onCancel}>cancel</button>
          <button className="btn danger big" onClick={onConfirm}>start moving {peerId}</button>
        </div>
      </div>
    </div>
  )
}
