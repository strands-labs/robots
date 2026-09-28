/** Where focus goes when an overlay opens, and whether it is given back — as pure rules. */

export const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

export interface FocusCandidate {
  /** the overlay marked this element with [data-autofocus] */
  autofocus?: boolean
  /** aria-label, then title */
  label?: string
  /** visible text */
  text?: string
}

/** A close affordance, by any of the three things a real close button actually carries. */
export function looksLikeClose(c: FocusCandidate): boolean {
  const words = `${c.label ?? ''} ${c.text ?? ''}`.trim().toLowerCase()
  if (/(^|\s)(close|dismiss)\b/.test(words)) return true
  // A bare glyph button is a close button on every sheet in this app, and it has no words at all.
  return /^[\u00d7\u2715\u2716\u274c\u2573x]$/.test((c.text ?? '').trim())
}

/** Controls that MOVE HARDWARE or destroy work. */
export function looksDangerous(c: FocusCandidate): boolean {
  const words = `${c.label ?? ''} ${c.text ?? ''}`.toLowerCase()
  return /\b(run|start|record|recording|stop|e-?stop|resume|delete|remove|despawn|deploy|train|calibrate|home|move|teleop|replay)\b/
    .test(words)
}

/** Which candidate to focus, or 'container' for the dialog element itself. */
export function focusPlan(candidates: FocusCandidate[]): number | 'container' {
  const autofocus = candidates.findIndex(c => c.autofocus)
  if (autofocus >= 0) return autofocus
  const close = candidates.findIndex(looksLikeClose)
  if (close >= 0) return close
  const safe = candidates.findIndex(c => !looksDangerous(c))
  if (safe >= 0) return safe
  return 'container'
}

/** The element to hand focus back to on close, or null for "leave it alone". */
export function rememberOpener(active: unknown, body?: unknown): { el: unknown } | null {
  if (!active || active === body) return null
  return { el: active }
}

/** Whether to pull focus back to the opener. */
export function shouldRestoreFocus(s: {
  activeInsideOverlay: boolean
  activeIsBody: boolean
  openerConnected: boolean
}): boolean {
  if (!s.openerConnected) return false
  return s.activeInsideOverlay || s.activeIsBody
}
