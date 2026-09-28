/**
 * The dashboard's global keys — JOURNEYS #12 ("11 tab stops to the e-stop on the armed-run
 * screen").
 */

export type HotkeyAction = 'estop' | 'help' | 'close' | null

/** The parts of a KeyboardEvent this decision actually depends on. */
export interface KeyLike {
  key: string
  metaKey?: boolean
  ctrlKey?: boolean
  altKey?: boolean
  shiftKey?: boolean
  /** Tag name of the event target, e.g. 'INPUT' — case-insensitive. */
  targetTag?: string
  /** True when the target is contenteditable (a rich text surface). */
  editable?: boolean
  repeat?: boolean
}

const TYPING_TAGS = /^(INPUT|TEXTAREA|SELECT)$/i

export function isTyping(e: KeyLike): boolean {
  return !!e.editable || TYPING_TAGS.test(e.targetTag ?? '')
}

export function hotkeyVerdict(e: KeyLike): HotkeyAction {
  // Escape belongs to whatever is on top, everywhere, always — including while
  // typing, because "I want out of this form" is the same intent.
  if (e.key === 'Escape') return 'close'

  // THE BRAKE, reachable from inside a half-typed task sentence. Alt is left
  // out on purpose: Alt+. types characters on several keyboard layouts.
  const stopChord = (e.metaKey || e.ctrlKey) && !e.altKey && e.key === '.'
  if (stopChord) return 'estop'

  // Bare keys only outside text entry: a task sentence ends in a full stop, and
  // "?" is a character people type.
  if (isTyping(e)) return null
  if (e.altKey || e.metaKey || e.ctrlKey) return null
  if (e.key === '.') return 'estop'
  if (e.key === '?') return 'help'
  return null
}

/** What to advertise on the button (`aria-keyshortcuts` syntax). */
export const ESTOP_KEYSHORTCUTS = '. Meta+. Control+.'
