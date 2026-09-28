/** When an overlay opens, focus goes INSIDE it — and comes back to whatever opened it on close. */
import { useEffect, useRef, type RefObject } from 'react'
import { FOCUSABLE, focusPlan, rememberOpener, shouldRestoreFocus } from './dialogFocus'

/** When an overlay opens, focus goes INSIDE it — and comes back to whatever opened it on close. */
export function useDialogFocus(ref: RefObject<HTMLElement | null>, open = true): void {
  const opener = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (!open) return
    const remembered = rememberOpener(document.activeElement, document.body)
    opener.current = remembered?.el instanceof HTMLElement ? remembered.el : null
    // One frame later: the overlay's children may still be mounting on the open tick.
    const id = requestAnimationFrame(() => {
      const node = ref.current
      if (!node || node.contains(document.activeElement)) return
      const els = [...node.querySelectorAll<HTMLElement>(FOCUSABLE)]
      const plan = focusPlan(els.map(el => ({
        autofocus: el.hasAttribute('data-autofocus'),
        label: el.getAttribute('aria-label') ?? el.getAttribute('title') ?? '',
        text: el.textContent ?? '',
      })))
      if (plan === 'container') {
        // A real answer, not a failure: focusing the dialog puts a screen reader inside it and makes
        // Tab start inside it.
        if (!node.hasAttribute('tabindex')) node.setAttribute('tabindex', '-1')
        node.focus()
        return
      }
      els[plan]?.focus()
    })
    return () => {
      cancelAnimationFrame(id)
      const active = document.activeElement
      if (shouldRestoreFocus({
        activeInsideOverlay: ref.current?.contains(active) ?? false,
        activeIsBody: active === document.body,
        openerConnected: opener.current?.isConnected ?? false,
      })) opener.current?.focus()
    }
  }, [ref, open])
}
