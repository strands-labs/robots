/**
 * Whether to take or drop the screen wake lock — as a pure decision. The lock exists so a
 * phone propped next to the arms does not sleep while a robot is moving: a sleeping screen
 * drops the camera sockets and the operator loses sight of a moving arm.
 */

export type WakeAction = 'request' | 'release' | 'none'

export interface WakeState {
  /** what the app last asked for: is any robot running */
  want: boolean
  /** do we currently hold a lock */
  held: boolean
  /** document.visibilityState === 'visible' */
  visible: boolean
  /** navigator.wakeLock exists (absent on iOS Safari before 16.4, and on Firefox) */
  supported: boolean
}

export function wakeLockAction(s: WakeState): WakeAction {
  // No API to call. The caller shows this to the operator instead of pretending the screen is held.
  if (!s.supported) return 'none'
  if (s.want) {
    if (s.held) return 'none'
    // A request while hidden is REFUSED by the browser (it throws NotAllowedError).
    return s.visible ? 'request' : 'none'
  }
  // Releasing does not need visibility, and a lock left held after the task ends drains the battery
  // of the phone the operator is watching the next task on.
  return s.held ? 'release' : 'none'
}

/**
 * What to tell the operator about the screen. `held` is the truth from the API, not our
 * intent, so "sleep is prevented" is never claimed on a platform that refused.
 */
export function wakeLockNote(s: WakeState): string | null {
  if (!s.want) return null
  if (!s.supported) return 'this browser cannot keep the screen awake — sleep may drop the camera view'
  return s.held ? null : 'screen sleep is not being prevented yet'
}
