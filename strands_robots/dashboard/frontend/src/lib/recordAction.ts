/** R1 / UX_REVIEW:107 — the copy for the scariest button in the collect flow. */

export interface OpenAction {
  label: string
  /** the consequence, shown next to the button — never empty */
  hint: string
  /** extra class for the button, '' when this is the real thing */
  cls: string
  aria: string
}

const REAL_HINT =
  'Both arms leave the fleet while recording: their peers are despawned and the ' +
  'ports handed to the recorder, and the follower is energised to hold position. ' +
  'Nothing is written until you start an episode.'

const MOCK_HINT =
  'Rehearsal: this backend has no /api/record, so no arm is touched, no port is ' +
  'taken and no dataset is written. The buttons work so you can learn the flow.'

/**
 * @param mock true = in-browser rehearsal, false = real recorder, null/undefined = not known
 * yet (claim neither).
 */
export function openActionCopy(mock: boolean | null | undefined): OpenAction {
  if (mock === true) {
    return {
      label: 'open a rehearsal session',
      hint: MOCK_HINT,
      cls: 'rehearsal',
      aria: 'open a rehearsal session — no arm is touched and nothing is written',
    }
  }
  if (mock === false) {
    return {
      label: 'open the arms for recording',
      hint: REAL_HINT,
      cls: '',
      aria: 'open the arms for recording — despawns both peers and energises the follower',
    }
  }
  // Not probed yet: name the action, promise nothing about which recorder runs.
  return {
    label: 'open the arms for recording',
    hint: REAL_HINT,
    cls: '',
    aria: 'open the arms for recording — despawns both peers and energises the follower',
  }
}
