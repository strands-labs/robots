/**
 * View rules for the calibration wizard — what one backend status (GET
 * /api/calibration/run/{sid}) should put on the screen. Pure, so every screen the
 * operator can reach is testable without an arm: the component just renders this.
 *
 * The flow these rules mirror (lerobot's own, driven by the backend pty session):
 * reuse? -> middle -> recording -> saved | failed. Two facts the copy must never
 * soften: torque is OFF from the start (the arm goes limp — someone should hold it),
 * and a joint skipped during recording keeps a one-point range that lerobot refuses.
 */

export interface WizardMotor {
  name: string
  min: number
  pos: number
  max: number
}

export interface WizardStatus {
  id: string
  step: 'starting' | 'reuse' | 'middle' | 'recording' | 'saved' | 'failed'
  alive: boolean
  prompt?: string
  motors?: WizardMotor[]
  path?: string
  reason?: string
  returncode?: number | null
  tail?: string[]
}

export interface WizardButton {
  /** what to POST to …/key — or the pseudo-keys 'cancel' | 'close' */
  key: 'enter' | 'c' | 'cancel' | 'close'
  label: string
  /** the visually primary action of this screen */
  primary?: boolean
  /** true for the action that abandons the run */
  danger?: boolean
}

export interface WizardView {
  title: string
  body: string
  buttons: WizardButton[]
  /** live min/pos/max rows, recording step only */
  motors: WizardMotor[] | null
  /** which joints have NOT moved yet (min === max) — the refusal lerobot would throw, pre-empted */
  unmoved: string[]
  tone: 'info' | 'ok' | 'bad'
  /** the run is over (saved or failed) — polling can stop */
  finished: boolean
  /** raw output worth offering behind a <details> (failed only) */
  detail: string | null
}

/** wrist_roll is calibrated as a full turn by lerobot itself — a still wrist_roll is fine. */
const FULL_TURN = 'wrist_roll'

export function wizardView(s: WizardStatus): WizardView {
  const cancel: WizardButton = { key: 'cancel', label: 'cancel — nothing is saved', danger: true }

  switch (s.step) {
    case 'starting':
      return {
        title: 'starting…',
        body: 'opening the arm — torque switches OFF as calibration begins, so the arm will go limp. Keep a hand near it.',
        buttons: [cancel],
        motors: null, unmoved: [], tone: 'info', finished: false, detail: null,
      }
    case 'reuse':
      return {
        title: 'a calibration for this id already exists',
        body: 'keep the file that is already on disk, or redo the measurement from scratch. Keeping it writes the existing values to the motors and ends the wizard.',
        buttons: [
          { key: 'c', label: 'recalibrate from scratch', primary: true },
          { key: 'enter', label: 'keep the existing file' },
          cancel,
        ],
        motors: null, unmoved: [], tone: 'info', finished: false, detail: null,
      }
    case 'middle':
      return {
        title: 'hold the arm at the middle of its range',
        body: 'torque is off — the arm is limp and nothing here will move it. With your hand, put every joint near the MIDDLE of its travel (upright, elbow half bent, gripper half open), hold it there, and continue.',
        buttons: [{ key: 'enter', label: "it's at the middle — continue", primary: true }, cancel],
        motors: null, unmoved: [], tone: 'info', finished: false, detail: null,
      }
    case 'recording': {
      const motors = s.motors ?? []
      const unmoved = motors.filter(m => m.name !== FULL_TURN && m.min === m.max).map(m => m.name)
      return {
        title: 'recording — move every joint through its FULL range',
        body:
          'move each joint by hand to both of its limits, one at a time (wrist_roll is a full turn — lerobot handles it). ' +
          'The table is live: a row whose min equals its max has not moved yet, and lerobot refuses to save a joint it never saw move.',
        buttons: [{ key: 'enter', label: 'every joint has been to both limits — stop & save', primary: true }, cancel],
        motors, unmoved, tone: 'info', finished: false, detail: null,
      }
    }
    case 'saved':
      return {
        title: 'calibration saved ✓',
        body: s.path
          ? `written to ${s.path} — this is the file the arm loads at spawn, under the id you calibrated.`
          : 'written — the arm loads it at spawn under the id you calibrated.',
        buttons: [{ key: 'close', label: 'done', primary: true }],
        motors: null, unmoved: [], tone: 'ok', finished: true, detail: null,
      }
    case 'failed':
      return {
        title: 'calibration did not finish',
        body: s.reason || `the run exited (${s.returncode ?? 'unknown'}) before saving`,
        buttons: [{ key: 'close', label: 'close', primary: true }],
        motors: null, unmoved: [], tone: 'bad', finished: true,
        detail: s.tail && s.tail.length ? s.tail.join('\n') : null,
      }
  }
}

/** The confirm sheet shown BEFORE anything starts — it names the port, the id, and the
 * one physical consequence. Pure so the sentence is pinned by a test. */
export function confirmSheet(args: { port: string; deviceId: string; model: string }): {
  title: string
  body: string
} {
  return {
    title: `calibrate ${args.deviceId}`,
    body:
      `This runs lerobot-calibrate on ${args.port} (${args.model}, saved under the id ` +
      `"${args.deviceId}"). The moment it starts, torque switches OFF and the arm goes LIMP — ` +
      'hold it or let it rest safely. Nothing here commands motion: your hand does all the moving.',
  }
}
