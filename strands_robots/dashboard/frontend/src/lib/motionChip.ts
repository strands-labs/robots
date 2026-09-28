/**
 * The motion chip: "is this arm moving RIGHT NOW", the question an operator asks before
 * reaching over the desk. It has THREE states, not two.
 */
export interface MotionChip {
  /** css modifier: still | moving | unknown */
  tone: 'still' | 'moving' | 'unknown'
  /** the word in the chip */
  label: string
  /** tooltip - the evidence behind the word */
  title: string
  /** screen-reader sentence: a coloured dot is not an announcement */
  aria: string
}

export function motionChip(moving: boolean | null | undefined, opts: {
  /** the peer publishes joint positions at all (null/undefined = unknown yet) */
  jointsSeen?: boolean | null
} = {}): MotionChip {
  if (moving === true) {
    return {
      tone: 'moving',
      label: 'moving',
      title: 'joints are changing (mean absolute delta over the last samples) - keep hands clear',
      aria: 'joints are moving, keep hands clear',
    }
  }
  if (moving === false) {
    return {
      tone: 'still',
      label: 'still',
      title: 'joints are not changing (mean absolute delta over the last samples)',
      aria: 'joints measured still',
    }
  }
  if (opts.jointsSeen === false) {
    return {
      tone: 'unknown',
      label: 'motion unknown',
      title: 'this robot publishes no joint positions, so movement cannot be measured here - '
        + 'treat the arm as able to move',
      aria: 'motion unknown, this robot publishes no joint positions',
    }
  }
  return {
    tone: 'unknown',
    label: 'measuring',
    title: 'not enough state samples yet to judge movement (about a second of telemetry)',
    aria: 'motion not measured yet',
  }
}
