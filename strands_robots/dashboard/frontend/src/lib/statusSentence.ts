/**
 * The 5-second answer: one plain-language sentence per robot card that says what the arm is
 * DOING and whether it is safe to approach.
 */

export type Severity = 'ok' | 'active' | 'warn' | 'danger'

export interface StatusFacts {
  /** peer marked stale by the server (no heartbeat ~15s) */
  stale: boolean
  /** seconds since last heartbeat, when known */
  lastSeenAgoS: number | null
  /** presence.connected - hardware attached (null = not reported) */
  hwConnected: boolean | null
  /** task.status from the state topic, e.g. 'idle' | 'running' */
  taskStatus: string | null
  /** task.instruction when one is set */
  instruction: string | null
  /** seconds the current task has been going, when known */
  taskDurationS: number | null
  moving: boolean | null
  /** Whether the peer publishes joint positions AT ALL (null = unknown yet). */
  jointsSeen?: boolean | null
  hostsChildren?: string[] | null
  /** seconds since the last state-topic sample (null = no samples yet) */
  stateAgeS: number | null
  lockout?: string | null
}

export interface StatusLine {
  severity: Severity
  /** the sentence itself */
  text: string
  /** short word for the leading glyph/chip */
  word: string
}

const quote = (s: string, max = 44) =>
  `\u201C${s.length > max ? s.slice(0, max - 1) + '\u2026' : s}\u201D`

export function statusSentence(f: StatusFacts): StatusLine {
  // Dead peer first: nothing else in the snapshot can be trusted.
  if (f.stale) {
    const ago = f.lastSeenAgoS != null ? ` for ${Math.round(f.lastSeenAgoS)}s` : ''
    return {
      severity: 'danger',
      word: 'offline',
      text: `no heartbeat${ago} — state unknown, treat the arm as unpredictable`,
    }
  }

  // Peer alive but its state stream froze: the joints below are a photograph.
  if (f.stateAgeS != null && f.stateAgeS > 5) {
    return {
      severity: 'warn',
      word: 'frozen',
      text: `peer is alive but its state stream stopped ${Math.round(f.stateAgeS)}s ago — joints shown are stale`,
    }
  }

  if (f.hwConnected === false) {
    return {
      severity: 'warn',
      word: 'no hw',
      text: 'hardware not connected — the arm is unplugged or unpowered, nothing can move',
    }
  }

  if ((f.lockout ?? '').trim().toLowerCase() === 'locked') {
    if (f.moving === true) {
      // A lockout means commands are refused. Joints moving anyway means the lockout is not holding,
      // or something outside the mesh is driving the arm. That is the worst state on this card.
      return {
        severity: 'danger',
        word: 'locked?!',
        text: 'an e-stop lockout is in place but the joints are MOVING — the lockout is not holding, or '
          + 'something outside the mesh is driving the arm; keep hands clear',
      }
    }
    return {
      severity: 'warn',
      word: 'locked',
      text: 'e-stop lockout — commands are refused and the arm is holding where the stop caught it; '
        + 'clearing the lockout is what makes it live again',
    }
  }

  const status = (f.taskStatus ?? '').trim().toLowerCase()
  const running = status === 'running'

  if (running) {
    const what = f.instruction ? ` ${quote(f.instruction)}` : ''
    const since = f.taskDurationS != null && f.taskDurationS >= 1
      ? `, ${Math.round(f.taskDurationS)}s in` : ''
    if (f.moving === false) {
      return {
        severity: 'warn',
        word: 'wedged?',
        text: `policy${what} says running but the arm is not moving${since} — wedged, or producing no-ops`,
      }
    }
    return {
      severity: 'active',
      word: 'running',
      text: `running${what}${since} — arm is under policy control, keep hands clear`,
    }
  }

  // Coming up.
  if (status === 'connecting') {
    return f.moving === true
      ? {
          severity: 'warn',
          word: 'starting',
          text: 'bringing the hardware up and the arm is ALREADY MOVING — homing or a queued command, keep hands clear',
        }
      : {
          severity: 'active',
          word: 'starting',
          text: 'bringing the hardware up — torque can engage and the arm move without warning, keep hands clear',
        }
  }

  // Idle per the task state - but is it actually still?
  if (f.moving === true) {
    return {
      severity: 'warn',
      word: 'moving',
      text: 'arm is MOVING with no task — teleop or another client is commanding it, keep hands clear',
    }
  }

  if (status === 'error') {
    return {
      severity: 'warn',
      word: 'failed',
      text: 'the task ended in ERROR — nothing is commanding the arm now, but it stopped wherever it '
        + 'got to; read the robot log before approaching',
    }
  }
  if (status === 'stopped') {
    // Deliberately not a warning: an operator pressing stop is normal, and an amber card after
    // every normal stop is alarm fatigue.
    return {
      severity: 'ok',
      word: 'stopped',
      text: 'the task was stopped before finishing — the arm is holding where it stopped, and a resume '
        + 'would move it from there',
    }
  }
  // An unrecognised status is NO EVIDENCE, and no evidence cannot earn the green sentence.
  if (status !== '' && status !== 'idle' && status !== 'completed') {
    return {
      severity: 'warn',
      word: 'unknown',
      text: `the robot reports task status ${quote(status, 24)} — this dashboard does not know that `
        + 'state, so stillness is not confirmed here',
    }
  }

  // THE THIRD LIE DETECTOR, and the one that was missing: "safe to approach" is a claim about
  // the physical world, so it must be EARNED by a motion measurement.
  if (f.moving == null || f.jointsSeen === false) {
    if (f.jointsSeen === false && f.hostsChildren && f.hostsChildren.length) {
      // A PROCESS IS NOT AN ARM (armHosts' law, which until now only the record screen knew).
      const kids = f.hostsChildren
      return {
        severity: 'ok',
        word: 'process',
        text: kids.length === 1
          ? `hosts ${kids[0]} — this is the process, not an arm; the joints are on that card`
          : `hosts ${kids.length} robots (${kids.join(', ')}) — this is the process, not an arm`,
      }
    }
    if (f.jointsSeen === false) {
      return {
        severity: 'warn',
        word: 'idle?',
        text: 'the robot reports idle, but it publishes no joint positions — stillness cannot be '
          + 'confirmed here, so treat the arm as able to move',
      }
    }
    // Transient: the ring needs ~1s of samples. Not a warning (every page load
    // would cry wolf), but it does not get to say "safe" either.
    return {
      severity: 'ok',
      word: 'idle',
      text: 'idle per the robot — motion not measured yet, a second of telemetry decides',
    }
  }

  return {
    severity: 'ok',
    word: 'idle',
    text: 'idle and still — safe to approach',
  }
}

/**
 * The ribbon renders the word as a coloured chip and the sentence beside it, so a sentence
 * that OPENS with that same word reads "IDLE idle and still…".
 */
export function ribbonDetail(line: StatusLine): string {
  const word = line.word.trim().toLowerCase()
  const text = line.text
  if (!word || !text.toLowerCase().startsWith(word)) return text
  // A WORD BOUNDARY IS REQUIRED: 'idle' must not eat the front of 'idling
  // along' and leave 'ing along'. Only a whole leading word is a duplicate.
  const after = text.charAt(word.length)
  if (after && /[A-Za-z0-9]/.test(after)) return text
  const rest = text.slice(word.length).replace(/^[\s\u2014:,-]+/, '')
  return rest.length > 0 ? rest : text
}

export function peerStatusFields(
  peer: { last_seen?: number | null; stale?: boolean | null; state?: any; lockout?: any; presence?: any },
  telemetry: { moving?: boolean | null; jointsSeen?: boolean | null; stateAgeS?: number | null },
  hostsChildren?: string[] | null,
): StatusFacts {
  const p = peer.presence
  return {
    stale: !!peer.stale,
    lastSeenAgoS: peer.last_seen ? Date.now() / 1000 - peer.last_seen : null,
    hwConnected: p?.connected ?? null,
    taskStatus: peer.state?.task?.status ?? p?.task_status ?? null,
    instruction: peer.state?.task?.instruction || p?.instruction || null,
    taskDurationS: peer.state?.task?.duration ?? null,
    moving: telemetry.moving ?? null,
    jointsSeen: telemetry.jointsSeen ?? null,
    stateAgeS: telemetry.stateAgeS ?? null,
    lockout: peer.lockout?.state ?? null,
    hostsChildren: hostsChildren ?? null,
  }
}
