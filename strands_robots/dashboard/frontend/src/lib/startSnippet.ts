/** The code the empty home screen hands the operator. */
export interface DetectedBoard {
  /** the OS device path, e.g. /dev/cu.usbmodem5AB0181806 */
  device: string
  /** the robot family it was last spawned as, when this board is remembered */
  robot_name?: string | null
}

export interface StartSnippet {
  /** the code, ready to copy */
  code: string
  /** one line under it: where the port came from, or that there isn't one */
  provenance: string
  /** true when the port in the code is a real detected device */
  real: boolean
}

const SIM = 'Robot("so101").run()   # sim, no hardware needed'

export function startSnippet(boards: DetectedBoard[] | null | undefined): StartSnippet {
  const real = (boards ?? []).find(b => b.device)
  if (!real) {
    return {
      code: `from strands_robots import Robot\n${SIM}\nRobot("so101", mode="real", port="/dev/ttyACM0").run()`,
      // Said out loud: no board was detected here, so that path is an EXAMPLE. The old snippet made
      // the same claim silently and it was wrong on this very machine.
      provenance: 'no servo board is plugged into this machine, so the port above is an example — ' +
        'a real one appears here once a board is detected',
      real: false,
    }
  }
  const family = (real.robot_name || 'so101').trim() || 'so101'
  return {
    code: `from strands_robots import Robot\nRobot("${family}").run()   # sim, no hardware needed\n` +
      `Robot("${family}", mode="real", port="${real.device}").run()`,
    // Naming the port's origin keeps it falsifiable: a port that moved is a visible claim, not a
    // mysterious failure inside lerobot.
    provenance: `the port is the board detected on this machine right now (${real.device})` +
      (real.robot_name ? ` — last spawned as "${family}"` : ''),
    real: true,
  }
}
