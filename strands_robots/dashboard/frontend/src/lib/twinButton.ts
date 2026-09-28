/** R2 / UX_REVIEW:61 — the button that SPAWNS A PROCESS was labelled `⿻`. */

export interface TwinButtonCopy {
  label: string
  title: string
  aria: string
  /** extra class: 'on' while a twin is live, '' otherwise */
  cls: string
  /** aria-pressed for the toggle. */
  pressed: boolean
}

export function twinButtonCopy(o: { peerId: string; twinLive: boolean; busy?: boolean }): TwinButtonCopy {
  const twinId = `${o.peerId}-twin`
  if (o.busy) {
    return {
      label: '…',
      cls: o.twinLive ? 'on' : '',
      pressed: !!o.twinLive,
      title: `waiting for ${twinId} — a sim peer takes a moment to start or stop`,
      aria: `sim twin of ${o.peerId}: working`,
    }
  }
  if (o.twinLive) {
    return {
      label: 'twin on',
      cls: 'on',
      pressed: true,
      title: `${twinId} is running: tasks sent to this robot are mirrored to it. `
        + `Click to stop the twin — the real arm is not affected either way.`,
      aria: `stop the sim twin of ${o.peerId}`,
    }
  }
  return {
    label: '+ twin',
    cls: '',
    pressed: false,
    title: `Start ${twinId}, a simulated copy of this arm as its own peer. `
      + `Tasks you send this robot are mirrored to it, so you can watch a policy `
      + `in sim before trusting it on metal. The real arm is not touched.`,
    aria: `start a sim twin of ${o.peerId}`,
  }
}
