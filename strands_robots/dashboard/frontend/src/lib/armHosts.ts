/**
 * A PROCESS is not an arm. Which peers on the mesh can actually be recorded from. Peers in
 * this fleet are named `parent` and `parent__child`: a process that HOSTS robots publishes
 * presence under its own id, and each robot it holds publishes as a child.
 */

/** A parent peer whose arm is somewhere else, with the child to pick instead. */
export interface HostVerdict {
  /** the child peers published under it */
  children: string[]
  /** the sentence for a disabled option / a note */
  why: string
}

/** Is `child` published under `parent`? Their separator is a double underscore. */
export function isChildOf(child: string, parent: string): boolean {
  return !!parent && !!child && child.startsWith(`${parent}__`) && child.length > parent.length + 2
}

export interface HostInput {
  peer_id: string
  /** how many joints this peer itself reports — 0 or null when none */
  joints?: number | null
}

/**
 * Which of these peers are hosts rather than arms. @returns a map keyed by peer id; a peer
 * absent from the map is offerable.
 */
export function armHosts(peers: HostInput[] | null | undefined): Record<string, HostVerdict> {
  const list = (peers ?? []).filter(p => p && p.peer_id)
  const out: Record<string, HostVerdict> = {}
  for (const p of list) {
    const children = list.map(x => x.peer_id).filter(id => isChildOf(id, p.peer_id))
    if (!children.length) continue
    // Evidence first: a parent that reports its own joints stays a candidate.
    if (typeof p.joints === 'number' && p.joints > 0) continue
    out[p.peer_id] = {
      children,
      why: children.length === 1
        ? `hosts ${children[0]} — pick that, this is the process`
        : `hosts ${children.length} robots (${children.join(', ')}) — pick one of those, this is the process`,
    }
  }
  return out
}
