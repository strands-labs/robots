/** Fleet-side answer to "where did my robot go?" — children we hold a process for that the snapshot no longer lists. */
import { deathVerdict } from './childDeath'

export interface AbsentChild {
  peer_id: string
  robot_name?: string | null
  mode?: string | null
  returncode?: number | null
  started_at?: number | null
}

export interface AbsentNotice {
  /** the single line the fleet bar shows */
  headline: string
  /** one line per absent child — the tooltip, and what a test can read */
  detail: string
  /** how many children this notice speaks for */
  count: number
}

/** The cause without its explanatory clause — the bar has one line, the tooltip has room. */
export function shortCause(returncode?: number | null): string {
  return deathVerdict(returncode).phrase.split(' — ')[0]
}

export function absentNotice(children: readonly AbsentChild[] | null | undefined): AbsentNotice | null {
  // An older server sends no field at all.
  if (!Array.isArray(children) || children.length === 0) return null
  const named = children.filter(c => c && typeof c.peer_id === 'string' && c.peer_id.length > 0)
  // A child that exited 0 finished its job: a recording that completed and left the mesh is
  // EXPECTED, and a bar that announces expected things gets ignored, taking the surprises with
  // it.
  const surprises = named.filter(c => c.returncode !== 0)
  if (surprises.length === 0) return null
  const detail = surprises
    .map(c => `${c.peer_id} — ${deathVerdict(c.returncode).phrase}`)
    .join('\n')
  const headline =
    surprises.length === 1
      // The name FIRST: the operator is looking for a specific robot, not a count.
      ? `${surprises[0].peer_id} is gone — ${shortCause(surprises[0].returncode)}`
      : `${surprises.length} robots you started are gone`
  return { headline, detail, count: surprises.length }
}

export function quietNotice(
  ids: readonly string[] | null | undefined,
  dead: readonly AbsentChild[] | null | undefined = [],
): AbsentNotice | null {
  // An older server sends no field: absent means "this server cannot tell you", which renders
  // nothing.
  if (!Array.isArray(ids) || ids.length === 0) return null
  const buried = new Set(
    (Array.isArray(dead) ? dead : []).map(c => c && c.peer_id).filter(Boolean) as string[],
  )
  // The server derives these from the LIVE managed set, so an id cannot be in both lists.
  const quiet = ids.filter(id => typeof id === 'string' && id.length > 0 && !buried.has(id))
  if (quiet.length === 0) return null
  const detail = quiet
    .map(id => `${id} — the process is running, but it has never joined the fleet`)
    .join('\n')
  const headline =
    quiet.length === 1
      // Name first: the operator is hunting one robot, not counting.
      ? `${quiet[0]} started but never joined the fleet`
      : `${quiet.length} robots you started never joined the fleet`
  return { headline, detail, count: quiet.length }
}
