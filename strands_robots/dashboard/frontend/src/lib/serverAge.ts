/** Is a 404 the server being OLDER than this bundle? */

/** Turn "/api/devices/camera/{index}/modes" into a matcher for "/api/devices/camera/2/modes". */
function templateMatches(template: string, path: string): boolean {
  if (!template.includes('{')) return template === path
  const parts = template.split(/\{[^}]*\}/)
  const rx = new RegExp('^' + parts.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('[^/]+') + '$')
  return rx.test(path)
}

/** The path a request actually asked for, without query or fragment or a trailing slash. */
export function normalisePath(path: string): string {
  const bare = path.split('#')[0].split('?')[0]
  return bare.length > 1 && bare.endsWith('/') ? bare.slice(0, -1) : bare
}

/** Does the running server route this path at all? */
export function routeKnown(livePaths: string[] | null | undefined, path: string): boolean | null {
  if (!Array.isArray(livePaths) || livePaths.length === 0) return null
  const wanted = normalisePath(path)
  return livePaths.some(t => templateMatches(t, wanted))
}

/** The sentence for a 404 whose route the server does not have. */
export function staleRouteMessage(path: string): string {
  return (
    `this dashboard's server does not have ${normalisePath(path)} — it is running code from before ` +
    `this feature existed. Restart the dashboard from a terminal to pick it up (the page itself is ` +
    `already new; a restart from an agent or daemon would come back with no camera access).`
  )
}

/**
 * The server ADMITTING, in its own 404 body, that it routes nothing at this path. routeKnown()
 * is the authoritative test and is preferred, but it needs /openapi.json — when that cannot be
 * read it returns null and this module deliberately stays silent rather than blame the server
 * for our own missing fetch.
 */
export function unroutedByDetail(detail: unknown): boolean {
  if (typeof detail !== 'string') return false
  const d = detail.trim().toLowerCase()
  return d.startsWith('no endpoint at') || d === 'not found'
}
