/** Which features on this page are DARK — routes the bundle calls that the running server does not have? */
import { routeKnown } from './serverAge'
import { BUNDLE_ROUTES } from './bundleRoutes.generated'

/**
 * Which features on this page are DARK — routes the bundle calls that the running server does
 * not have? lib/serverAge already explains one 404 after the fact.
 */
export function darkRoutes(livePaths: string[] | null | undefined, needed: readonly string[] = BUNDLE_ROUTES): string[] {
  if (!Array.isArray(livePaths) || livePaths.length === 0) return []
  return needed.filter(p => p.endsWith('/')
    // A BASE ('/api/auth/login/') is not a route: callers append a segment. It is satisfied by
    // anything beneath it, and judged as a route it would be permanently, wrongly dark.
    ? !livePaths.some(t => t.startsWith(p))
    : routeKnown(livePaths, p) === false).sort()
}

/** The banner sentence, or null when there is nothing to say. */
export function darkFeatureMessage(dark: readonly string[]): string | null {
  if (dark.length === 0) return null
  const n = dark.length
  return (
    `${n} ${n === 1 ? 'feature' : 'features'} on this page ${n === 1 ? 'is' : 'are'} dark: the server ` +
    `answering here is running older code and has no route for ${n === 1 ? 'it' : 'them'}. ` +
    `Restarting the dashboard from a terminal lights ${n === 1 ? 'it' : 'them'} up — not from a ` +
    `background daemon, which macOS can never grant camera access to.`
  )
}
