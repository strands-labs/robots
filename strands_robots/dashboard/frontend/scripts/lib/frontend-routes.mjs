/**
 * THE ONE EXTRACTOR: which /api and /ws paths does the frontend source name, and where?
 *
 * Two scripts ask this, for different comparisons:
 *   · check-routes-exist.mjs — does the PYTHON SOURCE serve each of them (a typo, knowable at build time)
 *   · gen-bundle-routes.mjs  — writes the list the PAGE compares against a RUNNING server (Q124's
 *                              dark-feature banner: an older server, knowable only at runtime)
 * They had a regex each, and the strip rules had ALREADY drifted: one removed whole-line comments only,
 * the other trailing ones too. Drift matters because these feed opposite conclusions — a path the gate
 * ignores but the generator keeps becomes a banner accusing the running server of missing a route
 * nobody calls, and a path the generator drops is a dark feature nobody is warned about.
 */
import fs from 'node:fs'
import path from 'node:path'

/** Every .ts/.tsx file under dir, minus tests and the file the generator itself writes. */
export function sourceFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) return sourceFiles(p)
    if (!/\.(ts|tsx)$/.test(e.name)) return []
    if (e.name.includes('.test.') || e.name === 'bundleRoutes.generated.ts') return []
    return [p]
  })
}

/**
 * Comments are not callers. A JSDoc line that MENTIONS a route in backticks (`/api/record`) is
 * documentation — the first version of the typo gate reported rehearsalNav.ts's prose as a missing
 * route, and the generator's own header quotes paths it must not then attribute to the bundle.
 * Trailing comments count as comments, but '://' must survive: it is a URL, not a comment.
 */
export function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')
}

/** raw literal -> Set of basenames that name it. Raw means `${expr}` holes are still visible. */
export function routeSites(dir) {
  const used = new Map()
  for (const f of sourceFiles(dir)) {
    for (const m of stripComments(fs.readFileSync(f, 'utf8')).matchAll(/['"`](\/(?:api|ws)\/[^'"`]*)['"`]/g)) {
      if (!used.has(m[1])) used.set(m[1], new Set())
      used.get(m[1]).add(path.basename(f))
    }
  }
  return used
}
