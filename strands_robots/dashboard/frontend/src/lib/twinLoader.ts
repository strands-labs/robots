/**
 * The three.js twin lives outside the bundle, at /static/twin.js, with its own
 * vendored three.module.min.js and OrbitControls beside it (static/vendor/).
 * It is loaded once, as a module script at runtime, so the 700 KB of three.js
 * is fetched only by a page that opens the Sim tab, and the file the Python
 * package ships is the one the browser runs - no second copy in this bundle.
 */
import { apiUrl, authToken } from './endpoints'

export interface TwinInstance {
  load(): Promise<unknown>
  poses(buffer: ArrayBuffer): void
  frame(): void
  toggleGroup(group: number): void
  resize(): void
  dispose(): void
}

type TwinCtor = new (
  canvas: HTMLCanvasElement,
  sessionId: string,
  opts?: { base?: string; headers?: Record<string, string> },
) => TwinInstance

let _module: Promise<{ Twin: TwinCtor }> | null = null

/** The URL the server actually mounts the file at, independent of the bundle's own base. */
export const TWIN_URL = '/static/twin.js'

export function loadTwinModule(): Promise<{ Twin: TwinCtor }> {
  if (!_module) {
    // A plain dynamic import of a runtime URL: /* @vite-ignore */ keeps the bundler
    // from trying to resolve and inline it.
    _module = import(/* @vite-ignore */ TWIN_URL).catch(e => {
      _module = null // let the next open retry rather than remember a blip forever
      throw e
    })
  }
  return _module
}

/** Build a twin for one session, carrying the page's backend base and bearer token. */
export async function makeTwin(canvas: HTMLCanvasElement, sessionId: string): Promise<TwinInstance> {
  const { Twin } = await loadTwinModule()
  const token = authToken()
  const base = apiUrl('')
  return new Twin(canvas, sessionId, { base, headers: token ? { Authorization: `Bearer ${token}` } : {} })
}
