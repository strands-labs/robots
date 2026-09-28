/** "Why does this arm report no joints?" — asked ONCE per peer, shared by every screen that asks. */
import { useEffect, useState } from 'react'
import { api } from './endpoints'
import { jointFailure, jointFailureLine, jointFailureBadge, verdictIsStale, type JointFailure } from './jointFailure'

/**
 * "Why does this arm report no joints?" — asked ONCE per peer, shared by every screen that
 * asks.
 */
type Entry = { line: string | null; badge: string | null; at?: number }
const CACHE = new Map<string, Entry>()
const INFLIGHT = new Map<string, Promise<Entry>>()
const LISTENERS = new Set<() => void>()

/** A log we could not read is not a diagnosis — but silence on screen would be worse than saying so. */
const NO_LOG: Entry = {
  line: 'no joints, and no log to read — this arm was started outside the dashboard, so its reason is in the console that launched it',
  badge: 'no log to read — started outside the dashboard',
}

/**
 * A check that only runs every TICK_MS must treat a verdict that will expire BEFORE THE NEXT
 * CHECK as already expired.
 */
const TICK_MS = 30_000

async function load(peerId: string): Promise<Entry> {
  const cached = CACHE.get(peerId)
  // An old excuse must not outlive the fault: a respawn this UI never saw (no pid/started_at on
  // a peer) would otherwise keep accusing an arm that was fixed. verdictIsStale is the pure,
  // tested policy.
  if (cached && !verdictIsStale(cached.at, Date.now() + TICK_MS / 6)) return cached
  const running = INFLIGHT.get(peerId)
  if (running) return running
  const p = (async () => {
    let entry: Entry
    try {
      const r = await api<{ lines?: string[] }>(`/api/devices/logs/${encodeURIComponent(peerId)}`)
      const f: JointFailure | null = jointFailure(r?.lines)
      entry = { line: jointFailureLine(f), badge: jointFailureBadge(f), at: Date.now() }
    } catch {
      entry = { ...NO_LOG, at: Date.now() }
    }
    CACHE.set(peerId, entry)
    INFLIGHT.delete(peerId)
    for (const l of LISTENERS) l()
    return entry
  })()
  INFLIGHT.set(peerId, p)
  return p
}

/**
 * Called when an arm is respawned: the next question must reach the NEW log, not the old
 * verdict.
 */
export function forgetJointFailure(peerId?: string | null): void {
  if (arguments.length === 0) { CACHE.clear(); return }
  if (peerId) CACHE.delete(peerId)
}

/** @param enabled ask only when the joints are actually missing — a healthy arm needs no explanation. */
export function useJointFailure(peerId: string, enabled: boolean): Entry {
  const [entry, setEntry] = useState<Entry>(() => CACHE.get(peerId) ?? { line: null, badge: null })
  useEffect(() => {
    if (!enabled) { setEntry({ line: null, badge: null }); return }
    let live = true
    const seen = CACHE.get(peerId)
    if (seen) setEntry(seen)   // shown while a stale one is re-checked: never flicker to blank
    // While the arm stays mute, keep the answer honest — one cheap request per TTL, and only for an arm
    // that is STILL missing its joints (the hook is disabled the moment joints arrive).
    const tick = setInterval(() => { void load(peerId).then(e => { if (live) setEntry(e) }) }, TICK_MS)
    void load(peerId).then(e => { if (live) setEntry(e) })
    const onChange = () => { const e = CACHE.get(peerId); if (live && e) setEntry(e) }
    LISTENERS.add(onChange)
    return () => { live = false; clearInterval(tick); LISTENERS.delete(onChange) }
  }, [peerId, enabled])
  return entry
}
