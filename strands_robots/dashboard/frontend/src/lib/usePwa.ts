/** Page plumbing: install prompt, online state, screen wake lock. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { bundleAgeText } from './swUpdate'
import { wakeLockAction, wakeLockNote } from './wakeLock'

/**
 * Install prompt, online state, screen wake lock. There is NO service worker: the
 * bundle is served by the robot's own process under /static/, and a page that
 * moves motors must never answer from a cache or replay a queued request. A new
 * build is picked up by a plain reload, so `needRefresh` is always false and
 * `update` is a reload.
 */
export function usePwa() {
  const needRefresh = false

  const [online, setOnline] = useState(navigator.onLine)
  // When this bundle started running, so the prompt can say how long they have
  // been on the old one instead of a bare "a new version is available".
  const loadedAtRef = useRef<number>(Date.now())
  const [installable, setInstallable] = useState(false)
  const promptRef = useRef<any>(null)
  const wakeRef = useRef<any>(null)
  const wantAwakeRef = useRef(false)

  useEffect(() => {
    const up = () => setOnline(true)
    const down = () => setOnline(false)
    window.addEventListener('online', up)
    window.addEventListener('offline', down)

    const onPrompt = (e: Event) => {
      e.preventDefault()          // keep our own chip instead of the mini-infobar
      promptRef.current = e
      setInstallable(true)
    }
    window.addEventListener('beforeinstallprompt', onPrompt)
    const onInstalled = () => { setInstallable(false); promptRef.current = null }
    window.addEventListener('appinstalled', onInstalled)

    return () => {
      window.removeEventListener('online', up)
      window.removeEventListener('offline', down)
      window.removeEventListener('beforeinstallprompt', onPrompt)
      window.removeEventListener('appinstalled', onInstalled)
    }
  }, [])

  const install = useCallback(async () => {
    const prompt = promptRef.current
    if (!prompt) return
    promptRef.current = null
    setInstallable(false)
    try { await prompt.prompt() } catch { /* user dismissed */ }
  }, [])

  const update = useCallback(() => { location.reload() }, [])

  /**
   * Hold the screen awake while any robot is running. A phone that sleeps mid-task drops the
   * camera sockets and the operator loses sight of a moving arm - the one moment the screen must
   * stay on.
   */
  const applyWakeLock = useCallback(async () => {
    const anyNav = navigator as any
    const action = wakeLockAction({
      want: wantAwakeRef.current,
      held: !!wakeRef.current,
      visible: document.visibilityState === 'visible',
      supported: !!anyNav.wakeLock,
    })
    if (action === 'request') {
      try {
        wakeRef.current = await anyNav.wakeLock.request('screen')
        // The browser releases the lock itself when the page is hidden; this keeps `held` honest so
        // the next visibility change knows to take it again.
        wakeRef.current.addEventListener?.('release', () => { wakeRef.current = null })
      } catch { /* denied, or the page went hidden mid-request */ }
    } else if (action === 'release') {
      try { await wakeRef.current.release() } catch { /* already gone */ }
      wakeRef.current = null
    }
  }, [])

  const keepAwake = useCallback(async (want: boolean) => {
    wantAwakeRef.current = want
    await applyWakeLock()
  }, [applyWakeLock])

  useEffect(() => {
    const onVisible = () => { void applyWakeLock() }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [applyWakeLock])

  const standalone = window.matchMedia('(display-mode: standalone)').matches
    || (navigator as any).standalone === true

  return {
    online, needRefresh, update, installable, install, keepAwake, standalone,
    /** honest word about the screen: null when there is nothing to say (see lib/wakeLock) */
    wakeNote: () => wakeLockNote({
      want: wantAwakeRef.current,
      held: !!wakeRef.current,
      visible: document.visibilityState === 'visible',
      supported: !!(navigator as any).wakeLock,
    }),
    /** how long this tab has been running the bundle it loaded, for the update prompt */
    bundleAge: () => bundleAgeText(loadedAtRef.current, Date.now()),
  }
}
