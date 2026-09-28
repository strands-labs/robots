import { useEffect, useRef, useState } from 'react'
import { wsUrl, authRefusedRecently } from '../lib/endpoints'
import { classifyCamera, type CamStatus } from '../lib/cameraState'
import { planRetry, CHURN_OPENS_PER_MIN } from '../lib/cameraRetry'
import { authToken } from '../lib/endpoints'
import { sessionVerdict } from '../lib/sessionExpiry'
import { pacingFromNotice, nextRequestedFps } from '../lib/cameraPacing'

interface Meta { t?: number; shape?: number[]; encoding?: string; displayable?: boolean; error?: string }

/** Binary JPEG stream over /ws/camera/{peer}/{cam} → <img>. */
/** What a struggling viewer asks for: one frame a second still shows a moving arm. */
const DEGRADED_FPS = 1

export default function CameraTile({ peerId, cam, big = false, meta, onConfigure }: {
  peerId: string; cam: string; big?: boolean; meta?: Meta
  /** offered only for a robot this dashboard manages: opens this camera's settings */
  onConfigure?: () => void
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [status, setStatus] = useState<CamStatus>(() =>
    classifyCamera({ now: Date.now(), conn: 'connecting', frames: 0 }))
  const [fps, setFps] = useState(0)

  // Per-frame bookkeeping lives in refs: 30 setState calls a second per tile is
  // how six camera tiles turn a fluid dashboard into a slideshow.
  const frames = useRef(0)
  const lastFrameAt = useRef<number | undefined>(undefined)
  const times = useRef<number[]>([])
  const conn = useRef<'connecting' | 'open' | 'closed'>('connecting')
  const error = useRef<string | null>(null)
  const retryAt = useRef<number | undefined>(undefined)
  const openLog = useRef<number[]>([])
  /** the reduced rate this tile is currently asking for, or null at full rate */
  const degraded = useRef<number | null>(null)
  const pacedFps = useRef<number | null>(null)
  const [pacedNote, setPacedNote] = useState<string | null>(null)
  // Attempts survive a re-run of the effect: peerId/cam churn must not hand a dead
  // endpoint a fresh 1s retry budget.
  const tries = useRef(0)
  // Read inside the ticker, so a metadata update (meta.t changes with every published
  // frame) no longer tears the socket down and rebuilds it.
  const metaRef = useRef(meta)
  metaRef.current = meta

  useEffect(() => {
    let ws: WebSocket | null = null
    let url: string | null = null
    let stopped = false
    let retryTimer: number | undefined
    let openedAt: number | undefined
    let framesThisSocket = 0

    // One ticker owns every visible change, so a stall becomes visible on its
    // own: when a stream dies, nothing arrives to trigger a render.
    const tick = window.setInterval(() => {
      const now = Date.now()
      setStatus(classifyCamera({
        now, conn: conn.current, frames: frames.current,
        lastFrameAt: lastFrameAt.current, error: error.current,
        publishedAt: metaRef.current?.t,
        retryInMs: retryAt.current !== undefined ? Math.max(0, retryAt.current - now) : undefined,
        attempt: retryAt.current !== undefined ? tries.current : undefined,
      }))
      const t = performance.now()
      times.current = times.current.filter(x => t - x < 2000)
      setFps(times.current.length > 1 ? (times.current.length - 1) / ((t - times.current[0]) / 1000) : 0)
    }, 400)

    const open = () => {
      if (stopped) return
      conn.current = 'connecting'
      // Per-socket evidence must be per SOCKET: leaving a previous connection's openedAt in place
      // would let a later socket that never opens inherit a long openMs and clear the failure
      // history it is supposed to be proving.
      framesThisSocket = 0
      openedAt = undefined
      const capped = openLog.current.length >= CHURN_OPENS_PER_MIN
      degraded.current = nextRequestedFps(capped ? DEGRADED_FPS : null, pacedFps.current)
      const path = `/ws/camera/${encodeURIComponent(peerId)}/${encodeURIComponent(cam)}`
      const rate = degraded.current
      ws = new WebSocket(wsUrl(rate === null ? path : `${path}?max_fps=${rate}`))
      ws.binaryType = 'blob'
      // NOT a reset: this handshake succeeding says nothing about whether frames exist.
      ws.onopen = () => {
        conn.current = 'open'
        openedAt = Date.now()
        retryAt.current = undefined
        openLog.current.push(openedAt)
        while (openLog.current.length && openedAt - openLog.current[0] > 60_000) openLog.current.shift()
      }
      ws.onmessage = (msg) => {
        if (typeof msg.data === 'string') {
          // The server tells us *why* there are no pixels (e.g. the peer is
          // publishing raw frames it could not transcode).
          try {
            const ev = JSON.parse(msg.data)
            const pacing = pacingFromNotice(ev)
            if (pacing) {
              // NOT an error: the pictures keep coming, slower. Routing this into
              // `error` would paint a red "camera error" over a working tile and send
              // the operator hunting a USB cable that is fine.
              pacedFps.current = pacing.fps
              setPacedNote(pacing.note)
            } else if (ev.type === 'camera_error') error.current = ev.error
          } catch { /* ignore */ }
          return
        }
        if (!(msg.data instanceof Blob)) return
        const next = URL.createObjectURL(msg.data)
        if (imgRef.current) imgRef.current.src = next
        if (url) URL.revokeObjectURL(url)
        url = next
        error.current = null
        frames.current += 1
        framesThisSocket += 1
        lastFrameAt.current = Date.now()
        times.current.push(performance.now())
      }
      ws.onclose = (ev) => {
        conn.current = 'closed'
        const plan = planRetry({
          attempt: tries.current,
          frames: framesThisSocket,
          openMs: openedAt !== undefined ? Date.now() - openedAt : undefined,
          code: ev.code,
          recentOpens: openLog.current.length,
          sessionExpired: sessionVerdict(authToken(), Date.now() / 1000).refusesUntilSignIn,
          pageRefused: authRefusedRecently(),
        })
        tries.current = plan.attempt
        if (plan.delayMs === null) {
          // The tile's status line reads this; 'unauthorized' covers both a refused socket (1008)
          // and a lapsed sign-in, and plan.reason says which.
          error.current = plan.reason.includes('sign in') ? plan.reason : 'unauthorized'
          retryAt.current = undefined
          return
        }
        if (stopped) return
        retryAt.current = Date.now() + plan.delayMs
        retryTimer = window.setTimeout(open, plan.delayMs)
      }
    }
    open()

    return () => {
      stopped = true
      window.clearInterval(tick)
      if (retryTimer) window.clearTimeout(retryTimer)
      ws?.close()
      if (url) URL.revokeObjectURL(url)
    }
  }, [peerId, cam])

  const shape = meta?.shape?.length ? `${meta.shape[1]}×${meta.shape[0]}` : null
  const cls = ['camtile', big ? 'big' : '', `cam-${status.kind}`, status.frozen ? 'frozen' : '']
    .filter(Boolean).join(' ')

  return (
    <div className={cls}>
      <img ref={imgRef} alt={`${peerId} ${cam} camera`} />
      {!status.live && (
        // aria-live: a stream dying is news. A sighted user gets it from the
        // overlay appearing; a screen reader needs it announced.
        <div className="camstate" role="status" aria-live="polite">
          <b>{status.title}</b>
          {status.detail && <span>{status.detail}</span>}
          {status.frozen && <span className="camstale">showing the last frame received</span>}
        </div>
      )}
      <span className={status.live ? 'camlabel live' : 'camlabel'}>
        {cam}
        {status.live && fps > 0 && <em> {fps.toFixed(0)}fps</em>}
        {shape && <em> {shape}</em>}
      </span>
      {onConfigure && (
        <button className="camcfg" onClick={onConfigure}
                aria-label={`adjust ${cam} camera settings`}
                title={`adjust ${cam} — fps, size, device (applying restarts the robot)`}>
          ⚙
        </button>
      )}
      {pacedNote && (
        // Subdued, and NOT in the error overlay: the tile is alive. Announced politely
        // because a rate change explains a jerky picture a sighted user can see.
        <span className="campaced" role="status" aria-live="polite">{pacedNote}</span>
      )}
    </div>
  )
}
