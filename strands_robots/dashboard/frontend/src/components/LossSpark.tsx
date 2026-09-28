import { useEffect, useRef } from 'react'
import { lossBand, lossPath, fmtStep, type LossPoint } from '../lib/lossTrace'

/**
 * A training job's loss curve, drawn on canvas (same rationale as JointSpark: these live in a
 * polled list and must never become a DOM-heavy chart lib).
 */
export default function LossSpark({ trace, height = 34 }: { trace: LossPoint[]; height?: number }) {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const parent = canvas.parentElement
    const cssW = Math.max(1, parent?.clientWidth ?? 160)
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    canvas.width = Math.round(cssW * dpr)
    canvas.height = Math.round(height * dpr)
    canvas.style.width = `${cssW}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssW, height)
    const pts = lossPath(trace, cssW, height, 3)
    if (!pts.length) return
    ctx.strokeStyle = 'rgba(124, 192, 255, 0.9)'
    ctx.lineWidth = 1.5
    ctx.lineJoin = 'round'
    ctx.beginPath()
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)))
    ctx.stroke()
    // last point marker
    const [lx, ly] = pts[pts.length - 1]
    ctx.fillStyle = '#7cc0ff'
    ctx.beginPath()
    ctx.arc(lx, ly, 2.2, 0, Math.PI * 2)
    ctx.fill()
  }, [trace, height])

  if (trace.length < 2) {
    return (
      <div className="loss-spark empty">
        {trace.length === 1
          ? `first reading: loss ${trace[0].loss.toPrecision(3)} @ ${fmtStep(trace[0].step)} steps — curve appears as polling continues`
          : 'no loss readings yet'}
      </div>
    )
  }
  const first = trace[0]
  const last = trace[trace.length - 1]
  // A flat trace is drawn flat (lib/lossTrace lossBand pads the band rather than magnifying
  // noise), and SAYS so: the shape alone cannot distinguish "no measurable change" from "not
  // much happened", and the difference is whether the run is worth leaving to finish.
  const flat = lossBand(trace).flat
  return (
    <div className="loss-spark">
      <canvas ref={ref} />
      <div className="loss-spark-label">
        <span>loss {last.loss.toPrecision(3)}</span>
        <span className="dim">
          {fmtStep(first.step)} → {fmtStep(last.step)} steps (observed)
          {flat ? ' · flat across this window' : ''}
        </span>
      </div>
    </div>
  )
}
