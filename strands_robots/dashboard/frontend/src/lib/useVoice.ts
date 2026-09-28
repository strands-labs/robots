/** Give the microphone back. */
import { useCallback, useRef, useState } from 'react'
import { wsUrl } from './endpoints'
import { interpretVoiceEvent, voiceCloseState } from './voiceSession'
import type { VoiceState } from './voiceSession'

export type { VoiceState } from './voiceSession'

/** Browser mic (PCM16 @16k) ↔ /ws/voice ↔ bidi fleet agent. */
export function useVoice() {
  const [state, setState] = useState<VoiceState>('idle')
  const [transcript, setTranscript] = useState('')
  /** A refusal raised inside a voice turn is spoken once and gone. */
  const [need, setNeed] = useState<unknown | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const nodesRef = useRef<{ src?: MediaStreamAudioSourceNode; proc?: ScriptProcessorNode; stream?: MediaStream }>({})
  const playRef = useRef<{ ctx?: AudioContext; nextT: number; rate: number }>({ nextT: 0, rate: 24000 })

  /** Give the microphone back. */
  const releaseMic = useCallback(() => {
    nodesRef.current.proc?.disconnect()
    nodesRef.current.src?.disconnect()
    nodesRef.current.stream?.getTracks().forEach(t => t.stop())
    nodesRef.current = {}
    ctxRef.current?.close()
    ctxRef.current = null
  }, [])

  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: 'stop' }))
    wsRef.current?.close()
    wsRef.current = null
    releaseMic()
    playRef.current.ctx?.close()
    playRef.current.ctx = undefined
    setState('idle')
  }, [releaseMic])

  const start = useCallback(async () => {
    if (state === 'live' || state === 'connecting') { stop(); return }
    setState('connecting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
      // Same backend (and token) as every other channel - the dashboard API can be on another host
      // entirely.
      nodesRef.current = { stream }
      const ws = new WebSocket(wsUrl('/ws/voice'))
      wsRef.current = ws
      ws.onerror = () => releaseMic()

      ws.onmessage = (msg) => {
        try {
          /** What a frame MEANS lives in ./voiceSession, tested there. */
          const eff = interpretVoiceEvent(JSON.parse(msg.data))
          if (eff.rate !== undefined) playRef.current.rate = eff.rate
          if (eff.play !== undefined) playPcm(eff.play)
          /* Say it in the transcript too: the sheet may be behind a collapsed dock, and a spoken
             sentence the operator half-heard is not a record of what was refused. */
          if (eff.transcript !== undefined) setTranscript(eff.transcript)
          if ('need' in eff) setNeed(eff.need)
          if (eff.state !== undefined) setState(eff.state)
        } catch { /* a frame we could not even parse */ }
      }
      ws.onclose = (e) => {
        releaseMic()
        setState(s => {
          const v = voiceCloseState(e.code, s)
          if (v.transcript) setTranscript(v.transcript)
          return v.state
        })
      }
      ws.onopen = () => {
        setState('live')
        // mic capture → downsample to 16k PCM16 → binary WS
        const ctx = new AudioContext()
        ctxRef.current = ctx
        const src = ctx.createMediaStreamSource(stream)
        const proc = ctx.createScriptProcessor(4096, 1, 1)
        nodesRef.current = { ...nodesRef.current, src, proc, stream }
        const inRate = ctx.sampleRate
        src.connect(proc)
        proc.connect(ctx.destination)
        proc.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return
          const f32 = e.inputBuffer.getChannelData(0)
          const ratio = inRate / 16000
          const outLen = Math.floor(f32.length / ratio)
          const pcm = new Int16Array(outLen)
          for (let i = 0; i < outLen; i++) {
            const v = f32[Math.floor(i * ratio)]
            pcm[i] = Math.max(-32768, Math.min(32767, v * 32767))
          }
          ws.send(pcm.buffer)
        }
      }
    } catch (e) {
      // getUserMedia may have succeeded before whatever threw next.
      releaseMic()
      setTranscript(`⚠ ${e}`)
      setState('error')
    }
  }, [state, stop, releaseMic])

  const playPcm = (b64: string) => {
    const p = playRef.current
    if (!p.ctx) { p.ctx = new AudioContext(); p.nextT = p.ctx.currentTime }
    const raw = atob(b64)
    const pcm = new Int16Array(raw.length / 2)
    for (let i = 0; i < pcm.length; i++) pcm[i] = (raw.charCodeAt(2 * i) | (raw.charCodeAt(2 * i + 1) << 8)) << 16 >> 16
    const buf = p.ctx.createBuffer(1, pcm.length, p.rate)
    const ch = buf.getChannelData(0)
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768
    const srcN = p.ctx.createBufferSource()
    srcN.buffer = buf
    srcN.connect(p.ctx.destination)
    const t = Math.max(p.ctx.currentTime, p.nextT)
    srcN.start(t)
    p.nextT = t + buf.duration
  }

  return { state, transcript, toggle: start, stop, need, clearNeed: () => setNeed(null) }
}
