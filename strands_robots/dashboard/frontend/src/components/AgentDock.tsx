import { useEffect, useRef, useState } from 'react'
import { useVoice } from '../lib/useVoice'
import { api, wsUrl, authRefusedRecently, noteAuthRefusal } from '../lib/endpoints'
import { useConfig } from '../lib/useConfig'
import { sendFailureVerdict, interruptionNotice, bubbleLabel } from '../lib/chatDelivery'
import { turnAnnouncement } from '../lib/agentAnnounce'
import {
  type MotionConfirm, parseInterruptEvent, parseStatusInterrupt,
  confirmQuestion, confirmDetail, answerNotice, interruptResponseBody,
} from '../lib/interruptConfirm'

/** GET /api/agent: which model the console will use and which tools ask first. */
interface AgentInfo { model?: string; asks_first?: string[]; interrupt?: string }
import ConsentSheet from './ConsentSheet'
import { type ConsentNeed } from '../lib/consent'

interface ChatMsg {
  role: 'user' | 'agent' | 'notice'
  text: string
  reasoning?: string
  tools?: { name: string; status: string }[]
  /** user bubbles only: false = it provably never left the browser. ABSENT
   *  means delivered, which is the normal case and needs no decoration. */
  delivered?: boolean
  /** notice bubbles: render as a failure, not as information. */
  bad?: boolean
}

/**
 * Bottom-docked fleet agent: text chat + speech-to-speech toggle.
 *
 * Speaks /ws/agent (routes_agent.py): the page sends {type:'say', text} to start
 * a turn and {type:'resume', id, approve, always} to answer a consent card; the
 * console streams {type: text|tool_use|tool_result|interrupt|done|error}. One
 * conversation per socket, so "clear" is simply a new socket.
 */
export default function AgentDock({ onSettings, startOpen = false, exampleRobot }: {
  onSettings?: () => void
  /** true when launched from the manifest's "Ask the agent" shortcut. */
  startOpen?: boolean
  /**
   * a real online robot to name in examples - the placeholder is the de-facto tutorial, and it
   * should teach a one-robot command before a fleet-wide one, with a name that actually exists
   * on this desk.
   */
  exampleRobot?: string
}) {
  const [open, setOpen] = useState(startOpen)
  const [input, setInput] = useState('')
  const [msgs, setMsgs] = useState<ChatMsg[]>([])
  const [busy, setBusy] = useState(false)
  const [connError, setConnError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  // onclose fires from a closure that captured `busy` at socket-open time, so the
  // "was a turn in flight?" question has to be asked of a ref, not of the state.
  const busyRef = useRef(false)
  // The trailing agent bubble as it stands right now, for the same reason: a
  // close handler must judge the answer it actually interrupted.
  const lastAgentRef = useRef<{ chars: number; running: string[] }>({ chars: 0, running: [] })
  const scrollRef = useRef<HTMLDivElement>(null)
  // A refused turn: the guard's decision, plus the sentence to re-send if it is granted.
  const [need, setNeed] = useState<ConsentNeed | null>(null)
  // The agent is PAUSED on a motion confirm: its turn resumes with the answer.
  const [confirm, setConfirm] = useState<MotionConfirm | null>(null)
  const refusedPrompt = useRef<string>('')
  const voice = useVoice()
  const { config } = useConfig()
  const agent = config?.agent
  const [info, setInfo] = useState<AgentInfo | null>(null)
  useEffect(() => {
    api<AgentInfo>('/api/agent').then(setInfo).catch(() => setInfo(null))
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    const last = msgs[msgs.length - 1]
    lastAgentRef.current = last?.role === 'agent'
      ? { chars: last.text.length, running: (last.tools ?? []).filter(t => t.status === 'running').map(t => t.name) }
      : { chars: 0, running: [] }
  }, [msgs])

  useEffect(() => { busyRef.current = busy }, [busy])

  // A reload with a confirm parked server-side re-renders the question instead of wedging.
  // Answered ids are remembered so a stale /api/config snapshot cannot resurrect the dialog.
  const answeredIds = useRef<Set<string>>(new Set())
  const statusInterrupt = (agent as any)?.interrupt
  useEffect(() => {
    if (!statusInterrupt || busy) return
    const c = parseStatusInterrupt(statusInterrupt)
    if (c && !answeredIds.current.has(c.id)) { setConfirm(prev => prev ?? c); setOpen(true) }
  }, [statusInterrupt, busy])

  /** Answer the motion confirm: the SAME turn resumes with the decision. */
  const answer = async (approve: boolean, always = false) => {
    const c = confirm
    if (!c) return
    answeredIds.current.add(c.id)
    setConfirm(null)
    setMsgs(prev => [...prev, { role: 'notice', text: answerNotice(c, approve, always) }])
    setBusy(true)
    try {
      const ws = await ensureWs()
      ws.send(JSON.stringify(interruptResponseBody(c.id, approve, always)))
    } catch (e: any) {
      setBusy(false)
      setConfirm(c) // the question is still pending server-side - keep it answerable
      setConnError(e?.message ?? String(e))
    }
  }

  /** Append to the trailing agent bubble, creating one if the last is not ours. */
  const patchAgent = (fn: (m: ChatMsg) => void) => {
    setMsgs(prev => {
      const next = [...prev]
      let last = next[next.length - 1]
      if (!last || last.role !== 'agent') {
        last = { role: 'agent', text: '', tools: [] }
        next.push(last)
      } else {
        last = { ...last, tools: [...(last.tools ?? [])] }
        next[next.length - 1] = last
      }
      fn(last)
      return next
    })
  }

  const ensureWs = (): Promise<WebSocket> => new Promise((resolve, reject) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return resolve(wsRef.current)
    const ws = new WebSocket(wsUrl('/ws/agent'))
    ws.onopen = () => { wsRef.current = ws; setConnError(null); resolve(ws) }
    ws.onerror = () => reject(new Error(
      authRefusedRecently()
        ? 'the agent socket was refused — this page is not signed in any more, sign in again'
        : 'could not reach the agent socket',
    ))
    ws.onmessage = (msg) => {
      let ev: any
      try { ev = JSON.parse(msg.data) } catch { return }
      // A notice is about the conversation itself, not an answer - putting it in
      // the agent bubble makes the model look like it said it.
      if (ev.type === 'notice') {
        setMsgs(prev => [...prev, { role: 'notice', text: ev.text }])
        setOpen(true)
        return
      }
      if (ev.type === 'pong') return
      if (ev.type === 'interrupt') {
        // The turn is parked server-side; the question replaces the spinner.
        const c = parseInterruptEvent(ev)
        if (c) { setConfirm(c); setBusy(false); setOpen(true) }
        return
      }
      if (ev.type === 'tool_result' && ev.needs_consent) setNeed(ev.needs_consent as ConsentNeed)
      patchAgent(last => {
        if (ev.type === 'text') last.text += ev.text
        else if (ev.type === 'reasoning') last.reasoning = (last.reasoning ?? '') + (ev.text ?? ev.data ?? '')
        else if (ev.type === 'tool_use') last.tools!.push({ name: ev.name, status: 'running' })
        else if (ev.type === 'tool_result') {
          // Results arrive in call order; the first still-running chip is the one this answers.
          const t = last.tools!.find(t => t.status === 'running')
          if (t) t.status = ev.status === 'error' ? 'error' : 'done'
        }
        else if (ev.type === 'done') setBusy(false)
        else if (ev.type === 'error') { last.text += `\n⚠ ${ev.message ?? ev.error}`; setBusy(false) }
      })
      // A reply landing in a collapsed dock is a reply the user never sees -
      // the transcript is the product here, so incoming activity reopens it.
      setOpen(true)
    }
    ws.onclose = ev => {
      wsRef.current = null
      setBusy(false)
      // 4401 is the dashboard's own "sign in required" close: it goes to the
      // same door an HTTP 401 opens (AuthGate re-verifies and shows the login),
      // instead of only a notice in the transcript.
      if (ev.code === 4401) noteAuthRefusal(401)
      const verdict = interruptionNotice({
        code: ev.code,
        wasBusy: busyRef.current,
        partialChars: lastAgentRef.current.chars,
        runningTools: lastAgentRef.current.running,
      })
      if (!verdict) return
      if (busyRef.current) {
        setMsgs(prev => [...prev, { role: 'notice', text: verdict.text, bad: true }])
        setOpen(true)
      } else {
        setConnError(verdict.text.replace(/^⚠ /, ''))
      }
    }
  })

  const send = async (retryText?: string) => {
    const text = (retryText ?? input).trim()
    if (!text || busy) return
    if (!retryText) setInput('')
    refusedPrompt.current = text
    setMsgs(prev => [...prev, { role: 'user', text }])
    setBusy(true)
    setConnError(null)
    try {
      const ws = await ensureWs()
      ws.send(JSON.stringify({ type: 'say', text }))
    } catch (e: any) {
      setBusy(false)
      const verdict = sendFailureVerdict({ error: e?.message ?? String(e) })
      // The message never left the browser: mark THAT bubble (it must not read as delivered) and
      // give the operator their text back rather than making them retype a sentence the UI silently
      // swallowed.
      setMsgs(prev => prev.map((m, i) =>
        i === prev.length - 1 && m.role === 'user' ? { ...m, delivered: false } : m))
      if (verdict.retrySafe) setInput(text)
      setConnError(verdict.text.replace(/^⚠ /, ''))
    }
  }

  /** One conversation lives exactly as long as its socket: forgetting it is closing it. */
  const clearHistory = async () => {
    const ws = wsRef.current
    wsRef.current = null
    busyRef.current = false
    try { ws?.close(1000, 'cleared') } catch { /* already gone */ }
    setMsgs([])
    setConfirm(null)
    setBusy(false)
    setConnError(null)
  }

  return (
    <>
      {open && (
        <div className="dock-panel">
          <div className="dock-head">
            <span className="chip static" title="the model the console answers with">
              🤖 {info?.model || agent?.model_id || 'default model'}
            </span>
            {busy && <span className="chip static">turn in flight</span>}
            {info?.asks_first?.length ? (
              <span className="chip static" title={`asks before: ${info.asks_first.join(', ')}`}>
                {info.asks_first.length} tools ask first
              </span>
            ) : null}
            {agent?.bridge_online === false && (
              <span className="badge danger" title="the agent has no mesh bridge - its fleet tools cannot reach any robot">
                no mesh
              </span>
            )}
            <span className="spacer" />
            <button className="btn ghost" onClick={clearHistory} title="Forget the conversation">clear</button>
            {onSettings && <button className="btn ghost" onClick={onSettings} title="Model & prompt">⚒</button>}
          </div>
          {/* a `log` (a transcript that appends), NAMED so it can be found, and with live updates explicitly OFF. patchAgent appends deltas token by token, so a live region here would stutter the reply word by word and re-interrupt itself for its whole length — unstoppable and unreadable. */}
          <div
            className="dock-scroll"
            ref={scrollRef}
            role="log"
            aria-label="conversation with the fleet agent"
            aria-live="off"
            aria-busy={busy || undefined}
          >
            {msgs.length === 0 && (
              <div className="dock-hint">
                Ask the fleet agent anything:<br />
                <em>"what robots are online?"</em><br />
                <em>"tell {exampleRobot ?? 'so101-arm-1'} to wave hello"</em><br />
                <em>"everyone stop" — the safety brake, it halts every robot</em>
                <p className="hint">
                  {/* this line used to read "it can start and stop real robots", which was true and was the problem — that path had no confirmation of any kind. */}
                  It can stop any robot, and start tasks in simulation. Starting a real arm stays with
                  you — press ▶ on its card. Everything it does is recorded in Activity.
                </p>
              </div>
            )}
            {msgs.map((m, i) => (
              m.role === 'notice' ? (
                <div key={i} className={m.bad ? 'dock-notice bad' : 'dock-notice'}>{m.bad ? '' : 'ⓘ '}{m.text}</div>
              ) : (
                <div key={i} className={`bubble ${m.role}${m.delivered === false ? ' undelivered' : ''}`}>
                  {m.tools?.map((t, j) => (
                    <span key={j} className={`toolchip ${t.status}`}>⚙ {t.name}</span>
                  ))}
                  {m.reasoning && (
                    <details className="reasoning"><summary>thinking</summary><pre>{m.reasoning}</pre></details>
                  )}
                  <div>{m.text || (busy && i === msgs.length - 1 ? '…' : '')}</div>
                  {bubbleLabel(m.delivered) && (
                    <div className="bubble-foot">
                      <span className="badge warn">{bubbleLabel(m.delivered)}</span>
                      <button className="btn tiny" onClick={() => void send(m.text)} disabled={busy}>
                        send again
                      </button>
                    </div>
                  )}
                </div>
              )
            ))}
          </div>
          {confirm && (
            <div className="dock-notice confirm" role="alertdialog" aria-label="motion confirmation">
              <div><strong>▶ {confirmQuestion(confirm)}</strong></div>
              {confirmDetail(confirm) && <div className="hint">{confirmDetail(confirm)}</div>}
              <div className="bubble-foot">
                <button className="btn" onClick={() => void answer(true)} autoFocus>
                  yes, run it
                </button>
                <button className="btn ghost" onClick={() => void answer(true, true)}
                        title="approve this and every later motion on the same target for this conversation">
                  yes, and stop asking for this one
                </button>
                <button className="btn ghost" onClick={() => void answer(false)}>
                  no - nothing moves
                </button>
              </div>
            </div>
          )}
          {/* One sentence per finished turn — the only thing spoken automatically. aria-atomic because a partial re-read of a reply is not a reply. */}
          <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
            {turnAnnouncement({ busy, last: msgs[msgs.length - 1], error: connError })}
          </div>
          {connError && <div className="dock-notice bad">⚠ {connError}</div>}
          {voice.transcript && <div className="voice-transcript">{voice.transcript}</div>}
        </div>
      )}
      <div className="dock-bar">
        <button
          className={`mic ${voice.state}`}
          onClick={() => { setOpen(true); voice.toggle() }}
          title="Speech-to-speech fleet control"
        >
          {voice.state === 'live' ? '🔴' : voice.state === 'connecting' ? '⏳' : '🎙'}
        </button>
        <input
          placeholder={confirm ? 'answer the motion confirm above first' : `ask the fleet agent… (e.g. '${exampleRobot ?? 'so101-arm-1'}, wave hello')`}
          aria-label="message to the fleet agent"
          value={input}
          onFocus={() => setOpen(true)}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') void send() }}
          disabled={busy || !!confirm}
        />
        {/* "↑" is not a name. */}
        <button className="dock-send" onClick={() => void send()} disabled={busy || !!confirm || !input.trim()}
                aria-label="send to the agent" title="send to the agent">↑</button>
        <button className="dock-min" onClick={() => setOpen(o => !o)}
                aria-label={open ? 'hide the conversation' : 'show the conversation'}
                title={open ? 'hide the conversation' : 'show the conversation'}>
          {open ? '▾ hide' : `▴ chat${msgs.length ? ` (${msgs.length})` : ''}`}
        </button>
      </div>

      {/* The refusal names a permission; this makes it a decision, in the place the operator was already looking. target='spawn' because a chat turn CAN simply be re-sent once the grant lands — no process holds a stale env (the fleet tool reads it per call), unlike a running peer that needs a respawn. */}
      {(need ?? voice.need) ? (
        <ConsentSheet
          need={(need ?? voice.need) as ConsentNeed}
          target="spawn"
          onCancel={() => { setNeed(null); voice.clearNeed() }}
          onRetry={() => {
            const again = refusedPrompt.current
            setNeed(null)
            /* A voice refusal has no sentence to re-send — the operator is mid-conversation and can
               simply say it again, now that it is allowed. Only a typed turn is replayed. */
            if (voice.need) voice.clearNeed()
            else void send(again)
          }}
        />
      ) : null}
    </>
  )
}
