import { useEffect, useState, useRef, useMemo } from 'react'
import { useDialogFocus } from '../lib/useDialogFocus'
import { authRemovalWarning } from '../lib/authRemoval'
import { connectionChange, needsConfirm, type ConnectionVerdict } from '../lib/connectionChange'
import { syncDrafts, dirtyFields, unsavedSummary, type Drafts } from '../lib/draftSync'
import type { MeshInfo } from '../types'
import {
  authToken, backendBase, backendLabel, normalize, post,
  setAuthToken, setBackendBase,
} from '../lib/endpoints'
import ConsentSettings from './ConsentSettings'
import PasskeyList from './PasskeyList'
import { useConfig, type ApplyResult } from '../lib/useConfig'
import {
  APPLY_LABEL, envKeyError, envValueError, searchSettings, settingMeta, validateSetting,
} from '../lib/settingsMeta'

/** Inline validation message + "what happens if I change this" chip for one field. */
function FieldMeta({ k, raw }: { k: string; raw: string }) {
  const meta = settingMeta(k)
  const err = validateSetting(k, raw)
  if (err) return <em className="field-err" role="alert">⚠ {err}</em>
  if (!meta) return null
  return (
    <em className="field-meta">
      {meta.effect}
      {meta.safeDefault !== '' && <> · default {meta.safeDefault}{meta.unit ? ` ${meta.unit}` : ''}</>}
      {' · '}<span className={`apply-chip ${meta.apply}`}>{APPLY_LABEL[meta.apply]}</span>
    </em>
  )
}

type Tab = 'connection' | 'agent' | 'voice' | 'mesh' | 'env' | 'security'

const DRAFT_LABELS: Record<string, string> = {
  modelId: 'the model id', prompt: 'the system prompt', temperature: 'temperature',
  maxTokens: 'max tokens', voiceProvider: 'the voice provider', voiceName: 'the voice',
  connect: 'mesh connect', listen: 'mesh listen', meshPort: 'the mesh port',
  meshBackend: 'the mesh transport', cameraHz: 'camera Hz', corsOrigins: 'CORS origins',
}

/** Which tab each draft field lives on — for the unsaved-work dot. */
const TAB_OF_FIELD: Record<string, string> = {
  modelId: 'agent', prompt: 'agent', temperature: 'agent', maxTokens: 'agent',
  voiceProvider: 'voice', voiceName: 'voice',
  connect: 'mesh', listen: 'mesh', meshPort: 'mesh', meshBackend: 'mesh', cameraHz: 'mesh',
  corsOrigins: 'security',
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'connection', label: 'Connection' },
  { id: 'agent', label: 'Agent' },
  { id: 'voice', label: 'Voice' },
  { id: 'mesh', label: 'Mesh' },
  { id: 'env', label: 'Env' },
  { id: 'security', label: 'Security' },
]

export default function SettingsDrawer({ open, onClose, mesh, initialTab }: {
  open: boolean; onClose: () => void; mesh: MeshInfo; initialTab?: Tab
}) {
  const { config, loading, error, reload, save } = useConfig()
  const [tab, setTab] = useState<Tab>('connection')
  const sheetRef = useRef<HTMLElement | null>(null)
  useDialogFocus(sheetRef, open)
  const [query, setQuery] = useState('')

  // Deep links (the header's wire-security chip opens straight to Mesh).
  useEffect(() => { if (open && initialTab) setTab(initialTab) }, [open, initialTab])
  const [status, setStatus] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  // --- connection (client-side only: which backend this browser talks to)
  const [base, setBase] = useState(backendBase())
  const [token, setToken] = useState(authToken())

  // --- server-side drafts, seeded from the loaded config
  const [modelId, setModelId] = useState('')
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [voiceProvider, setVoiceProvider] = useState('')
  const [voiceName, setVoiceName] = useState('')
  const [connect, setConnect] = useState('')
  const [listen, setListen] = useState('')
  const [meshPort, setMeshPort] = useState('')
  const [meshBackend, setMeshBackend] = useState('')
  const [cameraHz, setCameraHz] = useState('')
  const [trustRemote, setTrustRemote] = useState(false)
  const [envDraft, setEnvDraft] = useState<Record<string, string>>({})
  const [newKey, setNewKey] = useState('')
  const [newValue, setNewValue] = useState('')
  const [serverToken, setServerToken] = useState('')
  const [envUnset, setEnvUnset] = useState<string[]>([])
  const [removeArmed, setRemoveArmed] = useState(false)
  const [discardArmed, setDiscardArmed] = useState(false)
  const [corsOrigins, setCorsOrigins] = useState('')
  const [connVerdict, setConnVerdict] = useState<ConnectionVerdict | null>(null)

  const serverDrafts: Drafts = useMemo<Drafts>(() => {
    if (!config) return {} as Drafts
    const ms = (config.mesh.settings ?? {}) as Record<string, any>
    const out: Drafts = {
      modelId: config.agent.model_id ?? '',
      prompt: config.agent.system_prompt ?? '',
      temperature: config.agent.temperature === null ? '' : String(config.agent.temperature),
      maxTokens: config.agent.max_tokens === null ? '' : String(config.agent.max_tokens),
      voiceProvider: config.voice.provider,
      voiceName: config.voice.voice_name ?? '',
      connect: (ms.connect ?? []).join(', '),
      listen: (ms.listen ?? []).join(', '),
      meshPort: ms.port ? String(ms.port) : '',
      meshBackend: ms.backend ?? '',
      cameraHz: ms.camera_hz ? String(ms.camera_hz) : '',
      corsOrigins: (config.security.cors_origins ?? []).join(', '),
    }
    return out
  }, [config])

  const SETTERS: Record<string, (v: string) => void> = {
    modelId: setModelId, prompt: setPrompt, temperature: setTemperature, maxTokens: setMaxTokens,
    voiceProvider: setVoiceProvider, voiceName: setVoiceName, connect: setConnect, listen: setListen,
    meshPort: setMeshPort, meshBackend: setMeshBackend, cameraHz: setCameraHz, corsOrigins: setCorsOrigins,
  }
  const currentDrafts: Drafts = {
    modelId, prompt, temperature, maxTokens, voiceProvider, voiceName,
    connect, listen, meshPort, meshBackend, cameraHz, corsOrigins,
  }
  /* The snapshot the fields were seeded from — what "the operator has not touched it" means. */
  const seededRef = useRef<Drafts>({})
  const currentRef = useRef<Drafts>(currentDrafts)
  currentRef.current = currentDrafts

  useEffect(() => {
    if (!config) return
    const r = syncDrafts(currentRef.current, seededRef.current, serverDrafts)
    for (const [key, value] of Object.entries(r.next)) {
      if (currentRef.current[key] !== value) SETTERS[key]?.(value)
    }
    seededRef.current = serverDrafts
    if (r.conflicts.length) {
      // Never overwrite typing — but a draft pending against a value that MOVED is something only
      // the human can resolve, so it is said out loud instead of discovered on the next save.
      setStatus(
        `⚠ changed on the server while you were editing: ${r.conflicts.join(', ')} `
        + '— your version is still in the field, saving will overwrite theirs',
      )
    }
    setTrustRemote(config.runtime.trust_remote_code)
    setEnvDraft({})
    setServerToken('')
  }, [config, serverDrafts])

  if (!open) return null

  const dirty = dirtyFields(currentDrafts, serverDrafts)
  const unsaved = unsavedSummary(dirty, DRAFT_LABELS)
  const requestClose = () => {
    if (dirty.length) { setDiscardArmed(true); return }
    onClose()
  }

  const agentValid =
    validateSetting('agent.temperature', temperature) === null &&
    validateSetting('agent.max_tokens', maxTokens) === null
  const meshValid =
    validateSetting('mesh.port', meshPort) === null &&
    validateSetting('mesh.camera_hz', cameraHz) === null &&
    validateSetting('mesh.connect', connect) === null &&
    validateSetting('mesh.listen', listen) === null
  const envValid =
    envKeyError(newKey) === null && envValueError(newValue) === null &&
    Object.values(envDraft).every(v => envValueError(v) === null)
  const results = searchSettings(query)

  const report = (r: ApplyResult) => {
    const parts: string[] = []
    if (r.applied.length) parts.push(`applied ${r.applied.join(', ')}`)
    if (r.env_written.length) parts.push(`wrote ${r.env_written.join(', ')} to .env`)
    if (r.env_removed?.length) {
      parts.push(
        `removed ${r.env_removed.join(', ')} from .env `
        + '(gone for this process and every robot spawned from now on; already-running robots keep it)',
      )
    }
    if (r.agent_reset) parts.push('agent will rebuild on the next turn')
    if (r.skipped_masked.length) parts.push(`skipped unchanged secrets: ${r.skipped_masked.join(', ')}`)
    if (r.restart_required.length) parts.push(`needs a mesh restart: ${r.restart_required.join(', ')}`)
    if (r.startup_required?.length) {
      parts.push(
        `saved, takes effect at the next server start: ${r.startup_required.join(', ')} `
        + '(a removed origin is already refused for writes and websockets)',
      )
    }
    if (r.respawn_required?.length) {
      parts.push(
        `saved for robots spawned from now on: ${r.respawn_required.join(', ')} `
        + '(respawn a robot to change its rate)',
      )
    }
    if (r.mesh_restart) {
      parts.push(r.mesh_restart.mesh_online ? 'mesh re-pointed' : 'mesh re-point FAILED (offline)')
      if (r.mesh_restart.orphaned?.length) parts.push(`orphaned local robots: ${r.mesh_restart.orphaned.join(', ')}`)
    }
    if (r.errors.length) parts.push(`errors: ${r.errors.join('; ')}`)
    // A name the backend does not recognise is dropped without an error, so
    // without this line "nothing changed" covers both "already correct" and
    // "I did not understand what you sent".
    if (r.ignored?.length) parts.push(`⚠ not recognised, so not saved: ${r.ignored.join(', ')}`)
    setStatus(parts.join(' · ') || 'nothing changed')
  }

  const apply = async (body: Record<string, any>) => {
    setSaving(true); setStatus(null)
    try {
      report(await save(body))
    } catch (e: any) {
      setStatus(`⚠ ${e?.message ?? String(e)}`)
    } finally {
      setSaving(false)
    }
  }

  const goConnect = (tokenToSend: string) => {
    setBackendBase(base)
    setAuthToken(tokenToSend)
    // Remounting the app is the point: sockets, peer map and frame buffers all
    // belong to the backend we were talking to.
    location.reload()
  }

  const applyConnection = () => {
    const v = connectionChange({
      currentBase: backendBase(),
      currentToken: authToken(),
      nextBase: normalize(base) || base,
      nextToken: token,
      pageHost: typeof location !== 'undefined' ? location.host : '',
    })
    if (v.kind === 'ok') { setConnVerdict(null); goConnect(token); return }
    setConnVerdict(v)  // unparseable = a refusal; the other two are questions
  }

  const restartMesh = async (force = false) => {
    setSaving(true); setStatus(null)
    try {
      const r = await post<{ mesh_online: boolean; orphaned: string[] }>('/api/mesh/restart', { force })
      setStatus(r.mesh_online ? 'mesh re-opened' : '⚠ mesh is offline after restart')
      await reload()
    } catch (e: any) {
      setStatus(`⚠ ${e?.message ?? String(e)}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="drawer-backdrop" onClick={requestClose}>
      <aside ref={sheetRef} className="drawer" onClick={e => e.stopPropagation()}>
        <header className="drawer-head">
          <h2>Settings</h2>
          <button className="btn ghost" onClick={requestClose} aria-label="close settings" title="Escape">✕</button>
        </header>

        {discardArmed && (
          <div className="result bad" role="alert">
            <b>{unsaved}</b>
            <p>Closing now discards {dirty.length === 1 ? 'it' : 'them'}. Save the tab you were editing first, or discard.</p>
            <div className="sheet-actions">
              <button className="btn go" onClick={() => setDiscardArmed(false)}>keep editing</button>
              <button className="btn ghost danger" onClick={() => { setDiscardArmed(false); onClose() }}>
                discard and close
              </button>
            </div>
          </div>
        )}

        <div className="settings-search">
          <input
            type="search"
            placeholder="Search settings… (fps, token, prompt)"
            aria-label="Search settings"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          {results.length > 0 && (
            <ul className="search-results" role="listbox">
              {results.map(r => (
                <li key={r.key}>
                  <button
                    role="option"
                    onClick={() => { setTab(r.tab); setQuery('') }}
                  >
                    <b>{r.label}</b> <span className="tabname">{r.tab}</span>
                    <em>{r.effect}</em>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {query.trim() !== '' && results.length === 0 && (
            <p className="hint">no setting matches "{query.trim()}"</p>
          )}
        </div>

        <nav className="tabs">
          {TABS.map(t => {
            const tabDirty = dirty.some(k => (TAB_OF_FIELD[k] ?? '') === t.id)
            return (
              <button key={t.id} className={tab === t.id ? 'tab on' : 'tab'} aria-pressed={tab === t.id}
                      onClick={() => setTab(t.id)}
                      title={tabDirty ? 'unsaved changes on this tab' : undefined}>
                {t.label}{tabDirty && <em className="warn" aria-label="unsaved changes"> •</em>}
              </button>
            )
          })}
        </nav>

        {loading && !config && <div className="drawer-body"><p className="hint">loading…</p></div>}
        {error && <div className="drawer-body"><div className="result bad">⚠ {error}</div></div>}

        <div className="drawer-body">
          {tab === 'connection' && (
            <section>
              <h3>Backend</h3>
              <p className="hint">
                Currently talking to <b>{backendLabel()}</b>. The dashboard API can run on any
                machine — leave this empty to use the origin that served this page.
              </p>
              <label className="field">
                <span>API base URL</span>
                <input
                  placeholder={`${location.host} (this origin)`}
                  value={base}
                  onChange={e => setBase(e.target.value)}
                  onBlur={e => setBase(normalize(e.target.value) || e.target.value)}
                />
              </label>
              <label className="field">
                <span>Auth token (this browser)</span>
                <input type="password" placeholder="only if the server requires one"
                       value={token} onChange={e => setToken(e.target.value)} />
              </label>
              <div className="sheet-actions">
                <button className="btn go" onClick={applyConnection}>connect &amp; reload</button>
                {(base || token) && (
                  <button className="btn ghost" onClick={() => { setBase(''); setToken(''); setConnVerdict(null) }}>clear</button>
                )}
              </div>
              {/* a credential belongs to a host. */}
              {connVerdict && connVerdict.kind !== 'ok' && (
                <div className="result bad" role="alert">
                  <b>{connVerdict.kind === 'unparseable' ? 'That address cannot be dialled' : 'Send this token there?'}</b>
                  <p>{connVerdict.detail}</p>
                  {needsConfirm(connVerdict) && (
                    <div className="sheet-actions">
                      <button className="btn ghost danger" onClick={() => { setConnVerdict(null); goConnect(token) }}>
                        send it anyway
                      </button>
                      <button className="btn go" onClick={() => {
                        // The safe path must be one click too, or nobody takes it.
                        setToken(''); setConnVerdict(null); goConnect('')
                      }}>{'alternative' in connVerdict ? connVerdict.alternative : 'connect without a token'}</button>
                      <button className="btn ghost" onClick={() => setConnVerdict(null)}>cancel</button>
                    </div>
                  )}
                </div>
              )}
              <p className="hint">
                Tip: <code>?backend=https://robot.lan:8080&amp;token=…</code> in the URL sets both,
                so a bookmark or QR code points a phone straight at one robot.
              </p>
            </section>
          )}

          {tab === 'agent' && config && (
            <section>
              <h3>Fleet agent</h3>
              <p className="hint">
                {config.agent.built ? 'built' : 'not built yet'} ·{' '}
                {config.agent.busy ? 'busy' : 'idle'} · {config.agent.messages ?? 0} messages ·
                tools: {(config.agent.tools ?? []).join(', ') || 'fleet'}
                {config.agent.bridge_online === false && ' · ⚠ mesh bridge not attached'}
              </p>
              <label className="field">
                <span>Model id</span>
                <input list="known-models" value={modelId} placeholder="(provider default)"
                       onChange={e => setModelId(e.target.value)} />
                <datalist id="known-models">
                  {config.agent.known_models.map(m => <option key={m} value={m} />)}
                </datalist>
                <FieldMeta k="agent.model_id" raw={modelId} />
              </label>
              <div className="row">
                <label className="field">
                  <span>Temperature</span>
                  <input type="number" step="0.1" min="0" max="2" value={temperature}
                         placeholder="default" onChange={e => setTemperature(e.target.value)} />
                  <FieldMeta k="agent.temperature" raw={temperature} />
                </label>
                <label className="field">
                  <span>Max tokens</span>
                  <input type="number" min="1" value={maxTokens}
                         placeholder="default" onChange={e => setMaxTokens(e.target.value)} />
                  <FieldMeta k="agent.max_tokens" raw={maxTokens} />
                </label>
              </div>
              <label className="field">
                <span>System prompt {config.agent.is_default_prompt && <em>(default)</em>}</span>
                <textarea rows={10} value={prompt} onChange={e => setPrompt(e.target.value)} />
              </label>
              <div className="sheet-actions">
                <button className="btn go" disabled={saving || !agentValid}
                        title={agentValid ? undefined : 'fix the highlighted fields first'}
                        onClick={() => apply({
                  agent: {
                    model_id: modelId || null,
                    system_prompt: prompt,
                    temperature: temperature === '' ? null : Number(temperature),
                    max_tokens: maxTokens === '' ? null : Number(maxTokens),
                  },
                })}>save</button>
                <button className="btn ghost" disabled={saving}
                        onClick={() => apply({ reset_prompt: true })}>reset prompt</button>
                <button className="btn ghost" disabled={saving}
                        onClick={() => apply({ reset_agent: true, clear_history: true })}>
                  clear conversation
                </button>
              </div>
              <p className="hint">
                Model and prompt changes take effect on the next turn. Sampling knobs are applied
                to the resolved model, so a provider that ignores one will say so in the log.
              </p>
            </section>
          )}

          {tab === 'voice' && config && (
            <section>
              <h3>Voice</h3>
              <label className="field">
                <span>Provider</span>
                <select value={voiceProvider} onChange={e => setVoiceProvider(e.target.value)}>
                  {config.voice.providers.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
                <FieldMeta k="voice.provider" raw={voiceProvider} />
              </label>
              <label className="field">
                <span>Voice name</span>
                <input value={voiceName} placeholder="provider default"
                       onChange={e => setVoiceName(e.target.value)} />
                <FieldMeta k="voice.voice_name" raw={voiceName} />
              </label>
              <div className="sheet-actions">
                <button className="btn go" disabled={saving} onClick={() => apply({
                  voice: { provider: voiceProvider, voice_name: voiceName || null },
                })}>save</button>
              </div>
              <p className="hint">
                Each provider needs its own credential in the Env tab
                (<code>OPENAI_API_KEY</code>, <code>GOOGLE_API_KEY</code>, or AWS for Nova Sonic).
              </p>
            </section>
          )}

          {tab === 'mesh' && (
            <section>
              <h3>Mesh</h3>
              <dl className="kv">
                {/* Same tri-state trap as the motion chip: `online` is optional, and a ternary reported a MISSING field as a definite "offline". */}
                <dt>status</dt><dd>{mesh.online === false
                  ? 'offline'
                  : mesh.online
                    ? `online as ${mesh.peer_id}`
                    : 'not reported yet'}</dd>
                <dt>peers</dt><dd>{mesh.live_peers ?? 0} live / {mesh.peers ?? 0} known</dd>
                <dt>wire security</dt>
                <dd className={mesh.local_dev ? 'bad' : 'ok'}>{mesh.wire_security ?? 'unknown'}</dd>
                <dt>backend</dt><dd>{mesh.backend ?? 'zenoh'}</dd>
                <dt>cmd cap</dt><dd>{mesh.max_cmd_bytes ?? 0} B</dd>
                {mesh.policy_allow?.length ? (
                  <><dt>policy allowlist</dt><dd className="mono">{mesh.policy_allow.join(', ')}</dd></>
                ) : null}
              </dl>

              {mesh.local_dev && (
                <div className="explain">
                  <b>What "wire security off" means:</b> robot commands and camera frames travel
                  the mesh unencrypted and unauthenticated (<code>STRANDS_MESH_LOCAL_DEV=1</code>).
                  That is fine on a trusted home LAN. Before this network is shared or bridged,
                  restart the dashboard <em>without</em> that env var so the mesh requires mTLS —
                  then only <code>tls/</code> and <code>quic/</code> endpoints are accepted.
                  This is separate from dashboard login, which already guards the web UI.
                </div>
              )}

              <div className="sheet-actions">
                <button className="btn ghost" disabled={saving} title="Fills the fields below - nothing is saved until you click save & re-point" onClick={() => {
                  setMeshPort(''); setConnect(''); setListen(''); setMeshBackend(''); setCameraHz('15')
                  setStatus('SO-101 preset filled in below - review, then "save & re-point"')
                }}>
                  ✦ recommended for SO-101 desk setup
                </button>
              </div>
              <p className="hint">
                Preset: multicast discovery (no endpoints), default port 7447, camera 15 Hz —
                smooth preview for two USB arm cams without flooding the LAN. It only fills the
                form; nothing applies until you save.
              </p>

              <label className="field">
                <span>Connect endpoints</span>
                <input placeholder="tls/robot.lan:7447, tls/10.0.0.5:7447"
                       value={connect} onChange={e => setConnect(e.target.value)} />
                <FieldMeta k="mesh.connect" raw={connect} />
              </label>
              <label className="field">
                <span>Listen endpoints</span>
                <input placeholder="tls/0.0.0.0:7447"
                       value={listen} onChange={e => setListen(e.target.value)} />
                <FieldMeta k="mesh.listen" raw={listen} />
              </label>
              <div className="row">
                <label className="field">
                  <span>Port</span>
                  <input type="number" value={meshPort} placeholder="7447"
                         onChange={e => setMeshPort(e.target.value)} />
                  <FieldMeta k="mesh.port" raw={meshPort} />
                </label>
                <label className="field">
                  <span>Transport</span>
                  <select value={meshBackend} onChange={e => setMeshBackend(e.target.value)}>
                    <option value="">zenoh (default)</option>
                    <option value="iot">AWS IoT Core</option>
                    <option value="bridge">bridge</option>
                  </select>
                </label>
                <label className="field">
                  <span>Camera Hz</span>
                  <input type="number" step="1" value={cameraHz} placeholder="default"
                         onChange={e => setCameraHz(e.target.value)} />
                  <FieldMeta k="mesh.camera_hz" raw={cameraHz} />
                </label>
              </div>
              {!mesh.local_dev && (
                <p className="hint">
                  With mTLS wire security, only <code>tls/</code> and <code>quic/</code> endpoints are
                  accepted — a <code>tcp/</code> endpoint is refused at session open.
                </p>
              )}
              <div className="sheet-actions">
                <button className="btn go" disabled={saving || !meshValid}
                        title={meshValid ? undefined : 'fix the highlighted fields first'}
                        onClick={() => apply({
                  mesh: {
                    connect, listen,
                    port: meshPort === '' ? null : Number(meshPort),
                    backend: meshBackend || null,
                    camera_hz: cameraHz === '' ? null : Number(cameraHz),
                  },
                  restart_mesh: true,
                })}>save &amp; re-point</button>
                <button className="btn ghost" disabled={saving} onClick={() => restartMesh(false)}>
                  restart mesh
                </button>
              </div>
              <p className="hint">
                Re-pointing re-opens the shared mesh session. Locally spawned robots hold their own
                reference to the old one, so the server refuses unless they are despawned — or you
                force it and accept that they stay on the old endpoints.
              </p>
              <div className="sheet-actions">
                <button className="btn ghost danger" disabled={saving} onClick={() => restartMesh(true)}>
                  force restart (orphan local robots)
                </button>
              </div>
            </section>
          )}

          {tab === 'env' && config && (
            <section>
              <h3>Environment</h3>
              <p className="hint">
                Written to <code>{config.env_file}</code> (chmod 600). Secrets show masked; leaving a
                mask untouched leaves the stored value alone.{' '}
                {/* clearing a field writes KEY= — set and EMPTY, which almost nothing treats like absent (getenv returns "", an empty token authenticates as an empty token). */}
                Clearing a value stores an <em>empty</em> value; use <b>unset</b> to remove the
                variable entirely.
              </p>
              {config.env.some(r => r.shadowed) && (
                <p className="hint warn">
                  {config.env.filter(r => r.shadowed).map(r => r.key).join(', ')}{' '}
                  {config.env.filter(r => r.shadowed).length > 1 ? 'were' : 'was'} exported into
                  this process before it started, and that value wins over .env for as long as it
                  runs — saving here updates the file, not this run.
                </p>
              )}
              <div className="envlist">
                {config.env.map(row => (
                  <label className="field env" key={row.key}>
                    <span>
                      {row.key}
                      {row.secret && <em title="masked on read"> 🔒</em>}
                      {!row.in_file && row.set && <em title="from the process environment"> (env)</em>}
                      {/* .env is loaded at startup, but a value exported into the launch environment WINS. */}
                      {row.shadowed && (
                        <em className="warn" title="this process was launched with a different value, which wins over .env until it is restarted without it">
                          {' '}(shell overrides .env)
                        </em>
                      )}
                    </span>
                    <input
                      // Shown as text on purpose: the value is already masked
                      // server-side, and a password field would hide *which*
                      // characters are the mask.
                      value={envDraft[row.key] ?? row.value}
                      placeholder={row.set ? '' : 'not set'}
                      disabled={envUnset.includes(row.key)}
                      onChange={e => setEnvDraft(d => ({ ...d, [row.key]: e.target.value }))}
                    />
                    {row.in_file && (
                      envUnset.includes(row.key)
                        ? <em className="warn">will be removed on save —{' '}
                            <button className="btn ghost tiny"
                                    onClick={() => setEnvUnset(u => u.filter(k => k !== row.key))}>
                              keep it
                            </button>
                          </em>
                        : <button className="btn ghost tiny" title="remove this variable from the file"
                                  onClick={() => setEnvUnset(u => [...u, row.key])}>unset</button>
                    )}
                  </label>
                ))}
              </div>
              <div className="row">
                <label className="field">
                  <span>New key</span>
                  <input value={newKey} placeholder="MY_API_KEY"
                         onChange={e => setNewKey(e.target.value.toUpperCase())} />
                  {envKeyError(newKey) && <em className="field-err" role="alert">⚠ {envKeyError(newKey)}</em>}
                </label>
                <label className="field">
                  <span>Value</span>
                  <input value={newValue} onChange={e => setNewValue(e.target.value)} />
                  {envValueError(newValue) && <em className="field-err" role="alert">⚠ {envValueError(newValue)}</em>}
                </label>
              </div>
              <label className="field check">
                <input type="checkbox" checked={trustRemote} onChange={e => setTrustRemote(e.target.checked)} />
                <span>
                  Allow HuggingFace <code>trust_remote_code</code> (lerobot_local, kimodo) —
                  executes code from the model repo
                </span>
              </label>
              <div className="sheet-actions">
                <button className="btn go" disabled={saving || !envValid}
                        title={envValid ? undefined : 'fix the highlighted fields first'}
                        onClick={() => {
                  const env: Record<string, string | null> = { ...envDraft }
                  // A key marked unset wins over any draft edit to it: the operator's last word was
                  // "remove", and sending both would write the value and then delete the line.
                  for (const k of envUnset) env[k] = null
                  if (newKey.trim()) env[newKey.trim()] = newValue
                  void apply({ env, runtime: { trust_remote_code: trustRemote } })
                  setNewKey(''); setNewValue(''); setEnvUnset([])
                }}>save</button>
              </div>
            </section>
          )}

          {tab === 'security' && config && (
            <section>
              <h3>Security</h3>
              <div className={config.security.auth_enabled ? 'result ok' : 'result bad'}>
                {config.security.auth_enabled
                  ? '✓ a token is required on /api and /ws'
                  : '⚠ no auth: anyone who can reach this port can move motors'}
              </div>
              {/* "A token is required" can be TRUE and still not safe: the token itself may be sitting in this process's argv, where `ps` shows it to every local user (found exactly that way on this Mac — a pgrep for the pid printed the live token). */}
              {config.security.notice?.text && (
                <div className="result warn" data-notice={config.security.notice.kind}>
                  <span>⚠ {config.security.notice.text}</span>
                  {config.security.notice.remedy && (
                    <span className="hint">{config.security.notice.remedy}</span>
                  )}
                </div>
              )}
              <label className="field">
                <span>Server auth token</span>
                <input type="password" value={serverToken} placeholder={config.security.auth_enabled ? '•••••• (set)' : 'not set'}
                       onChange={e => setServerToken(e.target.value)} />
              </label>
              <label className="field">
                <span>CORS origins</span>
                <input value={corsOrigins} placeholder="* (any origin)"
                       onChange={e => setCorsOrigins(e.target.value)} />
              </label>
              <div className="sheet-actions">
                <button className="btn go" disabled={saving} onClick={async () => {
                  await apply({ security: { auth_token: serverToken || null, cors_origins: corsOrigins } })
                  // Lock ourselves out otherwise: the token we just set is what
                  // every subsequent request must carry.
                  if (serverToken) { setAuthToken(serverToken); setToken(serverToken) }
                }}>save</button>
                {config.security.auth_enabled && !removeArmed && (
                  <button className="btn ghost danger" disabled={saving}
                          onClick={() => setRemoveArmed(true)}>
                    remove token
                  </button>
                )}
              </div>
              {config.security.auth_enabled && removeArmed && (() => {
                const w = authRemovalWarning({
                  host: typeof location !== 'undefined' ? location.hostname : '',
                  corsOrigins,
                  // the mesh panel's own number: live peers, not merely remembered ones
                  peerCount: mesh.live_peers ?? mesh.peers ?? 0,
                })
                return (
                  <div className={w.severity === 'exposed' ? 'result bad' : 'result'} role="alert">
                    <b>Remove the token?</b>
                    <ul>{w.lines.map(l => <li key={l}>{l}</li>)}</ul>
                    <div className="sheet-actions">
                      <button className="btn ghost danger" disabled={saving} onClick={() => {
                        setRemoveArmed(false)
                        void apply({ security: { auth_token: null } })
                      }}>{w.confirmLabel}</button>
                      <button className="btn ghost" disabled={saving}
                              onClick={() => setRemoveArmed(false)}>keep it</button>
                    </div>
                  </div>
                )
              })()}
              <p className="hint">
                CORS origins apply at startup; the token applies immediately (and is saved into this
                browser so you are not locked out). <code>STRANDS_MESH_LOCAL_DEV=1</code> is separate
                and disables mesh <em>wire</em> security — see the Mesh tab.
              </p>
              {/* The consent sheet promises a way back; this is it (U18). */}
              <ConsentSettings />
              {/* the same question as the permissions list, about a sharper subject — which DEVICES hold a key to this dashboard. */}
              <PasskeyList authRequired={Boolean(config?.security?.auth_enabled)} />
            </section>
          )}
        </div>

        {status && <footer className="drawer-foot">{status}</footer>}
      </aside>
    </div>
  )
}
