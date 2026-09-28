import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { useMesh } from './lib/useMesh'
import { usePwa } from './lib/usePwa'
import { linkHealth, estopPosture } from './lib/linkHealth'
import LanHint from './components/LanHint'
import { darkRoutes, darkFeatureMessage } from './lib/darkFeatures'
import { lockoutBanner } from './lib/lockoutBadge'
import { reloadImpact } from './lib/swUpdate'
import { ConfigProvider } from './lib/useConfig'
import { authToken, backendKey, backendLabel, setAuthToken, subscribeAuth, serverRoutePaths } from './lib/endpoints'
import { sessionVerdict } from './lib/sessionExpiry'
import { serverNotice, staleServerNotice, fleetFieldGaps, type RefusedHandshakes } from './lib/serverNotice'
import FleetBar from './components/FleetBar'
import { getRecordApi } from './lib/recordApi'
import RobotCard from './components/RobotCard'
import { armHosts } from './lib/armHosts'
import RobotDetail from './components/RobotDetail'
import AgentDock from './components/AgentDock'
import SettingsDrawer from './components/SettingsDrawer'
import ActivityLog from './components/ActivityLog'
import DevicePanel from './components/DevicePanel'
import { noArmsVerdict, type RememberedBoard } from './lib/noArms'
import { startSnippet, type DetectedBoard } from './lib/startSnippet'
import { api as httpGet } from './lib/endpoints'
import EstopSheet from './components/EstopSheet'
import HelpSheet from './components/HelpSheet'
import EstopButton from './components/EstopButton'
import { hotkeyVerdict } from './lib/hotkeys'
import ErrorBoundary from './components/ErrorBoundary'
import TrainingTab from './components/TrainingTab'
import SimTab from './components/SimTab'
import RecordPanel from './components/RecordPanel'
import AuthGate from './components/AuthGate'

type Panel = 'settings' | 'activity' | 'devices' | 'estop' | 'training' | 'record' | 'sim' | 'help' | null

const PANELS: readonly Exclude<Panel, null>[] = ['settings', 'activity', 'devices', 'estop', 'training', 'record', 'sim', 'help']

/** The panel a fragment names: `#devices` -> 'devices', `#fleet` / empty -> null (the fleet is the page). */
function panelFromHash(hash: string): Panel {
  const want = hash.replace(/^#/, '')
  return (PANELS as readonly string[]).includes(want) ? (want as Panel) : null
}

/** `?panel=…` is what the manifest shortcuts deep-link to; the fragment wins when both are present. */
function initialPanel(): Panel {
  const fromHash = panelFromHash(location.hash)
  if (fromHash) return fromHash
  const want = new URLSearchParams(location.search).get('panel')
  return (PANELS as readonly string[]).includes(want ?? '') ? (want as Panel) : null
}

function Dashboard() {
  const { conn, dashboardId, peers, safetyFlash, mesh, activity, absentChildren, quietChildren, loaded, lastEventAt, everOpen } = useMesh()
  const pwa = usePwa()
  const [panel, setPanel] = useState<Panel>(initialPanel)

  // One router. The three ways into a view - a click, the loaded URL's fragment, a
  // fragment change (bookmark, link, Back, Forward) - all call route(), and route() is the
  // only writer of the panel state, so the address bar and the screen cannot disagree.
  // The fragment is written on every change; a hashchange that names the view already
  // shown (route()'s own write coming back) is skipped.
  function route(next: Panel) {
    setPanel(next)
    const want = next ?? 'fleet'
    if (location.hash.replace(/^#/, '') !== want) location.hash = want
  }
  const shownPanel = useRef<Panel>(panel)
  shownPanel.current = panel
  useEffect(() => {
    route(initialPanel())
    const onHash = () => {
      const next = panelFromHash(location.hash)
      if (shownPanel.current !== next) route(next)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  // the record screen's parting gift: the dataset it just finished, seeded into training
  const [trainPrefill, setTrainPrefill] = useState<{ dataset_root?: string } | undefined>(undefined)
  /** The verdict of copying the first-run snippet. */
  const [snipCopied, setSnipCopied] = useState<string | null>(null)
  const [boards, setBoards] = useState<(RememberedBoard & DetectedBoard)[] | null | undefined>(undefined)
  const [settingsTab, setSettingsTab] = useState<'mesh' | undefined>(undefined)
  const [detail, setDetail] = useState<string | null>(null)
  const [busyPeers, setBusyPeers] = useState<Record<string, boolean>>({})
  /**
   * UX_REVIEW #10: the record backend is probed ONCE per page load (lib/recordApi caches it), so
   * asking here costs nothing and lets the nav warn before the click instead of after. null
   * until the answer arrives — the nav must not guess in either direction.
   */
  const [recordMock, setRecordMock] = useState<boolean | null>(null)

  /** R2: which robots currently have a live sim twin. */
  const liveTwins = useMemo(() => new Set(
    Object.values(peers).filter(p => p.peer_id.endsWith('-twin') && !p.stale).map(p => p.peer_id),
  ), [peers])

  const list = useMemo(() => Object.values(peers)
    // Infrastructure peers are not robots: 'gateway' meshes (robot_mesh's robot-less fallback, one
    // per coordinating agent process) and the dashboard's own '-safety' signer have no cameras, no
    // joints, and no dispatch - a run form on them is a card that can only fail.
    .filter(p => {
      const t = p.presence?.robot_type
      if (t === 'gateway') return false
      if (p.peer_id.endsWith('-safety')) return false
      return true
    })
    .sort((a, b) => {
      // fresh first, then robots before sims, then name
      if (!!a.stale !== !!b.stale) return a.stale ? 1 : -1
      const at = a.presence?.robot_type ?? 'z', bt = b.presence?.robot_type ?? 'z'
      if (at !== bt) return at === 'robot' ? -1 : 1
      return a.peer_id.localeCompare(b.peer_id)
    }), [peers])

  const anyRunning = Object.values(busyPeers).some(Boolean)

  // A phone that sleeps mid-task drops the camera sockets, exactly when the
  // operator most needs to see a moving arm.
  const fleetEmpty = loaded && list.length === 0
  // undefined (not asked yet) is passed as a failed lookup: while the request is in flight,
  // "could not be reached" is the honest reading and never becomes "nothing is configured".
  const snippet = startSnippet(boards ?? null)
  const homeRoute = fleetEmpty
    ? noArmsVerdict(0, boards === undefined ? null : boards)?.route ?? null
    : null
  useEffect(() => {
    if (!fleetEmpty || boards !== undefined) return
    let alive = true
    httpGet<{
      serial_ports?: { device: string; remembered?: { peer_id: string; robot_name?: string | null } | null }[]
      managed?: Record<string, { alive?: boolean; port?: string }>
    }>('/api/devices')
      .then((doc) => {
        if (!alive) return
        const claimed = new Set(Object.values(doc.managed ?? {})
          .filter((m) => m?.alive && m?.port).map((m) => m.port as string))
        setBoards((doc.serial_ports ?? []).map(p => ({
          peer_id: p.remembered?.peer_id ?? '',
          claimed: claimed.has(p.device),
          device: p.device,
          robot_name: p.remembered?.robot_name ?? null,
        })))
      })
      .catch(() => { if (alive) setBoards(null) })
    return () => { alive = false }
  }, [fleetEmpty, boards])

  const fleetHosts = useMemo(() => armHosts(list.map(q => ({
    peer_id: q.peer_id, joints: Object.keys(q.state?.joints ?? {}).length }))), [list])

  const pairInputs = useMemo(() => list.map(q => ({
    peer_id: q.peer_id, joints: Object.keys(q.state?.joints ?? {}).length,
    role: q.role ?? null, role_volts: q.role_volts ?? null, role_source: q.role_source ?? null,
    robot_type: q.presence?.robot_type ?? null })), [list])

  useEffect(() => { void pwa.keepAwake(anyRunning) }, [anyRunning])  // eslint-disable-line react-hooks/exhaustive-deps

  const [dark, setDark] = useState<string[]>([])
  useEffect(() => {
    let live = true
    void serverRoutePaths().then(paths => { if (live) setDark(darkRoutes(paths)) }).catch(() => {})
    return () => { live = false }
  }, [backendKey()])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    let live = true
    void getRecordApi().then(a => { if (live) setRecordMock(a.mock) }).catch(() => {})
    return () => { live = false }
  }, [])

  // Keyboard: Escape closes, "." (or Cmd/Ctrl+. even while typing) opens the stop sheet, "?"
  // opens help.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      const verdict = hotkeyVerdict({
        key: e.key, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey, shiftKey: e.shiftKey,
        targetTag: el?.tagName, editable: el?.isContentEditable, repeat: e.repeat,
      })
      if (!verdict) return
      if (verdict === 'close') { route(null); return }
      // The chord must not also reach the browser (Cmd+. is "stop loading" in
      // some builds) or insert anything into the field it was pressed in.
      if (e.metaKey || e.ctrlKey) e.preventDefault()
      route(verdict === 'estop' ? 'estop' : 'help')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const detailPeer = detail ? peers[detail] : undefined
  useEffect(() => {
    // The peer vanished from the mesh entirely (never came back after a
    // re-point); a stage showing a peer that no longer exists is a lie.
    if (detail && !peers[detail]) setDetail(null)
  }, [detail, peers])

  // Re-evaluated on a 1s tick ONLY while something is wrong, so the "frozen
  // (Ns old)" number keeps counting instead of freezing with the view it
  // describes. A healthy link does no work here.
  const [linkTick, setLinkTick] = useState(0)
  const link = linkHealth({
    conn, browserOnline: pwa.online, lastEventAt, everOpen,
    meshOnline: mesh.online,
    // list.length, NOT the non-stale count: what is RENDERED is what can mislead.
    peerCount: list.length, now: Date.now(),
    sessionExpired: sessionVerdict(authToken(), Date.now() / 1000).refusesUntilSignIn,
  })
  // The tick runs ALWAYS, and that is the whole point.
  useEffect(() => {
    const id = setInterval(() => setLinkTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [])
  void linkTick

  const [refused, setRefused] = useState<RefusedHandshakes | null>(null)
  const [health, setHealth] = useState<unknown>(undefined)
  useEffect(() => {
    if (conn !== 'open') return
    let alive = true
    const poll = () => {
      httpGet<{ refused_handshakes?: RefusedHandshakes }>('/api/health')
        .then(h => { if (alive) { setRefused(h?.refused_handshakes ?? null); setHealth(h) } })
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 60_000)
    return () => { alive = false; clearInterval(id) }
  }, [conn])
  const notice = serverNotice(refused)
  // Staleness alone is the normal state of a long-running server and is NOT news; a field the
  // whole fleet is silent about is not news either.
  const stale = useMemo(() => staleServerNotice(health, fleetFieldGaps(peers)), [health, peers])

  return (
    <div className="stage">
      {/* FIRST IN THE DOM, therefore the FIRST TAB STOP on every screen (JOURNEYS #12: measured 14 to 30+ tab stops to reach it, and on the training screen it was unreachable inside 30). */}
      <EstopButton onClick={() => route('estop')} posture={estopPosture(link)} />

      <FleetBar
        conn={conn}
        peerCount={list.filter(p => !p.stale).length}
        dashboardId={dashboardId}
        safetyFlash={safetyFlash}
        mesh={mesh}
        online={pwa.online}
        installable={pwa.installable}
        activityCount={activity.length}
        absentChildren={absentChildren}
        quietChildren={quietChildren}
        recordMock={recordMock}
        onInstall={() => void pwa.install()}
        onSettings={() => { setSettingsTab(undefined); route('settings') }}
        onWireSecurity={() => { setSettingsTab('mesh'); route('settings') }}
        onActivity={() => route('activity')}
        onDevices={() => route('devices')}
        onTraining={() => { setTrainPrefill(undefined); route('training') }}
        onRecord={() => route('record')}
        onSim={() => route('sim')}
        onHelp={() => route('help')}
      />

      {pwa.needRefresh && (
        <div className="toast">
          A new dashboard version is ready
          {/* Say how long they have been on the old one: a phone left open beside the arms used to sit on an 11-hour-old bundle without ever being asked. */}
          {pwa.bundleAge() ? ` — this tab loaded ${pwa.bundleAge()}` : ''}.
          <button className="btn go" onClick={pwa.update}>reload</button>
          {/* what reloading costs RIGHT NOW. */}
          <span className="hint">{reloadImpact(Object.keys(busyPeers).filter(id => busyPeers[id])).text}</span>
        </div>
      )}

      {/* One judgment for all the ways this page can stop being attached to the fleet (lib/linkHealth): this device's network, a refused token, a dead API, a mute socket. */}
      {link.headline && (
        <div className={`toast ${link.commandsWork ? '' : 'warn'}`} role="status">
          <b>{link.headline}</b> {link.detail}
        </div>
      )}

      {/* the dark-feature banner. */}
      {dark.length > 0 && (
        <div className="toast warn" role="status">
          <b>Older server.</b> {darkFeatureMessage(dark)}
          <details>
            <summary className="muted small">which {dark.length === 1 ? 'one' : 'ones'}</summary>
            <ul className="muted small">{dark.map(p => <li key={p}>{p}</li>)}</ul>
          </details>
        </div>
      )}

      {/* somebody ELSE is being refused in a loop. */}
      {notice.text && !link.headline && (
        <div className="toast warn" role="status">
          <b>A client is being refused repeatedly</b> {notice.text}
        </div>
      )}

      {/* The server is older than the bundle it is serving, AND that is costing the operator something visible. */}
      {stale.text && !notice.text && !link.headline && (
        <div className="toast" role="status">
          <b>This server is older than this page</b> {stale.text}
        </div>
      )}

      {conn === 'unauthorized' ? (
        <div className="empty-fleet">
          <div className="empty-icon">🔒</div>
          <h2>This dashboard requires a token</h2>
          <p>
            <code>{backendLabel()}</code> refused the connection. Paste the token the server was
            started with:
          </p>
          <TokenPrompt />
          <p className="hint">
            It is the value of <code>--auth-token</code> / <code>DASHBOARD_AUTH_TOKEN</code>.
          </p>
        </div>
      ) : list.length === 0 ? (
        <div className="empty-fleet">
          <div className="empty-icon">{conn === 'open' ? '📡' : '🔌'}</div>
          {conn !== 'open' ? (
            <>
              <h2>Not connected to {backendLabel()}</h2>
              <p>
                {conn === 'connecting' ? 'Opening the mesh socket…' : 'The dashboard API is unreachable.'}
              </p>
              <p className="hint">
                If the API runs elsewhere, point this browser at it in Settings → Connection.
              </p>
              <button className="btn ghost" onClick={() => route('settings')}>open settings</button>
            </>
          ) : mesh.online === false ? (
            <>
              <h2>The dashboard's mesh session is down</h2>
              <p>The API is up, but it is not on the robot mesh — so no peer can be seen or commanded.</p>
              <p className="hint">Check the mesh endpoints, then restart the session.</p>
              <button className="btn ghost" onClick={() => route('settings')}>mesh settings</button>
            </>
          ) : (
            <>
              <h2>{loaded ? 'No robots on the mesh yet' : 'Loading the fleet…'}</h2>
              {/* the same sentence the record screen learned, from the same module — after a restart the arms are not unplugged, just not running, and this machine remembers them by USB serial. */}
              {loaded && homeRoute && (
                <p className="hint" role="status">{homeRoute}</p>
              )}
              <p>Start one anywhere on your network:</p>
              {/* this used to hardcode port="/dev/ttyACM0" — a Linux path, on a Mac whose arms live at /dev/cu.usbmodem*. */}
              {/* this <pre> was `white-space: pre` in a 340px column at phone width — measured scrollWidth 453 vs clientWidth 340, so the `mode="real", port=…` line, the whole point of the second line, sat outside the box behind a horizontal scroll nobody discovers. */}
              <pre className="startsnip">{snippet.code}</pre>
              <p className="row">
                <button className="btn ghost tiny" onClick={async () => {
                  try {
                    if (!navigator.clipboard) throw new Error('this page is not a secure origin, so the browser blocks copying')
                    await navigator.clipboard.writeText(snippet.code)
                    setSnipCopied('copied')
                  } catch (e: any) { setSnipCopied(e?.message ?? String(e)) }
                }}>copy</button>
                {snipCopied && <span className={snipCopied === 'copied' ? 'muted small' : 'warn small'} role="status">{snipCopied}</span>}
              </p>
              <p className="hint">{snippet.provenance}</p>
              <p className="hint">
                Set <code>STRANDS_MESH_LOCAL_DEV=1</code> + <code>STRANDS_MESH_MULTICAST=true</code> for local dev.
              </p>
              <button className="btn ghost" onClick={() => route('devices')}>spawn one here</button>
            </>
          )}
        </div>
      ) : (
        <main className="grid">
          {/* the fleet-wide lockout line. */}
          {/* local viewers should not stream camera frames out to the internet and back. */}
          <LanHint />
          {(() => {
            const lb = lockoutBanner(list)
            return lb ? (
              <div className={`lockout-banner ${lb.severity}`} role="status" style={{ gridColumn: '1 / -1' }}>
                <span aria-hidden="true">&#128721;</span><span>{lb.text}</span>
              </div>
            ) : null
          })()}
          {list.map(p => (
            <ErrorBoundary key={p.peer_id} label={`the card for ${p.peer_id}`}>
            <RobotCard
              key={p.peer_id}
              peer={p}
              twinLive={liveTwins.has(`${p.peer_id}-twin`)}
              hostsChildren={fleetHosts[p.peer_id]?.children ?? null}
              onOpen={setDetail}
              onBusyChange={(id, running) => setBusyPeers(s => (s[id] === running ? s : { ...s, [id]: running }))}
            />
            </ErrorBoundary>
          ))}
        </main>
      )}

      {detailPeer && (
        <ErrorBoundary label="the robot detail view" onDismiss={() => setDetail(null)}>
          <RobotDetail peer={detailPeer} twinLive={liveTwins.has(`${detailPeer.peer_id}-twin`)}
            hostsChildren={fleetHosts[detailPeer.peer_id]?.children ?? null} fleet={pairInputs}
                       onOpen={setDetail} onClose={() => setDetail(null)} />
        </ErrorBoundary>
      )}

      <ErrorBoundary label="settings" onDismiss={() => route(null)}>
        <SettingsDrawer open={panel === 'settings'} onClose={() => route(null)} mesh={mesh} initialTab={settingsTab} />
      </ErrorBoundary>
      <ErrorBoundary label="the activity log" onDismiss={() => route(null)}>
        <ActivityLog open={panel === 'activity'} onClose={() => route(null)} live={activity} />
      </ErrorBoundary>
      <ErrorBoundary label="the devices screen" onDismiss={() => route(null)}>
        <DevicePanel open={panel === 'devices'} onClose={() => route(null)} />
      </ErrorBoundary>
      <HelpSheet open={panel === 'help'} onClose={() => route(null)} />
      <EstopSheet open={panel === 'estop'} onClose={() => route(null)}
        linkWarning={link.commandsWork ? null : link.estopReason}
        meshBacked={mesh.online === true} />
      {panel === 'training' && (
        <ErrorBoundary label="the training screen" onDismiss={() => route(null)}>
          <TrainingTab onClose={() => route(null)} prefill={trainPrefill} />
        </ErrorBoundary>
      )}
      {panel === 'sim' && (
        <ErrorBoundary label="the simulation screen" onDismiss={() => route(null)}>
          <SimTab onClose={() => route(null)} />
        </ErrorBoundary>
      )}
      {panel === 'record' && (
        <ErrorBoundary label="the record screen" onDismiss={() => route(null)}>
          <RecordPanel peers={list.filter(p => !p.stale)} onClose={() => route(null)}
            onDevices={() => route('devices')}
            onTrain={prefill => { setTrainPrefill(prefill); route('training') }} />
        </ErrorBoundary>
      )}

      <ErrorBoundary label="the chat dock">
        <AgentDock
        onSettings={() => route('settings')}
        startOpen={new URLSearchParams(location.search).get('panel') === 'chat'}
        exampleRobot={list.find(p => !p.stale && p.presence?.robot_type === 'robot')?.peer_id}
        />
      </ErrorBoundary>
    </div>
  )
}

/** Minimal unlock form for the 1008 case - Settings itself is behind the token. */
function TokenPrompt() {
  const [token, setToken] = useState('')
  return (
    <form
      className="tokenprompt"
      onSubmit={e => { e.preventDefault(); setAuthToken(token) }}
    >
      <input type="password" value={token} placeholder="dashboard token" aria-label="dashboard token"
             onChange={e => setToken(e.target.value)} />
      <button className="btn go" type="submit" disabled={!token.trim()}>unlock</button>
    </form>
  )
}

export default function App() {
  // Remount everything when the backend or token changes: sockets, peer maps and frame buffers
  // all belong to one backend. subscribeAuth makes the key reactive — localStorage alone is not.
  const key = useSyncExternalStore(subscribeAuth, backendKey)
  return (
    <ConfigProvider key={key}>
      <AuthGate>
        <Dashboard />
      </AuthGate>
    </ConfigProvider>
  )
}
