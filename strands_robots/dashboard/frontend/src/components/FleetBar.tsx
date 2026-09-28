import type { MeshInfo } from '../types'
import type { ConnState } from '../lib/useMesh'
import { backendLabel } from '../lib/endpoints'
import { connBadge } from '../lib/connBadge'
import { recordNavFlag } from '../lib/rehearsalNav'
import { absentNotice, quietNotice, type AbsentChild } from '../lib/absentChildren'
import StrandsMark from './StrandsMark'

interface Props {
  conn: ConnState
  peerCount: number
  dashboardId: string
  safetyFlash: string | null
  mesh: MeshInfo
  online: boolean
  installable: boolean
  activityCount: number
  /** true = the record backend is a rehearsal, null = not probed yet */
  recordMock: boolean | null
  onInstall: () => void
  onSettings: () => void
  onWireSecurity: () => void
  onActivity: () => void
  absentChildren?: readonly AbsentChild[]
  quietChildren?: readonly string[]
  onDevices: () => void
  onTraining: () => void
  onRecord: () => void
  onSim: () => void
  onHelp: () => void
}

export default function FleetBar({
  conn, peerCount, dashboardId, safetyFlash, mesh, online, installable,
  activityCount, recordMock, absentChildren, quietChildren, onInstall, onSettings, onWireSecurity, onActivity, onDevices, onTraining, onRecord, onSim,
  onHelp,
}: Props) {
  // The mesh session and this browser's socket fail independently: the page can be LIVE while
  // the robot mesh is down, and vice versa.
  const absentDeath = absentNotice(absentChildren)
  const quiet = quietNotice(quietChildren, absentChildren)
  const meshDown = mesh.online === false
  const badge = connBadge(conn, { meshDown })
  // UX_REVIEW #10: a feature that cannot write a dataset says so in the nav.
  const rec = recordNavFlag(recordMock)

  return (
    <header className="fleetbar">
      <div className="brand">
        <span className="logo"><StrandsMark size={26} title="Strands Agents" /></span>
        <div>
          <h1>strands robots</h1>
          <div className="sub" title={`API: ${backendLabel()}`}>
            {dashboardId || 'fleet cockpit'}
            <span className="backend"> · {backendLabel()}</span>
          </div>
        </div>
      </div>

      <div className="fleet-right">
        {safetyFlash && (
          <span className={`safety ${safetyFlash}`}>
            {safetyFlash === 'estop' ? '🛑 E-STOP' : '✅ RESUMED'}
          </span>
        )}
        {!online && <span className="badge warn" title="this device has no network">offline</span>}
        {mesh.local_dev && (
          <button
            className="badge warnchip"
            onClick={onWireSecurity}
            title="Robot mesh traffic is not encrypted. Fine on a trusted LAN - click for details and how to enable wire security."
          >
            mesh unencrypted · local only
          </button>
        )}
        {meshDown && <span className="badge danger" title="the dashboard's own mesh session is closed">mesh down</span>}

        {installable && (
          <button className="chip" onClick={onInstall} title="Install as an app">⤓ install</button>
        )}
        <button className="chip" onClick={onDevices} title="Local hardware and managed robots">⚙ devices</button>
        {/* U22: a robot the operator started died and the fleet only got shorter. */}
        {quiet && (
          <button
            className="chip warn"
            onClick={onDevices}
            title={`${quiet.detail}\n\nOpen devices for its log — the refusal that kept it out of the fleet is in there (a missing calibration, a busy servo bus), and despawn is there too.`}
          >🫥 {quiet.headline}</button>
        )}
        {absentDeath && (
          <button
            className="chip warn"
            onClick={onDevices}
            title={`${absentDeath.detail}\n\nOpen devices for the exit status and the last output.`}
          >⚰ {absentDeath.headline}</button>
        )}
        <button
          className={`chip${rec.cls ? ` ${rec.cls}` : ''}`}
          onClick={onRecord}
          title={rec.title}
          aria-label={rec.aria}
        >⏺ record{rec.suffix}</button>
        <button className="chip" onClick={onTraining} title="Train policies on recorded datasets">🎓 train</button>
        <button className="chip" onClick={onSim} title="Simulated robots in this process: twin, camera, joint targets">🧊 sim</button>
        <button className="chip" onClick={onActivity} title="Command history">
          ☰ activity{activityCount > 0 ? ` (${activityCount})` : ''}
        </button>
        <button className="chip" onClick={onSettings} title="Settings">⚒ settings</button>
        {/* JOURNEYS #7: the page had 0 links and 0 onboarding words. */}
        <button
          className="chip"
          onClick={onHelp}
          title="What this page is, how to stop a robot, and where the docs are"
          aria-keyshortcuts="?"
        >? help</button>

        <span className="peers">{peerCount} peer{peerCount === 1 ? '' : 's'}</span>
        <span
          className={`conn ${conn}${badge.tone ? ` ${badge.tone}` : ''}`}
          title={badge.title}
          aria-label={badge.aria}
        >
          {badge.label}
        </span>
      </div>
    </header>
  )
}
