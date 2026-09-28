/** The fleet stop lives in its own fixed layer, above every drawer, sheet, dock and toast. */
/**
 * `posture` (lib/linkHealth.estopPosture) marks the button when this page cannot currently
 * deliver the stop.
 */
import { ESTOP_KEYSHORTCUTS } from '../lib/hotkeys'

export default function EstopButton({
  onClick, posture,
}: { onClick: () => void; posture?: { degraded: boolean; title: string } }) {
  const degraded = !!posture?.degraded
  return (
    <div className="estop-layer">
      <button
        className={`estop${degraded ? ' unreachable' : ''}`}
        onClick={onClick}
        title={posture?.title ?? 'Stop every robot on the mesh - keyboard: . anywhere, or Cmd/Ctrl+. even while typing'}
        aria-label={degraded
          ? 'Emergency stop: the link is down, so this may not reach the robots'
          : 'Emergency stop: stop every robot on the mesh'}
        aria-keyshortcuts={ESTOP_KEYSHORTCUTS}
      >
        {degraded ? '🛑 STOP ALL ⚠' : '🛑 STOP ALL'}
      </button>
    </div>
  )
}
