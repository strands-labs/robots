import { useCallback, useEffect, useState } from 'react'
import { api, post } from '../lib/endpoints'
import { nothingGranted } from '../lib/consent'

type TeleopEnvelope = {
  granted: boolean
  value_abs: string | null
  slew_abs: string | null
  is_degree_preset: boolean
}

type ConsentState = {
  trust_remote_code: boolean
  hf_repo_allow: string[]
  policy_type_allow?: string[]
  policy_host_allow?: string[]
  /* Absent from an older server: this screen listed two of the three kinds and the teleop envelope
     widening — the grant with physical reach — could not be seen or revoked here. */
  teleop_degree_units?: TeleopEnvelope
  agent_physical_motion?: boolean
  locks?: { task_requires_confirm: boolean; task_requires_confirm_env: string }
  env_file?: string
}

export default function ConsentSettings() {
  const [state, setState] = useState<ConsentState | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setState(await api<ConsentState>('/api/consent'))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { void load() }, [load])

  /**
   * The lock is a plain env var the dashboard reads per request, so flipping it goes through the
   * existing config rail rather than the consent endpoints — /api/consent grants and revokes
   * PERMISSIONS, and pushing a restriction through it would make "revoke" mean two opposite
   * things.
   */
  const setLock = async (on: boolean) => {
    const key = state?.locks?.task_requires_confirm_env ?? 'STRANDS_DASH_TASK_REQUIRES_CONFIRM'
    setBusy('lock'); setNote(null); setError(null)
    try {
      await post('/api/config', { env: { [key]: on ? '1' : '' } })
      /* Cleared, not deleted: an absent line lets a stale value from a shell profile or a launchd
         plist win the next restart — a change that silently does not hold. */
      setNote(on
        ? 'on — a task that would move a real robot now needs the ▶ confirmation'
        : 'off — any caller with the API token can start a real task again')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const revoke = async (kind: string, subject: string | null, label: string) => {
    setBusy(label); setNote(null); setError(null)
    try {
      const r = await post<{ revoked: boolean; note?: string }>('/api/consent/revoke', { kind, subject })
      setNote(r.note ?? (r.revoked ? `revoked ${label}` : `nothing to revoke for ${label}`))
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  if (error && !state) return <p className="hint">could not read permissions: {error}</p>

  const envelope = state?.teleop_degree_units
  const nothing = state && nothingGranted(state as unknown as Record<string, unknown>)

  return (
    <div className="consent-settings">
      <h3>Permissions you granted</h3>
      {nothing ? (
        <p className="hint">
          Nothing extra is allowed here. When a safety guard refuses something, the dashboard asks
          you first and what you approve shows up in this list.
        </p>
      ) : null}

      {state?.trust_remote_code ? (
        <div className="cg-row">
          <div>
            <b>Run model code from HuggingFace</b>
            <div className="hint">
              Any policy load may execute code from the model repository — this is the widest
              permission on the list.
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === 'trust'}
                  onClick={() => revoke('trust_remote_code', null, 'trust')}>
            {busy === 'trust' ? '…' : 'revoke'}
          </button>
        </div>
      ) : null}

      {state?.agent_physical_motion ? (
        <div className="cg-row">
          <div>
            <b>The agent may start motion on real robots</b>
            <div className="hint">
              A chat sentence or a voice command can put any real robot on this mesh in motion, with no
              confirmation and without the check that the policy fits that robot — the one the ▶ button
              does. Revoking leaves the agent able to stop robots, answer questions and run tasks in
              simulation; starting a real arm comes back to you.
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === 'agent motion'}
                  onClick={() => revoke('agent_physical_motion', null, 'agent motion')}>
            {busy === 'agent motion' ? '…' : 'revoke'}
          </button>
        </div>
      ) : null}

      {/* Not a grant — the only row here that makes this machine stricter, which is why it is shown in BOTH states: an operator cannot choose a lock they have never been told exists, and the ▶ button already sends the confirmation, so turning it on costs them nothing. */}
      {state?.locks ? (
        <div className="cg-row">
          <div>
            <b>Require the ▶ confirmation before real motion</b>
            <div className="hint">
              {state.locks.task_requires_confirm
                ? 'On. A task that would move a real robot is refused unless it comes from the ▶ button (or a script that says so explicitly). Simulated robots and stopping are never affected.'
                : 'Off. Anything holding this dashboard\u2019s API token — a script, a terminal, whoever finds the token if this dashboard is reachable from the internet — can start a real robot with one request and no confirmation. Turning this on does not change the ▶ button.'}
            </div>
          </div>
          <button className={`btn ghost${state.locks.task_requires_confirm ? '' : ' danger'}`}
                  disabled={busy === 'lock'}
                  onClick={() => void setLock(!state.locks!.task_requires_confirm)}>
            {busy === 'lock' ? '…' : state.locks.task_requires_confirm ? 'turn off' : 'turn on'}
          </button>
        </div>
      ) : null}

      {envelope?.granted ? (
        <div className="cg-row">
          <div>
            <b>Teleop envelope widened{envelope.is_degree_preset ? ' to degrees' : ''}</b>
            <div className="hint">
              Every teleop stream on this machine, not one arm: a single frame may command a reach
              of {envelope.value_abs ?? 'the default'} units
              {envelope.slew_abs ? ` at up to ${envelope.slew_abs} units/s` : ' (speed bound left at the default)'}.
              {envelope.is_degree_preset
                ? ' This is the degrees preset, which an SO-101 needs — a runaway far outside it is still refused.'
                : ' This is a hand-set bound, not the degrees preset — check it is the one you meant.'}
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === 'teleop'}
                  onClick={() => revoke('teleop_degree_units', null, 'the teleop envelope')}>
            {busy === 'teleop' ? '…' : 'revoke'}
          </button>
        </div>
      ) : null}

      {(state?.policy_type_allow ?? []).map(entry => (
        <div className="cg-row" key={`type-${entry}`}>
          <div>
            <b className="rc-mono">{entry}</b>
            <div className="hint">this policy may be built and run — one name, no wildcard</div>
          </div>
          <button className="btn ghost danger" disabled={busy === entry}
                  onClick={() => revoke('policy_type_allow', entry, entry)}>
            {busy === entry ? '…' : 'revoke'}
          </button>
        </div>
      ))}

      {(state?.policy_host_allow ?? []).map(entry => (
        <div className="cg-row" key={`host-${entry}`}>
          <div>
            <b className="rc-mono">{entry}</b>
            <div className="hint">
              {entry.includes('/')
                ? 'every address in this range may run policies for your robots — wider than one host'
                : 'policies may run on this host: it receives camera frames and joint states, and '
                  + 'what it returns drives the arms'}
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === entry}
                  onClick={() => revoke('policy_host_allow', entry, entry)}>
            {busy === entry ? '…' : 'revoke'}
          </button>
        </div>
      ))}

      {state?.hf_repo_allow.map(entry => (
        <div className="cg-row" key={entry}>
          <div>
            <b className="rc-mono">{entry}</b>
            <div className="hint">
              {entry.includes('/')
                ? 'this model only'
                : 'every model under this organisation — wider than a single approval'}
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === entry}
                  onClick={() => revoke('hf_repo_allow', entry, entry)}>
            {busy === entry ? '…' : 'revoke'}
          </button>
        </div>
      ))}

      {/* the answers to pressing "revoke". */}
      {note ? <p className="cs-note" role="status">{note}</p> : null}
      {error ? <p className="cs-error" role="alert">{error}</p> : null}
      <p className="hint">
        Revoking applies to robots started from now on: a peer that is already running keeps the
        permission it was started with until you respawn it.
      </p>
    </div>
  )
}
