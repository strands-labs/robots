import { useEffect, useMemo, useState } from 'react'
import { validationScope, type ValidatedInput } from '../lib/validationScope'
import { numField } from '../lib/numField'
import type { PolicyProvider } from '../types'
import { post, api as httpGet } from '../lib/endpoints'
import CheckpointPicker from './CheckpointPicker'
import { useConfig } from '../lib/useConfig'
import { peekDeployIntent, clearDeployIntent, type DeployIntent } from '../lib/deployIntent'
import { runRisk } from '../lib/runRisk'
import { fieldCopy, requirementSummary, missingSummary, localOnlySummary } from '../lib/policyCopy'
import { policyLabel, groupPolicies } from '../lib/policyLabels'
import RunConfirm from './RunConfirm'
import type { Presence } from '../types'

export interface RunBody {
  instruction: string
  policy_provider: string
  duration: number
  [key: string]: any
}

interface Props {
  peerId: string
  /** Judges whether ▶ moves metal - see lib/runRisk.ts. */
  presence: Presence | null | undefined
  running: boolean
  busy: boolean
  disabled?: boolean
  onRun: (body: RunBody) => void
  onStop: () => void
}

interface ValidateResult {
  /** did the preflight have a model to inspect at all (see validate_scope.py) */
  resolved?: boolean
  /** what the verdict does NOT cover, in plain words */
  scope_note?: string
  ok: boolean
  stage: string
  error?: string
  note?: string
  observation_keys?: string[]
  fit?: PolicyFit
}

export interface PolicyFit {
  ok: boolean
  /** true = this policy cannot drive this robot; no field correction changes that */
  blocking: boolean
  problems: { kind: string; detail: string }[]
  /** which axes were actually compared — so quiet reads as "verified", not "never looked" */
  checked: string[]
  /** false = nothing could be compared (unknown checkpoint, or a peer that announced nothing yet) */
  evidence?: boolean
}

/** The run form is *generated from the policy registry*, not hardcoded. */
export default function RunForm({ peerId, presence, running, busy, disabled, onRun, onStop }: Props) {
  const { policies } = useConfig()
  const [providerName, setProviderName] = useState('mock')
  const [instruction, setInstruction] = useState('')
  const [durationText, setDurationText] = useState('15')
  const wantedDuration = numField(durationText, { what: 'seconds', min: 1, max: 600, remedy: 'run it again to go longer' })
  const duration = wantedDuration.value
  const [advanced, setAdvanced] = useState(false)
  const [fields, setFields] = useState<Record<string, string>>({})
  const [validating, setValidating] = useState(false)
  const [validation, setValidation] = useState<ValidateResult | null>(null)
  // The INPUT a verdict was taken on: a "✓ resolves" that outlives the config it
  // vouched for is the most dangerous green tick in this form.
  const [validatedFor, setValidatedFor] = useState<ValidatedInput | null>(null)
  const [staged, setStaged] = useState<DeployIntent | null>(null)
  // A run body held back for confirmation: non-null means the sheet is up and
  // NOTHING has been sent yet.
  const [pending, setPending] = useState<RunBody | null>(null)

  // A deploy intent staged from the Training tab prefills THIS form - once, visibly, and only
  // into fields; running still takes the human pressing Run.
  useEffect(() => {
    const intent = peekDeployIntent()
    if (!intent) return
    const target = policies.find(p =>
      p.wire_safe && p.wire_fields?.some(f => f.key === 'pretrained_name_or_path' || f.key === 'model_path'))
    if (!target) return
    const pathKey = target.wire_fields.find(f => f.key === 'pretrained_name_or_path' || f.key === 'model_path')!.key
    setProviderName(target.name)
    setFields(prev => ({
      ...prev,
      [pathKey]: intent.checkpoint,
      ...(intent.policy_type && target.wire_fields.some(f => f.key === 'policy_type')
        ? { policy_type: intent.policy_type } : {}),
    }))
    setStaged(intent)
    // the prefilled field lives in the advanced section - open it, because a
    // prefill the user cannot SEE is not "review then Run", it is a surprise
    setAdvanced(true)
    clearDeployIntent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [policies])

  const provider: PolicyProvider | undefined = useMemo(
    () => policies.find(p => p.name === providerName),
    [policies, providerName],
  )
  const wireFields = provider?.wire_fields ?? []

  const value = (key: string, fallback: any) => {
    const raw = fields[key]
    if (raw !== undefined) return raw
    return fallback === null || fallback === undefined ? '' : String(fallback)
  }

  const [fit, setFit] = useState<PolicyFit | null>(null)
  const checkpointField = wireFields.find(
    f => f.key === 'pretrained_name_or_path' || f.key === 'model_path')?.key
  const checkpoint = checkpointField ? String(value(checkpointField, '')).trim() : ''
  /**
   * No norm_tag here on purpose: the registry lists it in `unsettable_over_mesh`, so the wire
   * schema drops it and this form can never own that field.
   */
  useEffect(() => {
    if (!checkpoint || !peerId) { setFit(null); return }
    let alive = true
    const t = setTimeout(() => {
      void httpGet<PolicyFit>(
        `/api/robots/${encodeURIComponent(peerId)}/policy-fit?repo_id=${encodeURIComponent(checkpoint)}`)
        .then(v => { if (alive) setFit(v) })
        // A failed lookup is not evidence of a mismatch: stay silent rather than cry wolf.
        .catch(() => { if (alive) setFit(null) })
    }, 400)
    return () => { alive = false; clearTimeout(t) }
  }, [checkpoint, peerId])
  // Refused, and deliberately not forceable: no tick makes a 2-value action drive 6 joints.
  const fitBlocked = !!fit?.blocking

  const missing = wireFields
    .filter(f => f.required && !String(value(f.key, f.default)).trim())
    .map(f => f.key)

  const parseField = (type: string, raw: string): any => {
    if (type === 'int') return Number.parseInt(raw, 10)
    if (type === 'float') return Number.parseFloat(raw)
    if (type === 'bool') return raw === 'true' || raw === '1'
    if (type === 'json') return JSON.parse(raw)
    return raw
  }

  /** Registry-keyed config for local validation. */
  const buildConfig = (): Record<string, any> => {
    const out: Record<string, any> = {}
    for (const f of wireFields) {
      const raw = String(value(f.key, f.default)).trim()
      if (!raw) continue
      try { out[f.key] = parseField(f.type, raw) } catch { out[f.key] = raw }
    }
    return out
  }

  const submit = () => {
    if (!instruction.trim() || missing.length || wantedDuration.problem) return
    const body: RunBody = {
      instruction: instruction.trim(),
      policy_provider: providerName,
      duration,
    }
    for (const f of wireFields) {
      const raw = String(value(f.key, f.default)).trim()
      if (!raw) continue
      try {
        body[f.wire_key] = parseField(f.type, raw)
      } catch {
        setValidation({ ok: false, stage: 'form', error: `${f.key}: not valid JSON` })
        return
      }
    }
    setValidation(null)
    // A real arm gets a confirmation naming itself first; sim runs stay a
    // single click, because there is nothing to be careful about.
    if (runRisk(presence).physical) {
      setPending(body)
      return
    }
    onRun(body)
  }

  const validate = async () => {
    setValidating(true)
    try {
      const asked: ValidatedInput = { provider: providerName, config: buildConfig() }
      setValidation(await post<ValidateResult>('/api/policies/validate', {
        policy_provider: providerName,
        policy_config: asked.config,
        peer_id: peerId,
      }))
      setValidatedFor(asked)
    } catch (e: any) {
      setValidation({ ok: false, stage: 'request', error: e?.message ?? String(e) })
      setValidatedFor({ provider: providerName, config: buildConfig() })
    } finally {
      setValidating(false)
    }
  }

  const locked = provider && !provider.wire_safe
  const blocked = !!disabled || busy

  const modelKey = wireFields.find(
    f => f.key === 'pretrained_name_or_path' || f.key === 'model_path')?.key
  const modelValue = modelKey ? String(value(modelKey, '') || '').trim() : ''

  return (
    <div className="runform">
      {pending && (
        <RunConfirm
          peerId={peerId}
          risk={runRisk(presence)}
          instruction={pending.instruction}
          provider={providerName}
          model={modelValue || null}
          durationS={duration}
          onCancel={() => setPending(null)}
          onConfirm={() => { const body = pending; setPending(null); onRun(body) }}
        />
      )}
      {staged && (
        <div className="deploy-banner">
          <span>🚀 prefilled from {staged.source} — review below, then press Run. Nothing has started.</span>
          <button className="btn ghost" onClick={() => { setStaged(null); setFields({}) }}>discard</button>
        </div>
      )}
      <div className="controls">
        <select
          value={providerName}
          onChange={e => { setProviderName(e.target.value); setFields({}); setValidation(null); setValidatedFor(null) }}
          disabled={blocked}
          // The control that decides what drives a physical arm was unlabelled:
          // a screen reader announced only the current value.
          aria-label="Policy — what will drive this robot"
          title={provider ? `${policyLabel(provider.name)} (${provider.name}) — ${provider.description}` : 'Policy — what will drive this robot'}
        >
          {/* Only when the policy list has not loaded: name what mock IS — the bare word next to live telemetry reads as "this is fake". */}
          {policies.length === 0 && <option value="mock">{policyLabel('mock')}</option>}
          {/* UX_REVIEW #1: registry identifiers (cosmos3, wbc_gait, lerobot_async) are not names a person recognises, and the groups say what the choice COSTS — a checkpoint, a server that must be up, or nothing at all. lib/policyLabels.ts never invents a label: an unknown provider renders verbatim under "Other" rather than being dressed up as something it might not be. */}
          {groupPolicies(policies, p => p.name).map(g => (
            <optgroup key={g.group} label={g.group}>
              {g.items.map(p => (
                <option key={p.name} value={p.name} disabled={!p.wire_safe}>
                  {p.wire_safe ? '' : '🔒 '}{policyLabel(p.name)}
                  {/* JOURNEYS #13: the option line says what the operator must HAVE ("needs a checkpoint + policy family"), not the constructor kwarg names. */}
                  {p.requires.length ? ` — needs ${requirementSummary(p.requires)}` : ''}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <input
          /* a placeholder is not a label: it disappears the moment the operator types, and a
             screen reader on the robot detail screen announced this one as an unnamed text box */
          aria-label="instruction for the policy"
          placeholder="pick up the red cube"
          value={instruction}
          onChange={e => setInstruction(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          disabled={blocked}
        />
        {running
          ? <button className="btn stop" onClick={onStop} disabled={busy} title="Stop this robot">■</button>
          : (
            <button
              className="btn go"
              onClick={submit}
              disabled={blocked || !instruction.trim() || missing.length > 0 || !!locked || !!wantedDuration.problem || fitBlocked}
              {...(fitBlocked ? { title: fit!.problems.map(p => p.detail).join(' — ') } : {})}
              title={locked
                ? `${providerName} is not in the mesh policy allowlist`
                : wantedDuration.problem ? `duration: ${wantedDuration.problem}`
                : missing.length ? `missing: ${missingSummary(missing)}` : 'Run'}
            >▶</button>
          )}
        <button
          className={advanced ? 'btn ghost on' : 'btn ghost'}
          onClick={() => setAdvanced(a => !a)}
          title="Provider options"
        >⚙</button>
      </div>

      {missing.length > 0 && !advanced && (
        <button className="needs" onClick={() => setAdvanced(true)}>
          {providerName} needs {missingSummary(missing)} → open options
        </button>
      )}

      {/* The duration box only exists inside options, so a bad value there would otherwise disable ▶ with its reason hidden in a tooltip — which a touch screen never shows. */}
      {wantedDuration.problem && !advanced && (
        <button className="needs" onClick={() => setAdvanced(true)}>
          duration: {wantedDuration.problem} → open options
        </button>
      )}

      {advanced && (
        <div className="advanced">
          <label className="field">
            <span>duration (s)</span>
            <input
              type="number" min={1} max={600} value={durationText}
              onChange={e => setDurationText(e.target.value)}
              disabled={blocked}
              aria-invalid={!!wantedDuration.problem} aria-describedby="run-duration-say"
            />
            <span id="run-duration-say" className={`fieldsay${wantedDuration.problem ? ' bad' : ''}`}>
              {wantedDuration.problem ?? wantedDuration.note ?? ''}
            </span>
          </label>

          {wireFields.map(f => (
            <label className={f.required && !String(value(f.key, f.default)).trim() ? 'field missing' : 'field'} key={f.key}>
              {/* Label in words, identifier kept beside it: operators paste `f.key` into their own scripts, so replacing it would trade one comprehension bug for a copy-paste one. */}
              <span>
                {fieldCopy(f.key).label}{f.required && <b title="required"> *</b>}
                {fieldCopy(f.key).known && <code className="ident" title="the API field name">{f.key}</code>}
                {f.wire_key !== f.key && <em title={`sent as ${f.wire_key}`}> →{f.wire_key}</em>}
              </span>
              {f.type === 'bool' ? (
                <select value={value(f.key, f.default) || 'false'} onChange={e => setFields(s => ({ ...s, [f.key]: e.target.value }))} disabled={blocked}>
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              ) : f.key === 'pretrained_name_or_path' || f.key === 'model_path' ? (
                /**
                 * Checkpoint fields get a Hub+local-cache type-ahead: the registry names the field, this names
                 * the VALUES (thousands of public LeRobot checkpoints).
                 */
                <CheckpointPicker
                  value={value(f.key, f.default)}
                  disabled={blocked}
                  onPick={(repoId, policyType) => setFields(s => {
                    const next = { ...s, [f.key]: repoId }
                    if (policyType && wireFields.some(w => w.key === 'policy_type') && !s.policy_type) {
                      next.policy_type = policyType
                    }
                    return next
                  })}
                />
              ) : (
                <input
                  type={f.type === 'int' || f.type === 'float' ? 'number' : 'text'}
                  value={value(f.key, f.default)}
                  /* "string" as a placeholder tells the operator the TYPE, which
                     they can see, and not the value, which they cannot guess. */
                  placeholder={f.type === 'json' ? '{"…": …}' : (fieldCopy(f.key).hint ?? f.type)}
                  onChange={e => setFields(s => ({ ...s, [f.key]: e.target.value }))}
                  disabled={blocked}
                />
              )}
            </label>
          ))}

          <div className="advanced-actions">
            <button className="btn ghost" onClick={validate} disabled={validating}>
              {validating ? 'checking…' : '✓ validate'}
            </button>
            {provider?.server_based && (
              <span className="hint">needs a running inference server at that host/port</span>
            )}
          </div>

          {provider && provider.unsettable_over_mesh.length > 0 && (
            <div className="hint local-only">
              these options only work when the policy is built on the robot itself —
              the mesh command cannot carry them:{' '}
              <code>{localOnlySummary(provider.unsettable_over_mesh)}</code>
            </div>
          )}
          {locked && (
            <div className="hint warn">
              🔒 {providerName} is rejected by the mesh policy allowlist. Add it to{' '}
              <code>STRANDS_MESH_POLICY_TYPE_ALLOW</code> on every peer.
            </div>
          )}
        </div>
      )}

      {fit && (fit.blocking || fit.evidence) && (
        <div className={fit.blocking ? 'validation bad' : 'validation ok'} role={fit.blocking ? 'alert' : undefined}>
          {fit.blocking
            ? <>
                <div>✗ this policy does not fit this robot</div>
                {fit.problems.map(p => <div className="hint" key={p.kind}>{p.detail}</div>)}
                <div className="hint">
                  Nothing on this form fixes that — pick a checkpoint trained on this robot, or run it in sim.
                </div>
              </>
            : <>✓ checkpoint matches this robot ({fit.checked.join(', ')})</>}
        </div>
      )}
      {validation && (() => {
        // A verdict describes the input it was taken on. Once a field moves it
        // stops describing the form, and saying so beats a stale green tick.
        const scope = validationScope(validatedFor, { provider: providerName, config: buildConfig() })
        const cls = !scope.applies ? 'validation stale' : validation.ok ? 'validation ok' : 'validation bad'
        return (
          <div className={cls}>
            {!scope.applies
              ? `${validation.ok ? '✓' : '✗'} (outdated) ${validation.ok ? `${validatedFor?.provider ?? providerName} resolved` : `${validation.stage}: ${validation.error}`}`
              : validation.ok
                // resolved === false means the preflight found no model to inspect:
                // "no objection could be raised" is not "I checked what you are
                // about to run on a real arm".
                ? validation.resolved === false
                  ? `— nothing to resolve for ${providerName}`
                  : `✓ ${providerName} resolves`
                : `✗ ${validation.stage}: ${validation.error}`}
            {!scope.applies && <div className="hint warn">{scope.note}</div>}
            {validation.scope_note && <div className="hint">{validation.scope_note}</div>}
            {validation.note && <div className="hint">{validation.note}</div>}
          </div>
        )
      })()}
    </div>
  )
}
