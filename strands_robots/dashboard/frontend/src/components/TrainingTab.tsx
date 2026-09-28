import { useEffect, useRef, useState } from 'react'
import { useDialogFocus } from '../lib/useDialogFocus'
import { numField } from '../lib/numField'
import { trainingFreshness } from '../lib/trainingFreshness'
import { api, post, HttpError } from '../lib/endpoints'
import { extraFields, missingForProvider } from '../lib/providerFields'
import { suggestOutputDir } from '../lib/outputDirSuggest'
import { holdout } from '../lib/holdout'
import { labelsGate, labelSummary, labelRowLine, type LabelView } from '../lib/episodeLabels'
import { fieldSupport } from '../lib/serverFields'
import { sideEffectVerdict, type SideEffectKind } from '../lib/submitOutcome'
import LossSpark from './LossSpark'
import { pushLoss, fmtStep, type LossPoint } from '../lib/lossTrace'
import { jobTransitions, type JobStateMap } from '../lib/jobAnnounce'
import { setDeployIntent } from '../lib/deployIntent'
import { datasetKey as dsKey, selectDataset, selectionKey, replayable, trainable, selectedRow, datasetMark, episodeChoice, type DatasetRow } from '../lib/datasetSelection'
import { datasetHint, isCurrentResponse } from '../lib/datasetHint'
import { jobsLedgerNotice } from '../lib/jobsLedger'
import { orderJobsNewestFirst } from '../lib/orderJobs'
import { newerThanApplied } from '../lib/requestOrder'
import { outputDirSay, trainGate, type OutputDirVerdict } from '../lib/outputDirIntent'

// One row is either LOCAL (has a `root` path, trains offline) or from the HUB (no root, trains
// from repo_id after a download). lib/datasetSelection owns that distinction and the rule that
// exactly one field reaches the trainer.
type Dataset = DatasetRow
interface Job { job_id: string; provider: string; dataset?: string; base_model?: string; output_dir?: string; steps?: number; submitted_at?: number }
interface JobStatus { status: string; data: { status?: string; metrics?: Record<string, unknown> }; text: string }
interface ArtifactVerdict { ok?: boolean; reason?: string; message?: string; warning?: string; path?: string }

/**
 * Training tab - submit / monitor / export policy training jobs. Backed by train_policy, the
 * one workflow tool with structured JSON results (job_id, status, metrics) - no prose parsing.
 */
// Same family-name heuristic the backend's checkpoint search uses - only a
// PREFILL for the run form's policy_type field, never a decision.
function guessPolicyType(baseModel: string | undefined): string | null {
  const m = (baseModel ?? '').replace(/_/g, '-').match(/\b(smolvla|act|diffusion|pi0-fast|pi05|pi0|tdmpc|vqbet)\b/i)
  return m ? m[1].toLowerCase().replace('-', '_').replace('pi0fast', 'pi0_fast') : null
}

export default function TrainingTab({ onClose, prefill }: {
  onClose: () => void
  /** seed from the record screen's close receipt — the dataset the operator JUST made */
  prefill?: { dataset_root?: string }
}) {
  const [trainers, setTrainers] = useState<string[]>([])
  const [unsupported, setUnsupported] = useState<Record<string, string>>({})
  const [srvFields, setSrvFields] = useState<string[] | null>(null)
  const [srvHeard, setSrvHeard] = useState(false)
  const sheetRef = useRef<HTMLDivElement | null>(null)
  useDialogFocus(sheetRef)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [statuses, setStatuses] = useState<Record<string, JobStatus>>({})
  const [jobSay, setJobSay] = useState('')
  const seenStates = useRef<JobStateMap>({})
  const [polledAt, setPolledAt] = useState<Record<string, number>>({})
  const [pollFail, setPollFail] = useState<Record<string, { n: number; msg: string }>>({})
  const [nowS, setNowS] = useState(() => Date.now() / 1000)
  const [traces, setTraces] = useState<Record<string, LossPoint[]>>({})
  const [form, setForm] = useState({ provider: 'lerobot_local', dataset_root: prefill?.dataset_root ?? '', dataset_repo_id: '', base_model: 'lerobot/smolvla_base', output_dir: '', steps: '10000', method: 'lora', embodiment: '', val_episodes: '' })
  // R6: the picker searches the Hub as you type.
  const [dsQuery, setDsQuery] = useState('')
  const [dsProblem, setDsProblem] = useState<string | null>(null)
  const [dsShownQuery, setDsShownQuery] = useState<string | null>(null)
  const dsSeq = useRef(0)
  // The status poll's own ordering: `tick` counts polling rounds, `applied` is the round whose
  // answer is on screen for each job.
  const tick = useRef(0)
  const tickBusy = useRef(false)
  const applied = useRef<Record<string, number>>({})
  const [dsAuth, setDsAuth] = useState<{ authenticated?: boolean; user?: string | null; detail?: string } | null>(null)
  // The ledger is a file that can go unreadable; an empty list alone cannot say so.
  const [jobsProblem, setJobsProblem] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  // Which episode each row's replay should ask for. Keyed per dataset, because the operator
  // comparing two recordings must not have one box silently follow them between rows.
  const [episodeBox, setEpisodeBox] = useState<Record<string, string>>({})

  const refresh = async () => {
    const seq = ++dsSeq.current
    try {
      const [t, d, j] = await Promise.all([
        api('/api/training/trainers'),
        api(`/api/training/datasets?q=${encodeURIComponent(dsQuery)}`),
        api('/api/training/jobs'),
      ])
      setTrainers(t.trainers ?? [])
      setUnsupported(t.unsupported ?? {})
      setSrvFields(Array.isArray(t.fields) ? t.fields : null)
      setSrvHeard(true)
      setJobs(orderJobsNewestFirst(j.jobs ?? []))
      setJobsProblem(j.problem ?? null)
      // A keystroke may have fired a newer dataset search while this was in
      // flight; the newer question owns the list.
      if (isCurrentResponse(seq, dsSeq.current)) {
        setDatasets(d.datasets ?? [])
        setDsProblem(d.problem ?? null)
        setDsAuth(d.hf_auth ?? null)
        setDsShownQuery(dsQuery)
      }
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
  }
  useEffect(() => { refresh() }, [])

  const outSeq = useRef(0)
  useEffect(() => {
    const path = form.output_dir.trim()
    setClearArmedFor(null)  // editing the field revokes a tick made for the old path
    if (!path) { setOutDir(null); return }
    const seq = ++outSeq.current
    const t = setTimeout(async () => {
      try {
        const v = await api(`/api/training/output-dir?path=${encodeURIComponent(path)}`)
        if (!isCurrentResponse(seq, outSeq.current)) return
        setOutDir(v)
      } catch {
        if (!isCurrentResponse(seq, outSeq.current)) return
        setOutDir(null)
      }
    }, 400)
    return () => clearTimeout(t)
  }, [form.output_dir])

  // Type-ahead: datasets only (re-polling jobs on every keystroke would be rude to a running
  // job's status endpoint). 250ms because each miss is a Hub round trip; the backend caches a
  // hit for 5 minutes, never a failure.
  useEffect(() => {
    const t = setTimeout(async () => {
      const seq = ++dsSeq.current
      const asked = dsQuery
      try {
        const d = await api(`/api/training/datasets?q=${encodeURIComponent(asked)}`)
        // Out-of-order Hub answers: a slow reply for "so" must not overwrite the
        // rows for "so101" that the user is about to pick from.
        if (!isCurrentResponse(seq, dsSeq.current)) return
        setDatasets(d.datasets ?? [])
        setDsProblem(d.problem ?? null)
        setDsAuth(d.hf_auth ?? null)
        setDsShownQuery(asked)
      } catch (e) {
        if (!isCurrentResponse(seq, dsSeq.current)) return
        setDsProblem(`search failed: ${(e as any)?.message ?? e}`)
        setDsShownQuery(asked)
      }
    }, 250)
    return () => clearTimeout(t)
  }, [dsQuery])

  useEffect(() => {
    const now: JobStateMap = {}
    for (const [id, st] of Object.entries(statuses)) {
      const state = st?.data?.status
      if (typeof state === 'string' && state) now[id] = state
    }
    const said = jobTransitions(seenStates.current, now)
    // Merged, not replaced: a job that drops out of the poll must not be re-announced if
    // it comes back, and a finished job's state is what proves it was already told.
    seenStates.current = { ...seenStates.current, ...now }
    if (said) setJobSay(said)
  }, [statuses])

  // poll running job statuses every 5s
  useEffect(() => {
    const id = setInterval(async () => {
      if (tickBusy.current) return
      tickBusy.current = true
      const round = ++tick.current
      try {
      // the five NEWEST (orderJobsNewestFirst put them there) - a finished run's
      // status cannot change, and the poll budget belongs to the live ones
      for (const job of jobs.slice(0, 5)) {
        if (!job.job_id) continue
        try {
          const s = await api(`/api/training/status?provider=${job.provider}&job_id=${encodeURIComponent(job.job_id)}`)
          // Only a newer round may speak for this job.
          if (!newerThanApplied(round, applied.current[job.job_id])) continue
          applied.current[job.job_id] = round
          setStatuses(prev => ({ ...prev, [job.job_id]: s }))
          setPolledAt(prev => ({ ...prev, [job.job_id]: Date.now() / 1000 }))
          setPollFail(prev => (prev[job.job_id] ? { ...prev, [job.job_id]: { n: 0, msg: '' } } : prev))
          const m = s?.data?.metrics as Record<string, unknown> | undefined
          if (m) setTraces(prev => ({
            ...prev,
            [job.job_id]: pushLoss(prev[job.job_id] ?? [], m.latest_step, m.latest_loss),
          }))
        } catch (e) {
          // ONE failed poll is transient; an unbounded swallow is how a dead run
          // keeps rendering "running" at 4.7k/10k steps forever. Count them, and
          // let the card say the numbers went old.
          const msg = String((e as any)?.message ?? e).slice(0, 120)
          setPollFail(prev => ({ ...prev, [job.job_id]: { n: (prev[job.job_id]?.n ?? 0) + 1, msg } }))
        }
      }
      } finally { tickBusy.current = false }
    }, 5000)
    return () => clearInterval(id)
  }, [jobs])

  // The age must advance while NOTHING arrives, so it cannot be driven by the
  // poll that stopped.
  useEffect(() => {
    const id = setInterval(() => setNowS(Date.now() / 1000), 5000)
    return () => clearInterval(id)
  }, [])

  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }))

  /**
   * A failed request that STARTS something is not the same as one that did not happen: a
   * rejected fetch covers "never left this machine" and "ran, then lost the answer".
   */
  const failed = (kind: SideEffectKind, e: unknown) => {
    const v = sideEffectVerdict({
      kind,
      status: e instanceof HttpError ? e.status : 0,
      message: (e as any)?.message ?? String(e),
    })
    setMsg(v.text)
    if (v.delivered === 'unknown') refresh()
  }

  const submit = async (validateOnly: boolean) => {
    const picked = selectedRow(datasets, form)
    const can = trainable(picked)
    if (!can.ok) {
      if (dsOverride !== selectionKey(form)) {
        setDsWarn({ key: selectionKey(form), reason: can.reason, recording: picked?.recording === true })
        setMsg(null)
        return
      }
      // CONSUMED BY THE RUN IT AUTHORISED.
      setDsOverride(null)
    }
    if (!validateOnly && !gate.ok) { setMsg(`✗ ${gate.why}`); return }
    setBusy(true); setMsg(null)
    const body = {
      provider: form.provider,
      dataset_root: form.dataset_root || undefined,
      dataset_repo_id: form.dataset_repo_id || undefined,
      base_model: form.base_model || undefined,
      output_dir: form.output_dir || undefined,
      ...(gate.confirmClear ? { confirm_clear: true } : {}),
      steps: wantedSteps.value,
      method: form.method || undefined,
      ...(extraFields(form.provider).some(f => f.key === 'embodiment') && form.embodiment.trim()
        ? { embodiment: form.embodiment.trim() } : {}),
      // Held out only when the operator asked for it: `null` and an absent key both mean "train
      // on every episode", and sending 0 would show a split in the form that the backend drops.
      ...(wantedHoldout.send !== null && holdoutSupport.ok ? { val_episodes: wantedHoldout.send } : {}),
    }
    try {
      const j = await post(validateOnly ? '/api/training/validate' : '/api/training/submit', body)
      setMsg(j.status === 'success' ? `✓ ${j.text?.slice(0, 200)}` : `✗ ${j.text?.slice(0, 300)}`)
      if (!validateOnly && j.status === 'success') refresh()
    } catch (e) {
      // A validate is read-only: it cannot leave a run behind, so it must not
      // claim it might have.
      failed(validateOnly ? 'export' : 'training', e)
    }
    setBusy(false)
  }

  const [dsWarn, setDsWarn] = useState<{ key: string; reason: string; recording?: boolean } | null>(null)
  const [dsOverride, setDsOverride] = useState<string | null>(null)
  const [outDir, setOutDir] = useState<OutputDirVerdict | null>(null)
  const [clearArmedFor, setClearArmedFor] = useState<string | null>(null)

  const outSay = outputDirSay(outDir)
  const gate = trainGate({ path: form.output_dir, verdict: outDir, armedFor: clearArmedFor })

  const [collect, setCollect] = useState({ dataset_root: '', instruction: 'pick up the red cube', n_episodes: '5', duration: '10', robot_name: 'so101' })
  const [showCollect, setShowCollect] = useState(false)

  const STEP_RULES = { what: 'steps', min: 1, max: 2_000_000, remedy: 'submit a shorter run' }
  const wantedSteps = numField(form.steps, STEP_RULES)
  /** The validation holdout. */
  const [labelsFor, setLabelsFor] = useState<string | null>(null)
  const [labelData, setLabelData] = useState<LabelView | null>(null)
  const [labelErr, setLabelErr] = useState<string | null>(null)

  async function openLabels(d: DatasetRow) {
    const key = dsKey(d)
    if (labelsFor === key) { setLabelsFor(null); return }
    setLabelsFor(key); setLabelData(null); setLabelErr(null)
    const path = '/api/datasets/labels'
    try {
      setLabelData(await api<LabelView>(`${path}?root=${encodeURIComponent(d.root || '')}`))
    } catch (e) {
      // No special-casing here: api() already replaces a 404 on a route this server does not route
      // with the "restart the dashboard" explanation (lib/serverAge), so the message is right for
      // every screen at once.
      setLabelErr(e instanceof HttpError ? e.message : String(e))
    }
  }

  const wantedHoldout = holdout(form.val_episodes, selectedRow(datasets, form)?.total_episodes ?? null)
  const holdoutSupport = fieldSupport(srvFields, 'val_episodes', srvHeard)
  const wantedEpisodes = numField(collect.n_episodes, { what: 'episodes', min: 1, max: 500, remedy: 'collect in batches' })
  const wantedSeconds = numField(collect.duration, { what: 'seconds per episode', min: 1, max: 600 })

  const submitCollect = async () => {
    if (!collect.dataset_root.trim()) return
    setBusy(true); setMsg(null)
    try {
      const j = await post('/api/collect', {
        dataset_root: collect.dataset_root,
        instruction: collect.instruction,
        n_episodes: wantedEpisodes.value,
        duration: wantedSeconds.value,
        robot_name: collect.robot_name,
      })
      setMsg(j.peer_id
        ? `▶ collecting ${j.n_episodes} episodes as ${j.peer_id} — watch it in the fleet grid; dataset appears below when done`
        : `⚠ ${JSON.stringify(j).slice(0, 200)}`)
      if (j.peer_id) setTimeout(refresh, 15000)
    } catch (e) { failed('collect', e) }
    setBusy(false)
  }

  const replay = async (d: Dataset) => {
    // Replay reads the chosen episode off the disk. A Hub row is not on this disk yet,
    // so replaying it would fail somewhere deep in a loader; say so here.
    if (!d.root) {
      setMsg(`⚠ ${d.repo_id} is on the Hub, not on this machine — train with it (the trainer downloads it) or clone it locally to replay`)
      return
    }
    setBusy(true); setMsg(null)
    try {
      const choice = episodeChoice(d, episodeBox[dsKey(d)])
      if (!choice.ok) { setMsg(`⚠ ${choice.reason}`); return }
      const j = await post('/api/replay', { repo_id: d.repo_id, root: d.root, episode: choice.episode })
      setMsg(j.peer_id
        ? `▶ replaying ${d.repo_id} ep${choice.episode} as ${j.peer_id} — watch it in the fleet grid`
        : `⚠ ${JSON.stringify(j).slice(0, 200)}`)
    } catch (e) { failed('replay', e) }
    setBusy(false)
  }

  /**
   * A checkpoint the server could not confirm on disk, held back from staging until the operator
   * says they mean it.
   */
  const [stageAnyway, setStageAnyway] = useState<{ job: Job; ckpt: string; message: string } | null>(null)

  const exportCkpt = async (job: Job) => {
    setBusy(true)
    try {
      const j = await post('/api/training/export', { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model })
      const art: ArtifactVerdict | undefined = j?.artifact
      if (j.status !== 'success') setMsg(`✗ ${j.text?.slice(0, 250)}`)
      else if (j.deployable === false) {
        // The export RAN - that is why the trainer's ✓ is still shown - and what it produced is not a
        // policy.
        setMsg(`⚠ the export succeeded but the artifact is not usable: ${art?.message ?? 'the checkpoint could not be confirmed on disk'}`)
      } else setMsg(`✓ ${j.text?.slice(0, 250)}${art?.warning ? ` — ⚠ ${art.warning}` : ''}`)
    } catch (e) { failed('export', e) }
    setBusy(false)
  }

  // "Deploy" cannot start a policy from here - the run form is per-robot and a policy moves a
  // real arm.
  const deployCkpt = async (job: Job) => {
    setBusy(true)
    try {
      const j = await post('/api/training/export', { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model })
      const ckpt = j?.data?.exported_model
      const art: ArtifactVerdict | undefined = j?.artifact
      if (j.status !== 'success' || typeof ckpt !== 'string' || !ckpt) {
        setMsg(`✗ nothing deployable: ${j.text?.slice(0, 200) ?? 'export returned no artifact path'}`)
      } else if (j.deployable === false) {
        // Do NOT stage it. Staging prefills a run form the operator then presses Run on, and by that
        // point the checkpoint's problem has become "the arm did not move and I do not know why".
        setStageAnyway({ job, ckpt, message: art?.message ?? 'the checkpoint could not be confirmed on disk' })
        setMsg(null)
      } else {
        setDeployIntent({
          checkpoint: ckpt,
          policy_type: guessPolicyType(job.base_model),
          source: `training job ${job.job_id} (${job.base_model || job.provider})`,
        })
        setMsg('🚀 checkpoint staged — close this sheet and open a robot\u2019s run form: it will be prefilled, and nothing runs until you press Run there')
      }
    } catch (e) { failed('export', e) }
    setBusy(false)
  }

  /** The operator overrode the disk check. Stage exactly what deploy would have staged. */
  const stageRegardless = () => {
    if (!stageAnyway) return
    const { job, ckpt } = stageAnyway
    setDeployIntent({
      checkpoint: ckpt,
      policy_type: guessPolicyType(job.base_model),
      source: `training job ${job.job_id} (${job.base_model || job.provider}) — staged over an unconfirmed artifact`,
    })
    setStageAnyway(null)
    setMsg('🚀 checkpoint staged over the warning — open a robot\u2019s run form; nothing runs until you press Run there')
  }

  // The form re-told as one sentence. A grid of fields answers "what can I
  // set"; the sentence answers the question that actually matters before a
  // multi-hour job: "what did I just ask for?"
  const datasetPicked = form.dataset_root || form.dataset_repo_id
  const datasetLabel = selectDataset(datasets, selectionKey(form)).label || null
  const stepsPhrase = wantedSteps.problem ? `an unset number of` : fmtStep(wantedSteps.value)
  const missingExtra = missingForProvider(form.provider, form as Record<string, string>)
  const story = datasetPicked
    ? `Fine-tune ${form.base_model || 'lerobot/smolvla_base'} on ${datasetLabel ?? datasetPicked} for ${stepsPhrase} steps (${form.method}), saving to ${form.output_dir || '…pick an output dir'}.${wantedHoldout.send ? ` Holding out the last ${wantedHoldout.send} episodes to score it.` : ''}`
      + (missingExtra ? ` This will be refused until you fill it in: ${missingExtra}.` : '')
    : 'Pick a dataset to begin — the plan reads back here before anything runs.'

  return (
    /**
     * role + label like RecordPanel's sheet: this is a full-bleed layer over the fleet, and a
     * screen reader that is not told it entered a dialog reads it as more of the page it just
     * left.
     */
    <div ref={sheetRef} className="train-sheet" role="dialog" aria-label="Training">
      <div className="train-head">
        <h2>🎓 Training</h2>
        <button className="dock-min" onClick={onClose} aria-label="close training" title="Escape">✕</button>
      </div>

      <div className="train-form">
        <p className={`train-story${datasetPicked ? '' : ' empty'}`}>{story}</p>
        <label className="field"><span>provider</span>
          <select value={form.provider} onChange={e => set('provider', e.target.value)} disabled={busy}>
            {/* Still LISTED, so the capability is not hidden and nobody hunts for a provider they know exists — but not selectable, with the reason attached rather than delivered as an error after the fact. */}
            {trainers.map(t => (
              <option key={t} disabled={t in unsupported}
                      title={unsupported[t] ?? undefined}>
                {t}{t in unsupported ? ' — not from this form' : ''}
              </option>
            ))}
          </select>
        </label>
        {/* The refusal reasons MOVED behind a disclosure (scannability law: prose is not default
            furniture). Nothing is lost: each refused option still says so inline, and its title
            carries the reason on hover — this is the same sentences, one tap away. */}
        {Object.keys(unsupported).length > 0 && (
          <details className="hint">
            <summary>{Object.keys(unsupported).length} provider{Object.keys(unsupported).length === 1 ? ' is' : 's are'} not trainable from this form — why?</summary>
            {/* One line per DISTINCT reason, names grouped: same reason never printed twice, different reasons never merged. */}
            {[...new Set(Object.values(unsupported))].map(reason => (
              <p className="hint" key={reason}>
                {Object.keys(unsupported).filter(k => unsupported[k] === reason).sort().join(' and ')}
                {' cannot be trained from here: '}{reason}.
              </p>
            ))}
          </details>
        )}
        <label className="field"><span>dataset</span>
          <input value={dsQuery} onChange={e => setDsQuery(e.target.value)} disabled={busy}
                 placeholder="search this machine and the Hub — e.g. pusht, so101, your org" />
          {/* Selecting sets EXACTLY ONE of dataset_root / dataset_repo_id: a local dataset trains from its path, a Hub one from its repo id, and sending both would leave the trainer to pick for you. */}
          <select value={selectionKey(form)}
                  onChange={e => {
                    const sel = selectDataset(datasets, e.target.value)
                    setForm(f => ({ ...f, dataset_root: sel.dataset_root, dataset_repo_id: sel.dataset_repo_id }))
                  }} disabled={busy}>
            <option value="">— pick a dataset —</option>
            {datasets.filter(d => d.local !== false).length > 0 && (
              <optgroup label="on this machine">
                {datasets.filter(d => d.local !== false).map(d => (
                  <option key={dsKey(d)} value={dsKey(d)}>
                    {/* an abandoned recording's folder lists as a dataset for ever. */}
                    {datasetMark(d).glyph}{d.repo_id} ({d.total_episodes ?? '?'} eps{d.robot_type && d.robot_type !== 'unknown' ? `, ${d.robot_type}` : ''})
                  </option>
                ))}
              </optgroup>
            )}
            {datasets.filter(d => d.local === false).length > 0 && (
              <optgroup label="HuggingFace Hub — downloaded when training starts">
                {datasets.filter(d => d.local === false).map(d => (
                  <option key={dsKey(d)} value={dsKey(d)}>
                    {d.repo_id}{d.downloads ? ` · ${d.downloads.toLocaleString()} downloads` : ''}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          {/* ONE verdict, so the rows, the failure and the sentence cannot describe three different moments (lib/datasetHint.ts). */}
          {(() => {
            const h = datasetHint({
              query: dsQuery, shownQuery: dsShownQuery, count: datasets.length,
              problem: dsProblem, anonymous: dsAuth?.authenticated === false,
              authDetail: dsAuth?.detail ?? null,
            })
            return (
              <>
                {h.text && <span className={`hint${h.tone === 'warn' ? ' warn' : ''}`} role="status" aria-live="polite">{h.text}</span>}
                {h.auth && <span className="hint">{h.auth}</span>}
              </>
            )
          })()}
        </label>
        <label className="field"><span>base model</span>
          <input value={form.base_model} onChange={e => set('base_model', e.target.value)} disabled={busy} placeholder="lerobot/smolvla_base" />
        </label>
        <label className="field"><span>output dir</span>
          <input value={form.output_dir} onChange={e => set('output_dir', e.target.value)} disabled={busy}
                 placeholder="/tmp/my_policy_ckpt" aria-describedby="train-outdir-say"
                 aria-invalid={outSay.blocked || outSay.confirmable} />
          {/* the one hand-typed path on the golden path, offered as one click — never
              silently written, and the output-dir verdict re-judges whatever lands here */}
          {(() => {
            const sug = suggestOutputDir(form, form.output_dir)
            return sug ? (
              <button type="button" className="btn ghost suggest" disabled={busy}
                      onClick={() => set('output_dir', sug)}>
                use {sug}
              </button>
            ) : null
          })()}
          {/* a run into an existing directory DELETES it (the trainer rmtree's a dir with no resumable checkpoint). */}
          <span id="train-outdir-say" className={`fieldsay${outSay.tone === 'info' ? '' : ' bad'}`} role="status" aria-live="polite">
            {outSay.text ?? ''}
          </span>
        </label>
        {outSay.confirmable && (
          <label className="field check consent-clear">
            <input type="checkbox" checked={clearArmedFor === (outDir?.path ?? form.output_dir.trim())}
                   onChange={e => setClearArmedFor(e.target.checked ? (outDir?.path ?? form.output_dir.trim()) : null)}
                   disabled={busy} />
            <span>{outSay.confirmLabel}</span>
          </label>
        )}
        {/* appears only for the provider that needs it. */}
        {extraFields(form.provider).map(f => (
          <label className="field" key={f.key}><span>{f.label}</span>
            <input value={form[f.key]} onChange={e => set(f.key, e.target.value)} disabled={busy}
                   placeholder={f.placeholder} aria-describedby={`train-${f.key}-say`}
                   aria-invalid={f.required && !form[f.key].trim()} />
            <span id={`train-${f.key}-say`} className={`fieldsay${f.required && !form[f.key].trim() ? ' bad' : ''}`}>
              {f.say}
            </span>
          </label>
        ))}
        <div className="train-row">
          <label className="field"><span>steps</span>
            <input type="number" value={form.steps} onChange={e => set('steps', e.target.value)} disabled={busy}
                   aria-invalid={!!wantedSteps.problem} aria-describedby="train-steps-say" />
            <span id="train-steps-say" className={`fieldsay${wantedSteps.problem ? ' bad' : ''}`}>
              {wantedSteps.problem ?? wantedSteps.note ?? ''}
            </span>
          </label>
          <label className="field"><span>val episodes</span>
            <input type="number" value={holdoutSupport.ok ? form.val_episodes : ''} placeholder="none"
                   onChange={e => set('val_episodes', e.target.value)}
                   disabled={busy || !holdoutSupport.ok}
                   aria-invalid={!!wantedHoldout.problem} aria-describedby="train-val-say" />
            <span id="train-val-say" className={`fieldsay${wantedHoldout.problem || !holdoutSupport.ok ? ' bad' : ''}`}>
              {holdoutSupport.why || wantedHoldout.problem || wantedHoldout.say}
            </span>
          </label>
          <label className="field"><span>method</span>
            <select value={form.method} onChange={e => set('method', e.target.value)} disabled={busy}>
              <option value="lora">lora</option>
              <option value="full">full</option>
            </select>
          </label>
        </div>
        <div className="train-actions">
          <button className="btn ghost" onClick={() => submit(true)} disabled={busy || !!wantedSteps.problem}>✓ validate</button>
          <button className="btn go wide" onClick={() => submit(false)}
                  disabled={busy || !datasetPicked || !!wantedSteps.problem || (holdoutSupport.ok && !!wantedHoldout.problem) || !gate.ok}
                  title={gate.why ?? undefined}>
            {outSay.confirmable && gate.ok ? '▶ delete and train' : '▶ train'}
          </button>
        </div>
        {msg && <div className="train-msg">{msg}</div>}
        {/* Not started, and not blocked either. */}
        {dsWarn && dsWarn.key === selectionKey(form) && (
          <div className="train-msg warn artifact-hold" role="alert">
            <div>{dsWarn.recording ? '⏺' : '⚠'} not started: {dsWarn.reason}</div>
            {/* for a LIVE session the useful next move is the record screen, not another dataset — and this tab cannot navigate, so it says where to look rather than pretending to take them there. */}
            {dsWarn.recording && <div className="jstate">the record screen shows this session's progress; training can start the moment it closes</div>}
            <div className="artifact-hold-actions">
              <button className="btn ghost" onClick={() => { setDsWarn(null); setDsOverride(null) }}>pick another dataset</button>
              <button className="btn" onClick={() => { setDsOverride(dsWarn.key); setDsWarn(null); setMsg('⚠ dataset warning overridden — press start training again') }}
                      title={dsWarn.recording
                        ? 'the trainer would read episodes as they are still being written - only useful if the session is about to close'
                        : 'the check reads metadata only - insist if you know the episodes are there'}>
                train on it anyway
              </button>
            </div>
          </div>
        )}
        {/* Held back, not blocked. */}
        {stageAnyway && (
          <div className="train-msg warn artifact-hold" role="alert">
            <div>⚠ not staged: {stageAnyway.message}</div>
            <div className="artifact-hold-actions">
              <button className="btn ghost" onClick={() => setStageAnyway(null)}>keep it unstaged</button>
              <button className="btn" onClick={stageRegardless}
                      title="the disk check can be wrong - stage it and find out on the run form">
                stage it anyway
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="train-form">
        <button className="btn ghost" onClick={() => setShowCollect(s => !s)}>
          {showCollect ? '▾' : '▸'} 📹 collect new dataset (sim rollouts)
        </button>
        {showCollect && (
          <>
            <label className="field"><span>dataset root (path)</span>
              <input value={collect.dataset_root} placeholder="/tmp/my_demos"
                onChange={e => setCollect(c => ({ ...c, dataset_root: e.target.value }))} disabled={busy} />
            </label>
            <label className="field"><span>instruction</span>
              <input value={collect.instruction}
                onChange={e => setCollect(c => ({ ...c, instruction: e.target.value }))} disabled={busy} />
            </label>
            <div className="train-row">
              <label className="field"><span>episodes</span>
                <input type="number" value={collect.n_episodes} aria-invalid={!!wantedEpisodes.problem}
                  aria-describedby="collect-episodes-say"
                  onChange={e => setCollect(c => ({ ...c, n_episodes: e.target.value }))} disabled={busy} />
                <span id="collect-episodes-say" className={`fieldsay${wantedEpisodes.problem ? ' bad' : ''}`}>
                  {wantedEpisodes.problem ?? wantedEpisodes.note ?? ''}
                </span>
              </label>
              <label className="field"><span>sec/episode</span>
                <input type="number" value={collect.duration} aria-invalid={!!wantedSeconds.problem}
                  aria-describedby="collect-seconds-say"
                  onChange={e => setCollect(c => ({ ...c, duration: e.target.value }))} disabled={busy} />
                <span id="collect-seconds-say" className={`fieldsay${wantedSeconds.problem ? ' bad' : ''}`}>
                  {wantedSeconds.problem ?? wantedSeconds.note ?? ''}
                </span>
              </label>
              <label className="field"><span>robot</span>
                <input value={collect.robot_name}
                  onChange={e => setCollect(c => ({ ...c, robot_name: e.target.value }))} disabled={busy} />
              </label>
            </div>
            <div className="train-actions">
              <button className="btn go wide" onClick={submitCollect}
                disabled={busy || !collect.dataset_root.trim() || !!wantedEpisodes.problem || !!wantedSeconds.problem}>
                📹 collect
              </button>
            </div>
          </>
        )}
      </div>

      <div className="train-jobs">
        <h3>Datasets</h3>
        {datasets.length === 0 && (
          <div className="dock-hint">
            {dsShownQuery !== null && dsShownQuery.trim() !== dsQuery.trim()
              ? `Searching for “${dsQuery.trim()}”…`
              : dsProblem
                ? `No local LeRobotDatasets, and the Hub could not be searched — ${dsProblem}`
                : dsQuery
                  ? `Nothing matches “${dsQuery}” here or on the Hub.`
                  : 'No LeRobotDatasets on this machine — type above to search the Hub, or record one in Collect.'}
          </div>
        )}
        {datasets.map(d => (
          <div className="train-job" key={dsKey(d)}>
            <div className="train-job-head">
              <b>{d.repo_id}</b>
              <span className="jstate">
                {/* "0 eps" on a dataset being recorded right now reads as empty when it is filling — the count is a snapshot of flushed metadata, not of the session. */}
                {d.recording ? '⏺ recording now' : d.root
                  ? `${d.total_episodes ?? '?'} eps · ${d.fps ?? '?'} fps`
                  : `Hub${d.downloads ? ` · ${d.downloads.toLocaleString()} downloads` : ''}`}
              </span>
            </div>
            <div className="train-job-actions">
              {/* Replay reads an episode off this disk, so it is offered only for what is actually here — a disabled button with the reason beats a click that dies inside a dataset loader. */}
              <input className="ep-box" type="number" min={0} inputMode="numeric"
                value={episodeBox[dsKey(d)] ?? ''}
                placeholder={typeof d.total_episodes === 'number' && d.total_episodes > 0 ? `0–${d.total_episodes - 1}` : '0'}
                aria-label={`episode to replay from ${d.repo_id}`}
                title={typeof d.total_episodes === 'number' && d.total_episodes > 0
                  ? `This dataset has ${d.total_episodes} episode${d.total_episodes === 1 ? '' : 's'} — blank replays episode 0`
                  : 'Episode index — blank replays episode 0'}
                disabled={busy || !replayable(d).ok}
                onChange={e => setEpisodeBox(prev => ({ ...prev, [dsKey(d)]: e.target.value }))} />
              <button className="btn ghost" onClick={() => replay(d)} disabled={busy || !replayable(d).ok}
                title={episodeChoice(d, episodeBox[dsKey(d)]).ok
                  ? `${replayable(d).reason} — ${episodeChoice(d, episodeBox[dsKey(d)]).reason}`
                  : episodeChoice(d, episodeBox[dsKey(d)]).reason}>
                🎬 replay in sim
              </button>
              {/* #2486: what was each episode judged to be? */}
              <button className="btn ghost" onClick={() => openLabels(d)} disabled={!labelsGate(d).ok}
                title={labelsGate(d).reason} aria-expanded={labelsFor === dsKey(d)}>
                🏷 labels
              </button>
            </div>
            {labelsFor === dsKey(d) && (() => {
              const sum = labelSummary(labelData, labelErr)
              return (
                <div className="ds-labels">
                  <div className={sum.tone === 'warn' ? 'dock-hint warn' : 'dock-hint'}>{sum.text}</div>
                  {(labelData?.episodes ?? []).map(ep => {
                    const line = labelRowLine(ep)
                    return (
                      <div className={line.muted ? 'ds-label-row muted' : 'ds-label-row'} key={ep.episode_index}>
                        <span className="ds-label-badge">{line.badge}</span>
                        <b>episode {ep.episode_index}</b>
                        <span className="jstate">{line.detail}</span>
                      </div>
                    )
                  })}
                </div>
              )
            })()}
          </div>
        ))}

        <h3>Jobs</h3>
        {/* The only automatic speech on this screen: one atomic sentence when a run ends. */}
        <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">{jobSay}</div>
        {/* Rendered for ANY count: a partial list is the dangerous case, because the cards that survived make it look complete. */}
        {(() => {
          const notice = jobsLedgerNotice({ count: jobs.length, problem: jobsProblem })
          return notice.text ? (
            <div className={notice.tone === 'warn' ? 'dock-hint warn' : 'dock-hint'}>{notice.text}</div>
          ) : null
        })()}
        {jobs.map(job => {
          const st = statuses[job.job_id]
          const state = st?.data?.status ?? '…'
          const fresh = trainingFreshness({
            polledAtS: polledAt[job.job_id], nowS,
            failures: pollFail[job.job_id]?.n, error: pollFail[job.job_id]?.msg || null,
            state: st?.data?.status ?? null,
          })
          return (
            <div className={`train-job${fresh.stale ? ' stalefeed' : ''}`} key={job.job_id ?? Math.random()}>
              <div className="train-job-head">
                <b>{job.provider}</b>
                {/* The chip is the word an operator trusts for hours, so it says how old that word is - always, not only when it goes stale. */}
                <span className={`jstate ${state}`} title={fresh.title}>{state}</span>
                {fresh.stale && <span className="jstale" title={fresh.title}>
                  as of {fresh.ageS != null && fresh.ageS < 90 ? `${Math.round(fresh.ageS)}s` : `${Math.round((fresh.ageS ?? 0) / 60)}m`} ago
                </span>}
              </div>
              {fresh.note && <div className="train-msg warn">{fresh.note}</div>}
              <div className="train-job-meta">
                {job.dataset?.split('/').slice(-2).join('/')} → {job.output_dir} · {job.steps} steps
              </div>
              {(() => {
                const m = st?.data?.metrics as Record<string, any> | undefined
                if (!m || Object.keys(m).length === 0) return null
                const step = typeof m.latest_step === 'number' ? m.latest_step : null
                const total = typeof job.steps === 'number' ? job.steps : Number(job.steps) || null
                return (
                  <>
                    {step !== null && total ? (
                      <div className="train-progress" role="progressbar"
                           aria-valuemin={0} aria-valuemax={total} aria-valuenow={Math.min(step, total)}>
                        <div className="train-progress-fill"
                             style={{ width: `${Math.min(100, (step / total) * 100)}%` }} />
                        <span className="train-progress-label">{fmtStep(step)} / {fmtStep(total)} steps</span>
                      </div>
                    ) : null}
                    <LossSpark trace={traces[job.job_id] ?? []} />
                    {m.latest_loss !== undefined && !Number.isFinite(m.latest_loss) && (
                      <div className="train-msg">⚠ loss is NaN — the run is executing but NOT learning (check LR / data)</div>
                    )}
                    {m.liveness_ok === false && state === 'running' && (
                      <div className="train-msg">⚠ no step lines in the log yet — still warming up, or stalled</div>
                    )}
                  </>
                )
              })()}
              <div className="train-job-actions">
                <button className="btn ghost" onClick={() => exportCkpt(job)} disabled={busy}>📦 export checkpoint</button>
                {state === 'success' && (
                  <button className="btn ghost" onClick={() => deployCkpt(job)} disabled={busy}
                          title="stages this checkpoint into a robot's run form — never starts it">🚀 deploy…</button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
