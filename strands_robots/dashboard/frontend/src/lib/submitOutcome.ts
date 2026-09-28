/**
 * What the training tab may claim when a request that STARTS SOMETHING fails. Every action
 * here answered a failure with `⚠ <message>`, which reads as "it did not happen".
 */
import { refusedBeforeActing } from './estopOutcome'

export type SideEffectKind = 'training' | 'collect' | 'replay' | 'export'

export interface SideEffectVerdict {
  text: string
  delivered: 'no' | 'unknown'
  /** True when pressing the button again could run the action a SECOND time. */
  doubleRunRisk: boolean
}

const NOUN: Record<SideEffectKind, string> = {
  training: 'the training job',
  collect: 'the collection run',
  replay: 'the replay',
  export: 'the export',
}

const DUPLICATE: Record<SideEffectKind, string> = {
  training: 'Pressing train again could start a SECOND run on the same GPU, writing into the same output_dir — check the job list below (it refreshes now) before you do.',
  collect: 'Pressing collect again could spawn a SECOND recorder appending to the same dataset — look for a new peer in the fleet grid first.',
  replay: 'Pressing replay again could spawn a SECOND peer driving the same arm — look for a new peer in the fleet grid first.',
  export: 'It is safe to retry an export, but check the artifact path before assuming nothing was written.',
}

export function sideEffectVerdict(input: {
  kind: SideEffectKind
  status?: number | null
  message?: string | null
}): SideEffectVerdict {
  const noun = NOUN[input.kind] ?? 'the request'
  const why = String(input.message ?? '').trim() || 'no detail'

  if (refusedBeforeActing(input.status)) {
    return {
      text: `✗ refused (${input.status}: ${why}) — ${noun} was NOT started, nothing is running.`,
      delivered: 'no',
      doubleRunRisk: false,
    }
  }
  const transport = !Number(input.status ?? 0)
  const head = transport
    ? `⚠ no answer came back (${why}) — ${noun} MAY have started; this page cannot tell.`
    : `⚠ the server failed mid-request (${input.status}: ${why}) — ${noun} MAY have started anyway.`
  return {
    text: `${head} ${DUPLICATE[input.kind] ?? ''}`.trim(),
    delivered: 'unknown',
    doubleRunRisk: input.kind !== 'export',
  }
}
