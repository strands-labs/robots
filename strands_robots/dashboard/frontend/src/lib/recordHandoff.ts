/**
 * The record → train handoff. Finishing a dataset used to end in a dismissable toast,
 * and the operator's very next job — training on it — began with re-finding that
 * minutes-old dataset by memory in another screen. The close receipt now carries the
 * dataset's identity and resolved path (record_worker.close), and this rule decides
 * when a "train on it" offer is honest.
 */

export interface CloseReceipt {
  ok: boolean
  detail?: string
  dataset?: string
  /** resolved on-disk directory, when the server could compute it */
  root?: string
  episodes_kept?: number
  camera_notice?: { present?: string[]; missing?: string[] } | null
}

export interface TrainHandoff {
  /** what the training form should be seeded with */
  prefill: { dataset_root: string }
  label: string
  /** a fact worth carrying into the decision, not a blocker */
  caveat: string | null
}

export function trainHandoff(r: CloseReceipt | null | undefined): TrainHandoff | null {
  if (!r?.ok) return null
  // an empty dataset cannot train anything — offering would hand the trainer a refusal
  if (!r.episodes_kept || r.episodes_kept <= 0) return null
  const where = (r.root ?? '').trim() || (r.dataset ?? '').trim()
  if (!where) return null
  const caveat =
    r.camera_notice && (r.camera_notice.missing?.length ?? 0) > 0
      ? (r.camera_notice.present?.length ?? 0) > 0
        ? 'some image channels are missing — a visual policy trains on less than you saw'
        : 'no image channel at all — this dataset cannot train a visual policy'
      : null
  return {
    prefill: { dataset_root: where },
    label: `train on it (${r.episodes_kept} episode${r.episodes_kept === 1 ? '' : 's'}) →`,
    caveat,
  }
}
