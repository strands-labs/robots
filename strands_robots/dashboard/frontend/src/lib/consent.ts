/** A refusal the operator can answer: the client half of consent-gated spawns and tasks. */
import { post } from './endpoints'

export type ConsentKind = 'trust_remote_code' | 'hf_repo_allow' | string

export interface ConsentNeed {
  kind: ConsentKind
  scope: string
  title: string
  risk: string
  env_var?: string
  subject?: string | null
  grants?: string[]
  grantable?: boolean
  message?: string
}

export interface ConsentResult {
  granted: boolean
  already_granted?: boolean
  respawn_required?: boolean
  scope?: string
  grants?: string[]
  env_written?: string[]
  env_removed?: string[]
  note?: string
}

function looksLikeNeed(value: any): value is ConsentNeed {
  return (
    !!value &&
    typeof value === 'object' &&
    typeof value.kind === 'string' &&
    typeof value.scope === 'string' &&
    typeof value.title === 'string' &&
    typeof value.risk === 'string'
  )
}

/** Dig `needs_consent` out of any shape the API answers with. */
export function findConsent(payload: any, depth = 3): ConsentNeed | null {
  if (!payload || typeof payload !== 'object' || depth <= 0) return null
  if (looksLikeNeed((payload as any).needs_consent)) return (payload as any).needs_consent
  for (const key of ['result', 'detail', 'body', 'error']) {
    const nested = (payload as any)[key]
    if (nested && typeof nested === 'object') {
      const found = findConsent(nested, depth - 1)
      if (found) return found
    }
  }
  return null
}

/** Can this be approved at all? */
export function canApprove(need: ConsentNeed): boolean {
  if (typeof need.grantable === 'boolean') return need.grantable
  // Fallback for a server older than that field: a subject is required wherever the grant IS the
  // subject — every allowlist kind, not just the model repository.
  if (need.kind === 'hf_repo_allow' || need.kind === 'policy_type_allow'
      || need.kind === 'policy_host_allow') return !!need.subject
  // Otherwise: approvable when the server named something it would actually set.
  return !!need.env_var || !!need.grants?.length
}

/** Why approving is impossible, in the words of the actual reason. */
export function blockedReason(need: ConsentNeed): string {
  if (need.kind === 'hf_repo_allow') {
    return 'The repository name in this refusal could not be read safely, so there is nothing to '
      + 'allow. Check the model path and try again.'
  }
  // Named separately because "check the model path" is false advice for these two, and a
  // security dialog's explanation is what the operator carries away (this function's own reason
  // for existing).
  if (need.kind === 'policy_type_allow') {
    return 'The policy name in this refusal could not be read safely, so there is nothing to allow. '
      + 'Check the policy type or provider you asked for and try again.'
  }
  if (need.kind === 'policy_host_allow') {
    return 'The address in this refusal could not be read as a host, so there is nothing to allow. '
      + 'Use a plain hostname or IP (optionally with a port) and try again.'
  }
  return 'This refusal did not say what approving would change, so there is nothing to grant from '
    + 'here. It may come from a newer guard than this page — reload, and if it persists, grant it '
    + 'in the environment instead.'
}

/** How dangerous is saying yes? Drives which button style the sheet uses. */
const OPEN_ENDED: ReadonlySet<string> = new Set(['trust_remote_code', 'agent_physical_motion'])
// policy_type_allow is bounded in the same way hf_repo_allow is: one name added to one list.
// policy_host_allow is deliberately NOT here — it stays 'danger' by the unknown-kind rule,
// because approving it sends camera frames and joint states to another machine and lets what
// that machine returns drive the arms.
const BOUNDED: ReadonlySet<string> = new Set(['hf_repo_allow', 'teleop_degree_units', 'policy_type_allow'])

export function severity(need: ConsentNeed): 'danger' | 'warn' {
  if (OPEN_ENDED.has(need.kind)) return 'danger'
  return BOUNDED.has(need.kind) ? 'warn' : 'danger'
}

/** The server owns the variable and the value; we send kind + subject only. */
export function approveConsent(need: ConsentNeed): Promise<ConsentResult> {
  return post<ConsentResult>('/api/consent', { kind: need.kind, subject: need.subject ?? null })
}

/** What to tell the operator after a grant, and whether an immediate retry can work. */
export function afterApproval(
  result: ConsentResult,
  target: 'spawn' | 'peer',
): { retryNow: boolean; note: string } {
  const granted = result.granted || result.already_granted
  if (!granted) {
    return { retryNow: false, note: result.note || 'nothing was granted — retrying would fail the same way' }
  }
  if (target === 'spawn') {
    // The refused process never started, so the next spawn is the retry and it
    // inherits the grant from this dashboard's environment.
    return { retryNow: true, note: result.note || 'granted — starting again' }
  }
  return {
    retryNow: false,
    note:
      result.note ||
      'granted — but this robot is already running with the old permissions. Respawn it, then run again.',
  }
}

/** Does this machine grant anything beyond the SDK's own defaults? */
const NOT_A_GRANT: ReadonlySet<string> = new Set(['locks', 'kinds', 'env_file'])

export function nothingGranted(state: Record<string, unknown> | null | undefined): boolean {
  if (!state) return false  // unknown is not "nothing": say nothing rather than reassure
  for (const [key, value] of Object.entries(state)) {
    if (NOT_A_GRANT.has(key)) continue
    if (Array.isArray(value)) {
      if (value.length > 0) return false
    } else if (value && typeof value === 'object') {
      if ((value as { granted?: unknown }).granted) return false
    } else if (value === true) {
      return false
    }
  }
  return true
}
