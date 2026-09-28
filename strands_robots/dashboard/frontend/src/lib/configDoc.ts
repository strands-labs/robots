/**
 * The settings document the drawer renders, from whichever route this server has.
 *
 * The branch served one composite `/api/config` (settings + agent status + mesh
 * info + policy catalog + .env rows). Main serves the settings tree alone at
 * `/api/settings` as `{settings: {section: {key: value}}, file}` and the console's
 * model at `/api/agent`. When `/api/config` is absent (404) the document is
 * composed here from those two, so the drawer works on main as it stands; the
 * fields only the composite carries (policies, env rows, mesh info) are empty and
 * the drawer already renders an empty list for each.
 */
import type { ConfigDoc } from '../types'
import { api, post, HttpError } from './endpoints'
import type { ApplyResult } from './useConfig'

interface SettingsDoc { settings: Record<string, Record<string, any>>; file: string }
interface AgentInfo { model?: string }

let _composite: boolean | null = null

/** Whether this server has the composite route. Remembered per page load; a backend switch remounts the app. */
export function forgetConfigRoute(): void { _composite = null }

export function composeConfigDoc(s: SettingsDoc, agent: AgentInfo | null): ConfigDoc {
  const t = s.settings ?? {}
  const a = t.agent ?? {}
  const v = t.voice ?? {}
  const m = t.mesh ?? {}
  const sec = t.security ?? {}
  return {
    agent: {
      model_id: a.model_id ?? null,
      known_models: agent?.model ? [agent.model] : [],
      system_prompt: a.system_prompt ?? '',
      is_default_prompt: !a.system_prompt,
      temperature: a.temperature ?? null,
      max_tokens: a.max_tokens ?? null,
      built: !!agent,
    },
    voice: { provider: v.provider ?? 'openai', voice_name: v.voice_name ?? null, providers: ['openai', 'nova_sonic'] },
    mesh: {
      connect: m.connect ?? [],
      listen: m.listen ?? [],
      port: m.port ?? undefined,
      backend: m.backend ?? undefined,
      camera_hz: m.camera_hz ?? undefined,
      policy_allow: m.policy_type_allow ?? [],
      settings: m,
    },
    runtime: { trust_remote_code: !!(t.runtime ?? {}).trust_remote_code },
    security: { auth_enabled: !!sec.auth_token, cors_origins: sec.cors_origins ?? [] },
    policies: [],
    env: [],
    env_file: '',
    settings_file: s.file ?? '',
  }
}

export async function fetchConfigDoc(): Promise<ConfigDoc> {
  if (_composite !== false) {
    try {
      const doc = await api<ConfigDoc>('/api/config')
      _composite = true
      return doc
    } catch (e) {
      if (!(e instanceof HttpError) || e.status !== 404) throw e
      _composite = false
    }
  }
  const [settings, agent] = await Promise.all([
    api<SettingsDoc>('/api/settings'),
    api<AgentInfo>('/api/agent').catch(() => null),
  ])
  return composeConfigDoc(settings, agent)
}

/** The keys main's `/api/settings` knows; anything else in a drawer patch is reported as ignored. */
const SETTINGS_SECTIONS = new Set(['agent', 'voice', 'mesh', 'runtime', 'security'])

export function toSettingsPatch(body: Record<string, any>): { patch: Record<string, any>; ignored: string[] } {
  const patch: Record<string, any> = {}
  const ignored: string[] = []
  for (const [k, v] of Object.entries(body)) {
    if (SETTINGS_SECTIONS.has(k) && v && typeof v === 'object') patch[k] = v
    else ignored.push(k)
  }
  return { patch, ignored }
}

export async function saveConfigDoc(body: Record<string, any>): Promise<ApplyResult> {
  if (_composite !== false) {
    try {
      return await post<ApplyResult>('/api/config', body)
    } catch (e) {
      if (!(e instanceof HttpError) || e.status !== 404) throw e
      _composite = false
    }
  }
  const { patch, ignored } = toSettingsPatch(body)
  if (!Object.keys(patch).length) {
    return { applied: [], restart_required: [], env_written: [], skipped_masked: [], agent_reset: false, errors: [], ignored }
  }
  const r = await post<{ changed: string[]; errors: string[] }>('/api/settings', patch)
  const changed = r.changed ?? []
  return {
    applied: changed,
    // Settings main stores but only a new process reads: main has no hot re-point yet.
    startup_required: changed.filter(k => k.startsWith('mesh.') || k.startsWith('security.')),
    restart_required: [],
    env_written: [],
    skipped_masked: [],
    agent_reset: changed.some(k => k.startsWith('agent.')),
    errors: r.errors ?? [],
    ignored,
  }
}
