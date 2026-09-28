import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { ConfigDoc, PolicyProvider } from '../types'
import { fetchConfigDoc, saveConfigDoc } from './configDoc'

export interface ApplyResult {
  /** setting names this backend does not know - dropped, never stored */
  ignored?: string[]
  applied: string[]
  restart_required: string[]
  /** Q51: saved and inherited by the next spawned robot — a mesh restart cannot deliver these. */
  respawn_required?: string[]
  /** Q52: stored, and only a server start can put it into effect. */
  startup_required?: string[]
  env_written: string[]
  /** Q75: keys REMOVED from .env — distinct from written-as-empty. */
  env_removed?: string[]
  skipped_masked: string[]
  agent_reset: boolean
  errors: string[]
  mesh_restart?: { mesh_online: boolean; orphaned: string[] }
}

interface Ctx {
  config: ConfigDoc | null
  policies: PolicyProvider[]
  provider: (name: string) => PolicyProvider | undefined
  loading: boolean
  error: string | null
  reload: () => Promise<void>
  save: (body: Record<string, any>) => Promise<ApplyResult>
}

const CTX = createContext<Ctx>({
  config: null, policies: [], provider: () => undefined,
  loading: true, error: null,
  reload: async () => {}, save: async () => ({ applied: [], restart_required: [], env_written: [], skipped_masked: [], agent_reset: false, errors: [] }),
})

/**
 * One shared /api/config fetch for the whole app.
 *
 * The policy catalog is the run form's schema, so every robot card needs it -
 * fetching it per card would hit the registry once per robot on every render
 * pass.
 */
export function ConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<ConfigDoc | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      setConfig(await fetchConfigDoc())
      setError(null)
    } catch (e: any) {
      setError(e?.message ?? String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void reload() }, [reload])

  const save = useCallback(async (body: Record<string, any>) => {
    const result = await saveConfigDoc(body)
    await reload()
    return result
  }, [reload])

  const policies = config?.policies ?? []
  const byName = useMemo(() => new Map(policies.map(p => [p.name, p])), [policies])
  const provider = useCallback((name: string) => byName.get(name), [byName])

  return (
    <CTX.Provider value={{ config, policies, provider, loading, error, reload, save }}>
      {children}
    </CTX.Provider>
  )
}

export const useConfig = () => useContext(CTX)
