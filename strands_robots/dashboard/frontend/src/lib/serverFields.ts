/** Does the SERVER this bundle is talking to accept a given form field? */

export interface FieldSupport {
  ok: boolean
  /** why it cannot be used, as something the operator can DO — or '' when it can */
  why: string
}

export function fieldSupport(
  fields: string[] | null | undefined,
  key: string,
  loaded: boolean,
): FieldSupport {
  if (!loaded) return { ok: true, why: '' }
  if (Array.isArray(fields)) {
    if (fields.includes(key)) return { ok: true, why: '' }
    return {
      ok: false,
      why: `this dashboard's server does not accept ${key} — it is running code from before the field existed. Restart the dashboard to pick it up.`,
    }
  }
  return {
    ok: false,
    why: `this dashboard's server is older than ${key} (it does not publish the field list yet). Restart the dashboard to pick it up.`,
  }
}
