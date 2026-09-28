/** The warning a SUCCESSFUL spawn can carry — currently the calibration gap. */
export type SpawnNotice = { text: string; tone: 'warn' }

export function spawnNotice(body: unknown): SpawnNotice | null {
  if (!body || typeof body !== 'object') return null
  const raw = (body as Record<string, unknown>).calibration_warning
  if (typeof raw !== 'string') return null
  const text = raw.trim()
  if (!text) return null
  return { text, tone: 'warn' }
}
