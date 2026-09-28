/** the rate a recording session DECLARES — and the one honest thing to do with the measured rate when it disagrees. */
export interface FpsField {
  /** the number to send; only meaningful when `problem` is null */
  value: number
  /** why this input cannot be used, in the operator's words — null when fine */
  problem: string | null
  /** a change we DID make and therefore must admit (30.6 → 30) */
  note: string | null
}

export const DEFAULT_FPS = 30
const MIN_FPS = 1
const MAX_FPS = 60

/** Parse the fps box. Mirrors episodeTarget's posture: correct nothing in silence. */
export function fpsField(raw: string): FpsField {
  const text = (raw ?? '').trim()
  if (!text) return { value: DEFAULT_FPS, problem: null, note: null }
  const n = Number(text)
  if (!Number.isFinite(n)) {
    return { value: DEFAULT_FPS, problem: `"${text}" is not a number of frames per second`, note: null }
  }
  if (n < MIN_FPS || n > MAX_FPS) {
    return {
      value: DEFAULT_FPS,
      problem: `fps must be between ${MIN_FPS} and ${MAX_FPS} — ${n} is outside what a dataset can declare`,
      note: null,
    }
  }
  const rounded = Math.round(n)
  return {
    value: rounded,
    problem: null,
    note: rounded !== n ? `recording at ${rounded} fps (a dataset's rate is a whole number)` : null,
  }
}

export interface FpsNoticeLike {
  declared_fps: number
  measured_fps: number
  ratio: number
  slower: boolean
  detail: string
}

export interface FpsSuggestion {
  /** the value to put in the box, as text (the field is raw text) */
  fps: string
  /** the button's words */
  label: string
  /** why, including what the mismatch costs — shown next to the button */
  why: string
}

export function fpsSuggestion(notice: FpsNoticeLike | null | undefined): FpsSuggestion | null {
  if (!notice || !Number.isFinite(notice.measured_fps)) return null
  const measured = Math.round(notice.measured_fps)
  if (measured < MIN_FPS || measured > MAX_FPS) return null
  if (measured === Math.round(notice.declared_fps)) return null
  return {
    fps: String(measured),
    label: `use ${measured} fps next session`,
    // Deliberately future-tense: the current session's episodes keep their declaration, and
    // pretending otherwise would be the third lie in this chain.
    why: notice.slower
      ? `this session is already stamped ${notice.declared_fps} fps and cannot be re-declared; `
        + `recording the next one at ${measured} makes its timestamps match real time`
      : `capture is running faster than declared; ${measured} fps would describe it honestly`,
  }
}
