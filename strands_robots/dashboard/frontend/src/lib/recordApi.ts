/** The record screen's contract with the backend (U8: teleop episode collection). */
import { api, post } from './endpoints'

export interface EpisodeSummary {
  index: number
  frames: number
  duration_s: number
  /** dataURL or backend URL per camera, e.g. { top: '/api/record/thumb/0/top' } */
  thumbnails: Record<string, string>
  discarded?: boolean
}

export interface InterruptedSession {
  dataset: string
  task: string
  arms: string[]
  /** seconds since it was opened, or null when the breadcrumb had no timestamp */
  opened_ago: number | null
  /** the whole sentence, composed server-side where the facts are */
  text: string
  /** the two real next actions, as words: nothing here deletes a dataset */
  next: string[]
}

export interface RecordSession {
  /** null when no session is open */
  dataset: string | null
  interrupted?: InterruptedSession | null
  disk_notice?: {
    level: 'tight' | 'critical'
    free_mb: number
    headline: string
    advice: string
  } | null
  task: string
  /** which teleop pair records: leader drives follower */
  leader: string | null
  follower: string | null
  target_episodes: number
  episodes: EpisodeSummary[]
  /** 'idle' = between episodes, 'recording' = frames being captured */
  phase: 'idle' | 'recording'
  /** The rate written into the dataset - a DECLARATION, not a measurement. */
  fps: number
  /** The rate frames were really captured at, or null before two frames exist. */
  fps_achieved?: number | null
  /** Present ONLY when `fps_achieved` differs from `fps` by more than 10%. */
  fps_notice?: {
    declared_fps: number
    measured_fps: number
    /** how many times the timestamps are off, e.g. 5.36 */
    ratio: number
    /** true = captured slower than declared (timestamps squeezed together) */
    slower: boolean
    detail: string
  } | null
  /** Present ONLY when a camera the session asked for never opened. */
  camera_notice?: {
    requested: string[]
    present: string[]
    missing: string[]
    message: string
  } | null
  motion_notice?: {
    still: true
    /** how long the pose has been held, seconds */
    seconds: number
    samples: number
    /** the largest peak-to-peak travel of any joint, in DEGREES */
    max_travel_deg: number
    quietest_joint: string
    message: string
  } | null
}

export interface UploadPreflight {
  /** false = ticking upload can only produce an end-of-session failure */
  ok: boolean
  state: 'ready' | 'no_credential' | 'credential_rejected' | 'foreign_namespace' | 'no_dataset'
    | 'destination_exists'
  /** a refusal that is the operator's call, not a certainty (an org you may belong to; a
   *  published take only they know they want replaced) */
  needs_force: boolean
  user: string | null
  /** the repo id the push would really create, e.g. "me/so101-pick" */
  destination: string | null
  detail: string
}

export interface RecordApi {
  mock: boolean
  session(): Promise<RecordSession>
  /** open a session (creates/appends the dataset) */
  /**
   * The other two camera gates are overridable the same way, and this type has to name them or
   * the form CANNOT send them: `ignore_missing_cameras` (an index this machine does not list at
   * all) and `ignore_camera_identity` (the index is listed and streaming, but a different camera
   * answers it now — an unplug renumbers the rest).
   */
  open(opts: {
    dataset: string; task: string; leader: string; follower: string
    target_episodes: number; fps?: number
    ignore_dead_cameras?: boolean
    ignore_missing_cameras?: boolean
    ignore_camera_identity?: boolean
  }): Promise<RecordSession>
  startEpisode(): Promise<RecordSession>
  /** stop and keep the in-flight episode */
  stopEpisode(): Promise<RecordSession>
  /** stop and immediately discard the in-flight episode (redo) */
  redoEpisode(): Promise<RecordSession>
  discard(index: number): Promise<RecordSession>
  /** close the session; upload=true pushes the dataset to the HF Hub.
   * The receipt carries where the dataset LIVES (dataset/root/episodes_kept) —
   * the record→train handoff is built from it (lib/recordHandoff). */
  close(opts?: { upload?: boolean; repo_id?: string }): Promise<import('./recordHandoff').CloseReceipt>
  uploadPreflight(): Promise<UploadPreflight>
}

/** A FRESH empty session, built per call. */
function emptySession(): RecordSession {
  return {
    dataset: null, task: '', leader: null, follower: null,
    target_episodes: 10, episodes: [], phase: 'idle', fps: 30,
  }
}

/* ------------------------------ mock ------------------------------ */

function makeMock(): RecordApi {
  let s: RecordSession = emptySession()
  let startedAt = 0
  const clone = () => JSON.parse(JSON.stringify(s)) as RecordSession
  return {
    mock: true,
    async session() {
      // tick the in-flight take's frames, as the real control loop would
      if (s.phase === 'recording' && s.episodes.length > 0) {
        const ep = s.episodes[s.episodes.length - 1]
        ep.frames = Math.max(0, Math.round(((Date.now() - startedAt) / 1000) * s.fps))
        ep.duration_s = Math.round(((Date.now() - startedAt) / 1000) * 10) / 10
      }
      return clone()
    },
    async open(opts) {
      // `...opts` would write fps: undefined when the caller omits it, and the mock's frame tick
      // multiplies by it - so the rehearsal would count NaN frames.
      s = { ...emptySession(), ...opts, fps: opts.fps ?? emptySession().fps, episodes: [], phase: 'idle' }
      return clone()
    },
    async startEpisode() {
      if (!s.dataset) throw new Error('no open session')
      if (s.phase === 'recording') return clone()
      s.phase = 'recording'
      startedAt = Date.now()
      // The real backend lists the take in flight as the last episode entry,
      // frames growing as they are captured - mirror that so the UI's live
      // frame tick behaves identically against both.
      s.episodes.push({ index: s.episodes.length, frames: 0, duration_s: 0, thumbnails: {} })
      return clone()
    },
    async stopEpisode() {
      if (s.phase !== 'recording') return clone()
      const duration = (Date.now() - startedAt) / 1000
      const ep = s.episodes[s.episodes.length - 1]
      ep.frames = Math.max(1, Math.round(duration * s.fps))
      ep.duration_s = Math.round(duration * 10) / 10
      s.phase = 'idle'
      return clone()
    },
    async redoEpisode() {
      if (s.phase === 'recording') s.episodes.pop()
      s.phase = 'idle'
      return clone()
    },
    async discard(index) {
      // The real route REFUSES both of these (record_worker.discard: _require_open, then KeyError ->
      // HTTP 404).
      if (!s.dataset) throw new Error('no open session')
      const ep = s.episodes.find(e => e.index === index)
      if (!ep) throw new Error(`no saved episode with index ${index}`)
      ep.discarded = true
      return clone()
    },
    async close() {
      s = emptySession()
      return { ok: true, detail: 'mock session closed (nothing was written)' }
    },
    async uploadPreflight() {
      // The rehearsal has no Hub credential and must not imply one: a mock that says "ready"
      // teaches the operator a green tick this machine never earned.
      return {
        ok: false,
        state: 'no_credential' as const,
        needs_force: false,
        user: null,
        destination: s.dataset,
        detail: 'this is the in-browser rehearsal - nothing is written and nothing can be published',
      }
    },
  }
}

/* ------------------------------ real ------------------------------ */

function makeReal(): RecordApi {
  return {
    mock: false,
    session: () => api<RecordSession>('/api/record/session'),
    open: opts => post<RecordSession>('/api/record/open', opts),
    startEpisode: () => post<RecordSession>('/api/record/episode/start'),
    stopEpisode: () => post<RecordSession>('/api/record/episode/stop'),
    redoEpisode: () => post<RecordSession>('/api/record/episode/redo'),
    discard: index => post<RecordSession>('/api/record/episode/discard', { index }),
    close: opts => post('/api/record/close', opts ?? {}),
    uploadPreflight: () => api<UploadPreflight>('/api/record/upload-preflight'),
  }
}

let cached: Promise<RecordApi> | null = null

/** Probe once per page load: real backend unless the route truly does not exist. */
export function getRecordApi(): Promise<RecordApi> {
  if (!cached) {
    cached = api('/api/record/session')
      .then(() => makeReal())
      .catch(e => (e && (e as { status?: number }).status === 404 ? makeMock() : makeReal()))
  }
  return cached
}
