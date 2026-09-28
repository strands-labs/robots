/** The header's connection badge. */
import type { ConnState } from './useMesh'

/**
 * The header's connection badge — UX_REVIEW #3 ("never let LIVE sit above 'connecting'"). The
 * badge only ever knew about ONE thing: this browser's websocket to the dashboard.
 */

export interface ConnBadge {
  /** What is printed. */
  label: string
  /** Tone class suffix: '' (neutral/good), 'warn', or 'bad'. */
  tone: '' | 'warn' | 'bad'
  /** Hover text — the full sentence, including what this does NOT cover. */
  title: string
  /** Screen-reader label; always names the subject. */
  aria: string
}

const SCOPE_NOTE =
  'This is the link between this page and the dashboard. It says nothing about the robots or the cameras — each tile reports its own state.'

export function connBadge(conn: ConnState, opts: { meshDown?: boolean } = {}): ConnBadge {
  const meshDown = !!opts.meshDown

  switch (conn) {
    case 'open':
      return meshDown
        ? {
            // The socket IS open, so OFFLINE would be a lie. Narrow the claim.
            label: 'LIVE · page only',
            tone: 'warn',
            title: `This page is connected, but the dashboard's own mesh session is closed, so robot telemetry and commands are not flowing. ${SCOPE_NOTE}`,
            aria: 'Dashboard link live, but the robot mesh session is down',
          }
        : {
            label: 'LIVE',
            tone: '',
            title: `Connected to the dashboard. ${SCOPE_NOTE}`,
            aria: 'Dashboard link: live',
          }
    case 'connecting':
      return {
        label: 'CONNECTING',
        tone: 'warn',
        title: `Opening the connection to the dashboard. ${SCOPE_NOTE}`,
        aria: 'Dashboard link: connecting',
      }
    case 'closed':
      return {
        label: 'OFFLINE',
        tone: 'bad',
        title: `No connection to the dashboard: nothing on this page is updating, and buttons will not reach the robots. ${SCOPE_NOTE}`,
        aria: 'Dashboard link: offline — nothing on this page is updating',
      }
    case 'unauthorized':
      return {
        label: 'NO ACCESS',
        tone: 'bad',
        title: 'The server rejected this token — set it in Settings. Nothing on this page is updating.',
        aria: 'Dashboard link: not authorised — the server rejected this token',
      }
    default: {
      // An unknown state must not be dressed as LIVE.
      const unknown: string = conn
      return {
        label: String(unknown).toUpperCase(),
        tone: 'warn',
        title: `Unrecognised connection state "${unknown}". ${SCOPE_NOTE}`,
        aria: `Dashboard link: ${unknown}`,
      }
    }
  }
}
