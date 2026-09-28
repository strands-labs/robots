/**
 * The content of the help sheet — JOURNEYS #7 ("zero onboarding, zero hyperlinks, zero help
 * affordance": 0 `<a>` on the page, docs that exist and are linked from nowhere).
 */

export interface HelpTopic {
  /** Short heading, safe to render as-is. */
  title: string
  /** One paragraph per line. Plain sentences — no markdown rendering here. */
  lines: string[]
}

export interface DocLink {
  label: string
  url: string
  /** Why this one is worth the click. */
  note: string
}

/** The published docs site root — the only host any help link may point at. */
export const DOCS_ORIGIN = 'https://strands-labs.github.io/robots'

export const DOC_LINKS: readonly DocLink[] = [
  {
    label: 'Strands Robots docs',
    url: `${DOCS_ORIGIN}/`,
    note: 'the whole library: robots, mesh, policies, training',
  },
  {
    label: 'Quickstart',
    url: `${DOCS_ORIGIN}/getting-started/quickstart/`,
    note: 'install, then a robot in three lines of Python',
  },
]

/** Pages that exist in this repo but are NOT on the deployed site yet. */
export const REPO_DOC_PATHS: readonly string[] = [
  'docs/dashboard/quickstart.md — this screen, with an SO-101',
  'docs/dashboard/collect-train-deploy.md — the full loop',
  'docs/dashboard/troubleshooting.md — when a camera or an arm stays dark',
  'docs/dashboard/remote-access.md — reaching this dashboard from outside',
]

export const HELP_TOPICS: readonly HelpTopic[] = [
  {
    title: 'What this page is',
    lines: [
      'A live console for every robot on your mesh: each card is one peer, its joint strips are real positions arriving over the network, and any command you send goes to real hardware unless that peer is a simulation.',
      'The dashboard does not own the robots. It joins the same mesh they do, so a peer can appear or vanish without anything here being broken.',
    ],
  },
  {
    title: 'Stopping things — read this first',
    lines: [
      'STOP ALL is in the top-right corner of every screen and is the first thing the Tab key reaches. The "." key opens it from anywhere, including over an open drawer, and Cmd+. (Ctrl+. on Windows and Linux) works even while you are typing in a field.',
      'It broadcasts an emergency stop and then LOCKS OUT commands until you resume; it does not power anything down.',
      'If this page cannot reach the server it says so and marks the button degraded — the arm\'s own power switch is then the only brake that does not go through this page.',
    ],
  },
  {
    title: 'A safe first action',
    lines: [
      'Open a robot card, pick the "mock" policy (a built-in sine test that needs no model), type any task sentence and press ▶ on a SIM peer. Nothing physical moves.',
      'On a real arm, ▶ asks for confirmation first and names the robot it is about to move. That confirmation is the last step before metal moves.',
    ],
  },
  {
    title: 'Collect → train → deploy',
    lines: [
      '1. ⚙ devices — see the USB arms and cameras this machine can find, name them, and spawn one as a peer.',
      '2. ⏺ record — teleoperate a leader arm and capture episodes into a LeRobot dataset.',
      '3. 🎓 train — point a trainer at that dataset (local folder or a Hugging Face repo) and watch the loss.',
      '4. Deploy the checkpoint from the training screen: it prefills the run form on a robot card, and you press ▶.',
    ],
  },
  {
    title: 'When something looks wrong',
    lines: [
      'A card greys out when its peer stops announcing itself — that is the mesh going quiet, not a crash.',
      'A camera tile that never shows a frame is usually an OS permission: on macOS a dashboard started by a background daemon can never be granted camera access, and no prompt will appear.',
      'The ☰ activity log records every command this dashboard sent, with what came back.',
    ],
  },
]
