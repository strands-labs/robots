/**
 * Human names and grouping for the policy selector — UX_REVIEW's #1 ranked finding ("the bare
 * word `mock` beside live telemetry is the single most trust-corrosive string in the app", and
 * `cosmos3` / `curobo` / `wbc_gait` are registry identifiers, not names a person recognises).
 */

export const POLICY_GROUPS = [
  'Safe test',
  'Learned policies (need a checkpoint)',
  'Remote inference (a server does the thinking)',
  'Motion planning (nothing learned)',
  'Humanoid whole-body motion',
  'Other',
] as const

export type PolicyGroup = (typeof POLICY_GROUPS)[number]

interface Known { label: string; group: PolicyGroup }

const KNOWN: Record<string, Known> = {
  mock: { label: 'Mock — sine test (safe, no model, moves gently)', group: 'Safe test' },

  lerobot_local: { label: 'LeRobot — local checkpoint (runs here)', group: 'Learned policies (need a checkpoint)' },
  groot: { label: 'NVIDIA GR00T — N1.5 or N1.6', group: 'Learned policies (need a checkpoint)' },
  cosmos3: { label: 'NVIDIA Cosmos 3 — omnimodal VLA', group: 'Learned policies (need a checkpoint)' },
  vera: { label: 'MIT VERA — video-to-action', group: 'Learned policies (need a checkpoint)' },
  molmoact2: { label: 'MolmoAct 2 — SO-100 / SO-101', group: 'Learned policies (need a checkpoint)' },

  lerobot_async: { label: 'LeRobot — remote server (gRPC)', group: 'Remote inference (a server does the thinking)' },
  remote: { label: 'Remote inference — WebSocket', group: 'Remote inference (a server does the thinking)' },

  curobo: { label: 'NVIDIA cuRobo — collision-aware planner', group: 'Motion planning (nothing learned)' },
  moveit2: { label: 'MoveIt 2 — planner bridge', group: 'Motion planning (nothing learned)' },

  wbc: { label: 'GR00T whole-body control (SONIC)', group: 'Humanoid whole-body motion' },
  wbc_gait: { label: 'GR00T whole-body control — gait clock', group: 'Humanoid whole-body motion' },
  motionbricks: { label: 'NVIDIA MotionBricks — G1 motion', group: 'Humanoid whole-body motion' },
  kimodo: { label: 'NVIDIA Kimodo — G1 text-to-motion', group: 'Humanoid whole-body motion' },
  protomotions: { label: 'ProtoMotions — G1 motion tracker', group: 'Humanoid whole-body motion' },
}

/** The friendly name, or the raw identifier when we have no verified one. */
export function policyLabel(name: string): string {
  return KNOWN[name]?.label ?? name
}

export function isKnownPolicy(name: string): boolean {
  return name in KNOWN
}

export function policyGroup(name: string): PolicyGroup {
  return KNOWN[name]?.group ?? 'Other'
}

export interface GroupedPolicies<T> { group: PolicyGroup; items: T[] }

/**
 * Group in POLICY_GROUPS order, dropping empty groups, preserving the backend's order inside
 * each group. Every input comes back exactly once.
 */
export function groupPolicies<T>(items: readonly T[], nameOf: (item: T) => string): GroupedPolicies<T>[] {
  const out: GroupedPolicies<T>[] = []
  for (const group of POLICY_GROUPS) {
    const inGroup = items.filter(i => policyGroup(nameOf(i)) === group)
    if (inGroup.length) out.push({ group, items: inGroup })
  }
  return out
}
