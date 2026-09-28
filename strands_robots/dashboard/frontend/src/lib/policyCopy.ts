/**
 * Human words for the policy registry's *code identifiers*. The run form is generated from
 * `registry/policies.json`, which names a provider's **constructor kwargs** —
 * `pretrained_name_or_path`, `policy_type`, `data_config`.
 */

export interface FieldCopy {
  /** What to lead with. Equals `key` when nothing is curated for it. */
  label: string
  /** One short clause: what the operator has to *have* to fill this in. */
  hint?: string
  /** false => `label` IS the raw identifier; do not present it as English. */
  known: boolean
}

/** Curated only where we can state the truth in a few words. */
const COPY: Record<string, { label: string; hint?: string }> = {
  // --- checkpoint / model identity -----------------------------------------
  pretrained_name_or_path: {
    label: 'checkpoint',
    hint: 'a Hugging Face repo id or a local training output directory',
  },
  model_path: { label: 'checkpoint path', hint: 'a directory on the robot itself' },
  checkpoint: { label: 'checkpoint', hint: 'a trained policy to load' },
  repo_id: { label: 'checkpoint', hint: 'a Hugging Face repo id' },
  policy_type: {
    label: 'policy family',
    hint: 'which architecture the checkpoint is (act, smolvla, diffusion…)',
  },
  model: { label: 'model' },
  model_id: { label: 'model id' },
  onnx_path: { label: 'ONNX file', hint: 'an exported model file on the robot' },

  // --- inference server ----------------------------------------------------
  port: { label: 'server port', hint: 'the inference server must already be running' },
  policy_port: { label: 'server port', hint: 'the inference server must already be running' },
  host: { label: 'server host', hint: 'where that inference server listens' },
  policy_host: { label: 'server host', hint: 'where that inference server listens' },
  server_address: { label: 'server address', hint: 'host:port of a running inference server' },
  server_port: { label: 'server port' },
  endpoint: { label: 'endpoint URL' },
  api_key: { label: 'API key', hint: 'a credential — set it in Settings › Env, not here' },
  api_token: { label: 'API token', hint: 'a credential — set it in Settings › Env, not here' },
  connect_timeout: { label: 'connect timeout (s)' },
  request_timeout: { label: 'request timeout (s)' },
  timeout_ms: { label: 'timeout (ms)' },
  auto_launch_server: { label: 'launch the server for me' },
  server_mode: { label: 'server mode' },

  // --- motion / control ----------------------------------------------------
  action_horizon: { label: 'action horizon', hint: 'how many future steps each inference returns' },
  actions_per_chunk: { label: 'actions per chunk' },
  actions_per_step: { label: 'actions per step' },
  control_frequency: { label: 'control rate (Hz)' },
  n_steps: { label: 'steps' },
  fast_mode: { label: 'fast mode' },
  target_pose: { label: 'target pose', hint: 'a cartesian goal, as JSON' },
  target_joints: { label: 'target joints', hint: 'goal joint positions, as JSON' },
  robot_name: { label: 'robot name', hint: 'which robot on that peer' },
  robot: { label: 'robot' },
  robot_config: { label: 'robot config' },

  // --- data plumbing -------------------------------------------------------
  data_config: { label: 'data config', hint: "the server's name for this robot's I/O layout" },
  image_keys: { label: 'camera keys', hint: 'which camera streams the policy expects' },
  observation_mapping: { label: 'observation mapping' },
  action_mapping: { label: 'action mapping' },
  action_space: { label: 'action space' },
  strict_keys: { label: 'strict key matching' },
  device: { label: 'compute device', hint: 'cpu, cuda or mps on the machine that runs the policy' },
  dtype: { label: 'numeric precision' },
  cache_dir: { label: 'cache directory' },
  trust_remote_code: {
    label: 'allow the checkpoint to run its own code',
    hint: 'a consent decision — the dashboard asks before it is set',
  },
  prompt: { label: 'prompt' },
  text_prompt: { label: 'prompt' },
  seed: { label: 'random seed' },
  world_config: { label: 'world config' },
  world_update: { label: 'world update', hint: 'obstacles/scene, as JSON' },
}

/** Human words for one registry/wire key. Never invents a meaning. */
export function fieldCopy(key: string): FieldCopy {
  const hit = COPY[key]
  if (!hit) return { label: key, known: false }
  return { label: hit.label, hint: hit.hint, known: true }
}

/**
 * `["policy_type","pretrained_name_or_path"]` -> "needs a checkpoint + policy family", for a
 * one-line `<option>` where the identifiers do not fit.
 */
export function requirementSummary(keys: readonly string[]): string {
  const labels: string[] = []
  for (const k of keys) {
    const label = fieldCopy(k).label
    if (!labels.includes(label)) labels.push(label)
  }
  return labels.join(' + ')
}

/**
 * The blocking-fields sentence: leads with words, keeps the identifiers in parentheses because
 * that is the string the operator will search the docs and their own script for.
 */
export function missingSummary(keys: readonly string[]): string {
  return keys
    .map(k => {
      const c = fieldCopy(k)
      return c.known ? `${c.label} (${k})` : k
    })
    .join(', ')
}

/**
 * The "local-only kwargs" list: same label+identifier shape, but this list is informational
 * (these fields cannot be sent at all), so it stays compact.
 */
export function localOnlySummary(keys: readonly string[]): string {
  return keys.map(k => {
    const c = fieldCopy(k)
    return c.known ? `${c.label} (${k})` : k
  }).join(', ')
}
