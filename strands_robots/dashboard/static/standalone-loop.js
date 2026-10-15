/**
 * The standalone control loop, run inside mujoco-worker.js (only when no server is reachable: the browser never
 * steps physics while a server is connected).
 *
 * At hz the loop builds the observation the way strands_robots.policy.onnx.MicroduckPolicy does
 * (base_ang_vel · projected_gravity · joint_pos − default · joint_vel · last_action · command), posts it over a
 * MessageChannel port to policy-worker.js, and on the action reply writes ctrl = default + a · scale and steps one
 * control step (timestep × substeps of mj_step). A skill switch swaps default pose and scale in place, without a
 * reset, so the body carries on from where it is. Fall guard: trunk below fallZ for > 400 ms → reset; 3 falls in
 * 15 s → policy off, and the page is told why. Stats {hz, policyMs, stepMs, z, x, y, yaw, t, falls} once a second.
 */

function quatRotateInverse([w, x, y, z], v) {
  const s = 2 * w * w - 1;
  const c = [y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]];
  const d = x * v[0] + y * v[1] + z * v[2];
  return [v[0] * s - 2 * w * c[0] + 2 * x * d, v[1] * s - 2 * w * c[1] + 2 * y * d, v[2] * s - 2 * w * c[2] + 2 * z * d];
}

export class StandaloneLoop {
  /** env: {mj, model, data, rest(), pose(), post(msg)} — the worker's engine, borrowed for the loop's lifetime. */
  constructor(env, msg) {
    const { mj, model } = env;
    this.env = env;
    const name = (kind, i) => mj.mj_id2name(model, kind.value, i);
    const jointId = (n) => { for (let j = 0; j < model.njnt; j++) if (name(mj.mjtObj.mjOBJ_JOINT, j) === n) return j; return -1; };
    this.joints = msg.joints.map((n) => {
      const j = jointId(n);
      if (j < 0) throw new Error(`standalone: joint ${n} not in the model`);
      let act = -1;
      for (let u = 0; u < model.nu; u++) if (model.actuator_trntype[u] === mj.mjtTrn.mjTRN_JOINT.value && model.actuator_trnid[2 * u] === j) act = u;
      if (act < 0) throw new Error(`standalone: joint ${n} has no actuator`);
      return { qadr: model.jnt_qposadr[j], vadr: model.jnt_dofadr[j], act };
    });
    let root = -1;
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] === mj.mjtJoint.mjJNT_FREE.value && name(mj.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j]) === msg.root) root = j;
    }
    if (root < 0) throw new Error(`standalone: no free joint on root body ${msg.root}`);
    const n = this.joints.length;
    if (6 + 3 * n + msg.commandWidth !== msg.obsWidth) throw new Error(`standalone: obs layout ${6 + 3 * n + msg.commandWidth} != model input ${msg.obsWidth}`);
    Object.assign(this, {
      n, dt: 1 / msg.hz, qRoot: model.jnt_qposadr[root], vRoot: model.jnt_dofadr[root], fallZ: msg.fallZ, port: msg.port,
      lastAction: new Float64Array(n), command: new Float64Array(msg.commandWidth), obs: new Float32Array(msg.obsWidth),
      busy: false, down: 0, falls: [], steps: 0, policyMs: 0, policyN: 0, stepMs: 0, lastStats: performance.now(), z: 0,
    });
    this.skill(msg);
    this.port.onmessage = (e) => {
      try { this.onAction(e.data); } catch (err) { this.env.post({ type: "standalone", error: `standalone loop: ${err?.message || err}` }); this.stop(); }
    };
    env.rest();
    this.settle();
    this.x0 = env.data.qpos[this.qRoot];
    this.y0 = env.data.qpos[this.qRoot + 1];
    this.timer = setInterval(() => this.tick(), 1000 / msg.hz);
  }

  /** New skill: its default pose and action scale; the body is not reset. */
  skill({ defaultPose, actionScale }) {
    if (defaultPose.length !== this.n) throw new Error(`standalone: ${defaultPose.length} default angles for ${this.n} joints`);
    this.defaultPose = Float64Array.from(defaultPose);
    this.scale = actionScale;
    this.lastAction.fill(0);
  }

  setCommand({ vx = 0, vy = 0, wz = 0 }) { this.command[0] = vx; this.command[1] = vy; this.command[2] = wz; }

  settle() { for (let i = 0; i < this.n; i++) this.env.data.ctrl[this.joints[i].act] = this.defaultPose[i]; }

  stop() {
    clearInterval(this.timer);
    try { this.port.onmessage = null; this.port.close(); } catch { /* gone */ }
    this.stopped = true;
  }

  tick() {
    if (this.stopped || this.busy) return; // the policy has not answered the last observation: skip (real time wins)
    const { data } = this.env, o = this.obs, q = data.qpos, v = data.qvel, n = this.n, r = this.qRoot;
    // MuJoCo's free-joint angular velocity is already in the body frame; gravity is projected with the inverse quat.
    o[0] = v[this.vRoot + 3]; o[1] = v[this.vRoot + 4]; o[2] = v[this.vRoot + 5];
    const g = quatRotateInverse([q[r + 3], q[r + 4], q[r + 5], q[r + 6]], [0, 0, -1]);
    o[3] = g[0]; o[4] = g[1]; o[5] = g[2];
    for (let i = 0; i < n; i++) {
      const j = this.joints[i];
      o[6 + i] = q[j.qadr] - this.defaultPose[i];
      o[6 + n + i] = v[j.vadr];
      o[6 + 2 * n + i] = this.lastAction[i];
    }
    for (let i = 0; i < this.command.length; i++) o[6 + 3 * n + i] = this.command[i];
    this.busy = true;
    this.sent = performance.now();
    this.port.postMessage({ obs: o });
  }

  onAction({ action, error }) {
    if (this.stopped) return;
    this.busy = false;
    const { mj, model, data, post } = this.env;
    if (error) { post({ type: "standalone", error }); this.stop(); return; }
    this.policyMs += performance.now() - this.sent; this.policyN++;
    for (let i = 0; i < this.n; i++) {
      this.lastAction[i] = action[i];
      data.ctrl[this.joints[i].act] = this.defaultPose[i] + action[i] * this.scale;
    }
    const t0 = performance.now(), target = data.time + this.dt - 1e-9;
    for (let k = 0; data.time < target && k < 200; k++) mj.mj_step(model, data);
    this.stepMs += performance.now() - t0; this.steps++;
    this.guard();
    if (this.stopped) return;
    this.env.pose();
    this.stats();
  }

  guard() {
    const now = performance.now();
    this.z = this.env.data.qpos[this.qRoot + 2];
    if (this.z < this.fallZ) { if (!this.down) this.down = now; } else this.down = 0;
    if (!this.down || now - this.down <= 400) return;
    this.down = 0;
    this.falls = [...this.falls.filter((t) => now - t < 15000), now];
    if (this.falls.length >= 3) { this.env.post({ type: "standalone", error: "fell 3 times in 15 s — policy off" }); this.stop(); this.env.pose(); return; }
    this.env.rest(); this.lastAction.fill(0); this.settle();
    this.env.post({ type: "standalone", fell: this.falls.length });
  }

  stats() {
    const now = performance.now();
    if (now - this.lastStats < 1000) return;
    const q = this.env.data.qpos, r = this.qRoot, [w, x, y, z] = [q[r + 3], q[r + 4], q[r + 5], q[r + 6]];
    this.env.post({
      type: "standalone",
      stats: {
        steps: this.steps, hz: this.steps / ((now - this.lastStats) / 1000), policyMs: this.policyMs / Math.max(1, this.policyN),
        stepMs: this.stepMs / Math.max(1, this.steps), z: this.z, x: q[r] - this.x0, y: q[r + 1] - this.y0,
        yaw: Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), t: this.env.data.time, falls: this.falls.length,
      },
    });
    Object.assign(this, { steps: 0, policyMs: 0, policyN: 0, stepMs: 0, lastStats: now });
  }
}
