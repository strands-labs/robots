/* Standalone mode, the page's half: with no server reachable, the twin's worker steps the bundled world and
 * policy-worker.js runs one ONNX skill; this class wires the two over a MessageChannel and switches skills.
 * The bundle (scene.json with a `standalone` block, assets/, policy/<skill>.onnx + .json) is written by
 * `python -m strands_robots.dashboard.standalone`. Nothing here runs while a server is connected.
 */
const ORT_URL = new URL("./vendor/ort.wasm.min.mjs", import.meta.url).href;
const ORT_DIR = new URL("./vendor/", import.meta.url).href;

export class Standalone {
  /** @param twin a Twin that has just loaded the bundle's scene.json from sceneUrl */
  constructor(twin, desc, sceneUrl, { onStats } = {}) {
    if (!desc.standalone) throw new Error("scene has no standalone block");
    this.twin = twin;
    this.sa = desc.standalone;
    this.root = new URL(sceneUrl, location.href);
    this.onStats = onStats || (() => {});
    this.stats = null;
    this.errors = [];
    this.skills = this.sa.skills.map((s) => s.name);
    twin.engine.onStandalone = (m) => {
      if (m.stats) this.stats = m.stats;
      if (m.error) this.errors.push(m.error);
      this.onStats(m);
    };
  }

  _policy(msg, transfer) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => (e.data.type === "error" ? reject(new Error(e.data.message)) : resolve(e.data));
      this.worker.onerror = (e) => reject(new Error(e.message || "policy worker failed"));
      this.worker.postMessage(msg, transfer || []);
    });
  }

  async _load(name) {
    const s = this.sa.skills.find((k) => k.name === name);
    if (!s) throw new Error(`unknown skill ${name}; known: ${this.skills.join(", ")}`);
    const ready = await this._policy({ type: "load", modelUrl: new URL(s.model, this.root).href, sidecarUrl: new URL(s.sidecar, this.root).href });
    const joints = ready.meta.joint_names.split(",").map((n) => `${this.sa.robot}/${n.trim()}`);
    const defaultPose = ready.meta.default_joint_pos.split(",").map(Number);
    const commandWidth = ready.obsWidth - 6 - 3 * joints.length;
    if (commandWidth < 3) throw new Error(`${s.model}: input ${ready.obsWidth} too narrow for ${joints.length} joints and a twist`);
    this.skill = name;
    this.ready = { ...ready, skill: name, joints: joints.length, commandWidth };
    return { joints, defaultPose, actionScale: Number(ready.meta.action_scale || 1), obsWidth: ready.obsWidth, commandWidth };
  }

  /** Start the loop on the bundle's default skill (or `skill`); resolves with what the policy worker reported. */
  async start(skill = this.sa.skill) {
    const channel = new MessageChannel();
    this.worker = new Worker(new URL("./policy-worker.js", import.meta.url), { type: "module" });
    await this._policy({ type: "init", ortUrl: ORT_URL, wasmDir: ORT_DIR, port: channel.port2 }, [channel.port2]);
    const layout = await this._load(skill);
    this.twin.engine.post({ type: "standalone", on: true, hz: this.sa.hz, root: this.sa.root, fallZ: this.sa.fall_z, ...layout, port: channel.port1 }, [channel.port1]);
    return this.ready;
  }

  /** Swap the running skill; the body carries on from where it is. */
  async use(skill) {
    const layout = await this._load(skill);
    this.twin.engine.post({ type: "skill", defaultPose: layout.defaultPose, actionScale: layout.actionScale });
    return this.ready;
  }

  command(vx, vy, wz) { this.twin.engine.post({ type: "command", vx, vy, wz }); }

  stop() {
    this.twin.engine.post({ type: "standalone", on: false });
    try { this.worker?.postMessage({ type: "dispose" }); } catch { /* gone */ }
    this.worker = null;
  }
}
