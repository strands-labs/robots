/* The twin: the server's merged MJCF, compiled by MuJoCo WASM in a Worker for kinematics only,
 * drawn by three.js at the geom poses mj_forward computes.
 *
 * load(sceneUrl) fetches {xml, assets:[{path,url}]} (GET /api/scene, or a static scene.json),
 * streams every asset into the worker's VFS, compiles, and builds one three.js mesh per geom
 * from the model snapshot the worker posts (plain typed arrays, no WASM heap on this thread).
 * setQposByName({q, poses}) is the 30 Hz path: joint values keyed by name, free-base poses by
 * body name. No physics runs here — the server's Simulation is the truth.
 * MuJoCo is Z-up, three.js is Y-up: the world hangs under one group rotated -90deg about X.
 */
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const HIDDEN_GROUPS = new Set([3]); // MuJoCo convention: 3 = collision geoms
const MUJOCO_URL = new URL("./vendor/mujoco.js", import.meta.url).href;

class Engine {
  constructor(onPose, onInfo) {
    this._worker = new Worker(new URL("./mujoco-worker.js", import.meta.url), { type: "module" });
    this._pending = new Map();
    this._onPose = onPose;
    this._onInfo = onInfo || (() => {});
    this._ready = new Promise((resolve, reject) => this._pending.set("init", { resolve, reject }));
    this._worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "pose") { this._onPose(msg); return; }
      if (msg.type === "ready") { this._settle("init", "resolve"); return; }
      if (msg.type === "compiled") { this._settle(msg.id, "resolve", msg); return; }
      if (msg.type === "unresolved") { this._onInfo(msg); return; }
      if (msg.type === "error") {
        if (this._pending.has(msg.id ?? "init")) this._settle(msg.id ?? "init", "reject", new Error(msg.message));
        else console.error("mujoco worker:", msg.message);
      }
    };
    this._worker.onerror = (e) => {
      console.error("mujoco worker:", e.message || e);
      for (const p of this._pending.values()) p.reject(new Error(e.message || "worker failed"));
      this._pending.clear();
    };
    this._worker.postMessage({ type: "init", url: MUJOCO_URL });
  }
  _settle(id, how, value) { const p = this._pending.get(id); if (p) { p[how](value); this._pending.delete(id); } }
  ready() { return this._ready; }
  compile(sceneXml, files) {
    const id = "c" + Math.random().toString(36).slice(2);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({ type: "compile", id, sceneXml, files }, files.map(([, b]) => b.buffer));
    });
  }
  setQposByName(frame, seq) { this._worker.postMessage({ type: "setQposByName", q: frame.q, poses: frame.poses, seq }); }
  reset() { this._worker.postMessage({ type: "reset" }); }
  dispose() { try { this._worker.postMessage({ type: "dispose" }); } catch { /* gone */ } this._worker.terminate(); }
}

export class Twin {
  /** @param canvas the <canvas> to draw into; its parent decides the size */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.base = opts.base || "";
    this.onPick = opts.onPick || (() => {});
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2; // Z-up -> Y-up
    this.scene.add(this.world);
    this.camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 100);
    this.camera.position.set(1.5, 1.2, 1.8);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    this.scene.add(sun);
    this.grid = new THREE.GridHelper(6, 60, 0x46505c, 0x303740);
    this.grid.position.y = 0.0005;
    this.scene.add(this.grid);
    this.palette = { plane: 0x111213, major: 0x46505c, minor: 0x303740 }; // strands --sr-viewer-bg / --sr-grid-* (dark)
    this.geoms = [];
    this.model = null; // snapshot from the worker
    this.pose = null; // last {xpos, xmat, qpos, time}
    this.revision = null;
    this.frames = 0; // setQposByName calls applied (measured by the status line)
    this.unresolved = 0;
    this.framed = false;
    this._mat = new THREE.Matrix4();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement);
    this.resize();
    canvas.addEventListener("dblclick", () => { this.framed = false; this.frame(); });
    canvas.addEventListener("click", (e) => this._pick(e));
    this._raycaster = new THREE.Raycaster();
    this._raf = requestAnimationFrame(() => this.tick());
    this.engine = new Engine((p) => this._onPose(p), (m) => { this.unresolved += m.count; });
  }

  /** Fetch the scene description, stream its assets, compile, build the meshes. Returns the description. */
  async load(sceneUrl) {
    const desc = await this._get(sceneUrl).then((r) => r.json());
    this._sceneUrl = sceneUrl;
    const root = new URL(sceneUrl, location.href);
    const xml = desc.xml.trim().startsWith("<") ? desc.xml : await this._get(new URL(desc.xml, root).href).then((r) => r.text());
    const files = await Promise.all(
      (desc.assets || []).map(async (a) => [a.path, new Uint8Array(await this._get(new URL(a.url, root).href).then((r) => r.arrayBuffer()))]),
    );
    this.assetBytes = files.reduce((n, [, b]) => n + b.byteLength, 0);
    await this.engine.ready();
    this._clear();
    const t0 = performance.now();
    const { model, pluginsStripped } = await this.engine.compile(xml, files);
    this.compileMs = performance.now() - t0;
    this.pluginsStripped = pluginsStripped;
    this.model = model;
    this.revision = desc.revision ?? null;
    this.desc = desc;
    this._build();
    // A keyframe inside an attached robot is NOT rebased to its frame, so the compile-time rest pose can
    // put a humanoid at the origin; re-frame once the first server frame places everything for real.
    this._reframeOnPose = true;
    return desc;
  }

  /** Stage colours follow the strands tokens: dark = --sr-viewer-bg #111213, paper = #f4f4f4 with the paper grid. */
  setScheme(scheme) {
    this.palette = scheme === "paper" ? { plane: 0xf4f4f4, major: 0xb6b6b6, minor: 0xd9d9d9 } : { plane: 0x111213, major: 0x46505c, minor: 0x303740 };
    const colors = this.grid.geometry.attributes.color, maj = new THREE.Color(this.palette.major), min = new THREE.Color(this.palette.minor);
    // GridHelper: 4 vertices per division line; the centre line (30 of 60) wears the major colour.
    for (let i = 0; i < colors.count; i++) { const c = (i >> 2) === 30 ? maj : min; colors.setXYZ(i, c.r, c.g, c.b); }
    colors.needsUpdate = true;
    for (const o of this.geoms) if (o && o.userData.plane) o.material.color.setHex(this.palette.plane);
    this.scheme = scheme;
  }

  _get(url) {
    return fetch(`${this.base}${url}`).then((r) => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r; });
  }

  _clear() {
    for (const obj of this.geoms) if (obj) { this.world.remove(obj); obj.geometry.dispose(); obj.material.dispose(); }
    this.geoms = [];
    this.pose = null;
  }

  /** One three.js mesh per geom, from the snapshot's plain arrays. */
  _build() {
    const m = this.model, G = m.enums.geom;
    const meshCache = new Map();
    for (let g = 0; g < m.ngeom; g++) {
      const type = m.geom_type[g];
      const size = [m.geom_size[3 * g], m.geom_size[3 * g + 1], m.geom_size[3 * g + 2]];
      let geometry;
      if (type === G.MESH) {
        const id = m.geom_dataid[g];
        geometry = meshCache.get(id) ?? this._meshGeometry(id);
        meshCache.set(id, geometry);
      } else if (type === G.PLANE) geometry = new THREE.PlaneGeometry(size[0] ? 2 * size[0] : 40, size[1] ? 2 * size[1] : 40);
      else if (type === G.SPHERE) geometry = new THREE.SphereGeometry(size[0], 24, 16);
      else if (type === G.CAPSULE) geometry = new THREE.CapsuleGeometry(size[0], 2 * size[1], 6, 16).rotateX(Math.PI / 2);
      else if (type === G.CYLINDER) geometry = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 24).rotateX(Math.PI / 2);
      else if (type === G.BOX) geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
      else if (type === G.ELLIPSOID) geometry = new THREE.SphereGeometry(1, 24, 16).scale(size[0], size[1], size[2]);
      else { this.geoms[g] = null; continue; } // hfield / sdf: not drawn
      let rgba = [m.geom_rgba[4 * g], m.geom_rgba[4 * g + 1], m.geom_rgba[4 * g + 2], m.geom_rgba[4 * g + 3]];
      const matid = m.geom_matid[g];
      if (matid >= 0) rgba = [m.mat_rgba[4 * matid], m.mat_rgba[4 * matid + 1], m.mat_rgba[4 * matid + 2], m.mat_rgba[4 * matid + 3]];
      const isPlane = type === G.PLANE;
      const material = isPlane
        ? new THREE.MeshStandardMaterial({ color: this.palette.plane, roughness: 0.9, metalness: 0, side: THREE.DoubleSide })
        : new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), transparent: rgba[3] < 1, opacity: rgba[3], roughness: 0.55, metalness: 0.1 });
      const obj = new THREE.Mesh(geometry, material);
      obj.matrixAutoUpdate = false;
      const group = m.geom_group[g];
      obj.visible = !HIDDEN_GROUPS.has(group);
      obj.userData = { geom: g, group, plane: isPlane, body: m.geom_bodyid ? m.geom_bodyid[g] : -1 };
      this.world.add(obj);
      this.geoms[g] = obj;
    }
    this.framed = false;
    if (this.pose) this._syncPoses();
  }

  _meshGeometry(id) {
    const m = this.model;
    const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id];
    const fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.mesh_vert.subarray(3 * va, 3 * (va + vn))), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))), 1));
    geo.computeVertexNormals();
    return geo;
  }

  /** The 30 Hz path: {q: {jointName: value}, poses: {bodyName: [x y z qw qx qy qz]}}. */
  setQposByName(frame) {
    if (!this.model) return;
    this.frames++;
    this.engine.setQposByName(frame, this.frames);
  }

  _onPose(p) {
    this.pose = p;
    if (this._reframeOnPose && p.seq !== undefined) { this._reframeOnPose = false; this.framed = false; }
    if (this.geoms.length) this._syncPoses();
    // follow: keep a walking robot in frame by moving target + camera with its root body (MuJoCo xyz → three x z -y)
    if (this.follow && this.framed) {
      const b = this.bodyPosition(this.follow);
      if (b) {
        const here = new THREE.Vector3(b[0], b[2], -b[1]);
        if (this._followLast) { const d = here.clone().sub(this._followLast); d.y = 0; this.controls.target.add(d); this.camera.position.add(d); }
        this._followLast = here;
      }
    }
  }

  _syncPoses() {
    const { xpos, xmat } = this.pose;
    for (const obj of this.geoms) {
      if (!obj) continue;
      const g = obj.userData.geom, p = 3 * g, r = 9 * g;
      obj.matrix.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1,
      );
      obj.matrixWorldNeedsUpdate = true;
    }
    if (!this.framed) this.frame();
  }

  /** qpos of a joint by name, read from the last worker pose (what the Playwright gate reads). */
  qpos(jointName) {
    if (!this.model || !this.pose) return null;
    const j = this.model.jnt_names.indexOf(jointName);
    return j < 0 ? null : this.pose.qpos[this.model.jnt_qposadr[j]];
  }

  /** World position of the first geom on the first body (in model order) whose name starts with prefix. */
  bodyPosition(prefix) {
    if (!this.model || !this.pose || !this.model.body_names) return null;
    for (let bid = 0; bid < this.model.body_names.length; bid++) {
      if (!this.model.body_names[bid].startsWith(prefix)) continue;
      const g = this.model.geom_bodyid.indexOf(bid);
      if (g >= 0) return [this.pose.xpos[3 * g], this.pose.xpos[3 * g + 1], this.pose.xpos[3 * g + 2]];
    }
    return null;
  }

  /** Point the camera at everything once. */
  frame() {
    const box = new THREE.Box3();
    for (const obj of this.geoms) {
      if (!obj || !obj.visible || obj.userData.plane) continue;
      obj.updateMatrixWorld(true);
      box.expandByObject(obj);
    }
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const center = sphere.center, radius = sphere.radius || 0.5;
    // Distance that fits the sphere in the narrower of the vertical / horizontal fov, plus margin.
    const vfov = THREE.MathUtils.degToRad(this.camera.fov);
    const visible = Math.max(0.2, 1 - (this.insetRight || 0) / Math.max(1, this.canvas.clientWidth)); // share not under the drawer
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect * visible);
    const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.15;
    this.controls.target.copy(center);
    // Look from MuJoCo -y (three +z), slightly above: robots placed along x spread left→right instead of
    // hiding behind each other, and a small arm next to a humanoid stays visible.
    this.camera.position.copy(center).add(new THREE.Vector3(0.25, 0.55, 1).normalize().multiplyScalar(dist));
    this.camera.near = Math.max(0.01, dist / 100);
    this.camera.far = dist * 50;
    this.camera.updateProjectionMatrix();
    this.framed = true;
  }

  _pick(e) {
    if (!this.model || !this.model.body_names) return;
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this._raycaster.setFromCamera(ndc, this.camera);
    const hit = this._raycaster.intersectObjects(this.geoms.filter((o) => o && o.visible && !o.userData.plane), false)[0];
    if (!hit) return;
    const body = this.model.body_names[hit.object.userData.body] || "";
    this.onPick(body.split("/")[0], body);
  }

  toggleGroup(group) {
    if (HIDDEN_GROUPS.has(group)) HIDDEN_GROUPS.delete(group); else HIDDEN_GROUPS.add(group);
    for (const obj of this.geoms) if (obj) obj.visible = !HIDDEN_GROUPS.has(obj.userData.group);
  }

  resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth || 320, h = el.clientHeight || 240;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // A drawer covers the right `insetRight` px of the canvas: shift the view so the scene centres in what is visible.
    if (this.insetRight > 0) this.camera.setViewOffset(w, h, this.insetRight / 2, 0, w, h); else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }

  /** Tell the twin how many pixels on the right are covered by UI (0 = nothing); re-frames if it changes. */
  setInset(rightPx) {
    const px = Math.max(0, Math.round(rightPx || 0));
    if (px === (this.insetRight || 0)) return;
    this.insetRight = px;
    this.resize();
    this.framed = false; this.frame();
  }

  tick() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    // render fps, measured over the last second (the status chip and the gates read twin.fps)
    const now = performance.now();
    this._fpsN = (this._fpsN || 0) + 1;
    if (!this._fpsT) this._fpsT = now;
    if (now - this._fpsT >= 1000) {
      this.fps = Math.round((this._fpsN * 1000) / (now - this._fpsT)); this._fpsN = 0; this._fpsT = now;
      // a weak GPU (or a software one) gets fewer pixels, not a frozen page: 3 slow seconds → one step down, floor 0.5
      this._slow = this.fps < 24 ? (this._slow || 0) + 1 : 0;
      const pr = this.renderer.getPixelRatio();
      if (this._slow >= 3 && pr > 0.5) { this.renderer.setPixelRatio(Math.max(0.5, pr - 0.25)); this.resize(); this._slow = 0; this.pixelRatioDrops = (this.pixelRatioDrops || 0) + 1; }
    }
    this._raf = requestAnimationFrame(() => this.tick());
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this._clear();
    this.engine.dispose();
    this.renderer.dispose();
  }
}
