/* The twin: MuJoCo's compiled geoms, drawn by three.js at the poses MuJoCo computed.
 *
 * No physics runs here and no MJCF is parsed. GET scene -> build one mesh per geom;
 * the telemetry socket (?poses=1) sends a binary frame of [x y z | 3x3 row-major]
 * per geom after every JSON snapshot; we copy it into the objects' matrices. The
 * row width is read from the scene (pose_row_floats), never restated here.
 * MuJoCo is Z-up, three.js is Y-up: the whole world hangs under one group rotated
 * -90deg about X, so every pose is used exactly as sent.
 */
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const HIDDEN_GROUPS = new Set([3]); // MuJoCo convention: 3 = collision geoms

export class Twin {
  /**
   * @param canvas the <canvas> to draw into; its parent decides the size
   * @param sessionId the sim session whose scene and poses this twin follows
   * @param opts {base, headers}: where /api lives (default: this origin) and what every
   *        fetch carries (default: nothing - the session cookie is enough on the same origin)
   */
  constructor(canvas, sessionId, opts = {}) {
    this.canvas = canvas;
    this.sessionId = sessionId;
    this.base = opts.base || "";
    this.headers = opts.headers || {};
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2; // Z-up -> Y-up
    this.scene.add(this.world);
    this.camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 100);
    this.camera.position.set(0.9, 0.7, 0.9);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    this.scene.add(sun);
    const grid = new THREE.GridHelper(4, 40, 0x3a434c, 0x232b32); // rgba() css vars do not parse as Colors
    grid.position.y = 0.0005;
    this.scene.add(grid);
    this.geoms = [];
    this.framed = false;
    this.poseRow = 0; // floats per pose row; the scene publishes it, load() reads it
    this._mat = new THREE.Matrix4();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement);
    this.resize();
    canvas.addEventListener("dblclick", () => { this.framed = false; this.frame(); });
    this._raf = requestAnimationFrame(() => this.tick());
  }

  async load() {
    const desc = await this._get(`/api/sim/${this.sessionId}/scene`).then((r) => r.json());
    // The frame's row width is the server's to state: scene.POSE_ROW_FLOATS is
    // published here so this file never carries a second copy of it.
    if (!Number.isInteger(desc.pose_row_floats) || desc.pose_row_floats <= 0) {
      throw new Error(`scene published no pose_row_floats: ${desc.pose_row_floats}`);
    }
    this.poseRow = desc.pose_row_floats;
    const meshes = new Map();
    await Promise.all(
      desc.meshes.map(async (m) => {
        const buf = await this._get(`/api/sim/${this.sessionId}/${m.url}`).then((r) => r.arrayBuffer());
        meshes.set(m.id, decodeSRM1(buf));
      }),
    );
    for (const g of desc.geoms) {
      const geometry = geometryFor(g, meshes);
      if (!geometry) {
        this.geoms[g.id] = null;
        continue;
      }
      const [r, gg, b, a] = g.rgba;
      // MuJoCo's ground is usually a white checker texture; here the grid is the floor
      // and the plane only catches light, so it takes the page's own dark tone.
      const material = g.type === "plane"
        ? new THREE.MeshStandardMaterial({ color: 0x0b0e11, roughness: 0.9, metalness: 0, side: THREE.DoubleSide })
        : new THREE.MeshStandardMaterial({ color: new THREE.Color(r, gg, b), transparent: a < 1, opacity: a, roughness: 0.55, metalness: 0.1 });
      const obj = new THREE.Mesh(geometry, material);
      obj.matrixAutoUpdate = false;
      obj.visible = !HIDDEN_GROUPS.has(g.group);
      obj.userData = g;
      this.world.add(obj);
      this.geoms[g.id] = obj;
    }
    this.ngeom = desc.ngeom;
    return desc;
  }

  async _get(path) {
    const r = await fetch(`${this.base}${path}`, { headers: this.headers });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r;
  }

  /** One binary telemetry frame: ngeom rows of `pose_row_floats`, as the scene published it. */
  poses(buffer) {
    if (!this.poseRow) return; // no scene yet, so no objects to move and no width to stride by
    const f = new Float32Array(buffer);
    const n = Math.min(this.geoms.length, f.length / this.poseRow);
    const m = this._mat;
    for (let i = 0; i < n; i++) {
      const obj = this.geoms[i];
      if (!obj) continue;
      // Offsets inside the row are the layout `[x y z | 3x3]`, which the width does not describe.
      const o = i * this.poseRow;
      // MuJoCo xmat is row-major R; three's Matrix4.set takes row-major too.
      m.set(
        f[o + 3], f[o + 4], f[o + 5], f[o + 0],
        f[o + 6], f[o + 7], f[o + 8], f[o + 1],
        f[o + 9], f[o + 10], f[o + 11], f[o + 2],
        0, 0, 0, 1,
      );
      obj.matrix.copy(m);
      obj.matrixWorldNeedsUpdate = true;
    }
    if (!this.framed && n) this.frame();
  }

  /** Point the camera at the robot once, from the first poses we get. */
  frame() {
    const box = new THREE.Box3();
    for (const obj of this.geoms) {
      if (!obj || !obj.visible || obj.userData.type === "plane") continue;
      obj.updateMatrixWorld(true);
      box.expandByObject(obj);
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(1, 0.7, 1.2).normalize().multiplyScalar(size * 1.4));
    this.camera.near = size / 100;
    this.camera.far = size * 50;
    this.camera.updateProjectionMatrix();
    this.framed = true;
  }

  toggleGroup(group) {
    if (HIDDEN_GROUPS.has(group)) HIDDEN_GROUPS.delete(group);
    else HIDDEN_GROUPS.add(group);
    for (const obj of this.geoms) if (obj) obj.visible = !HIDDEN_GROUPS.has(obj.userData.group);
  }

  resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth || 320, h = el.clientHeight || 240;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  tick() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._raf = requestAnimationFrame(() => this.tick());
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    for (const obj of this.geoms) if (obj) { obj.geometry.dispose(); obj.material.dispose(); }
    this.renderer.dispose();
  }
}

function decodeSRM1(buf) {
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "SRM1") throw new Error(`bad mesh header ${magic}`);
  const nvert = view.getUint32(4, true), nface = view.getUint32(8, true);
  const verts = new Float32Array(buf, 12, nvert * 3);
  const faces = new Uint32Array(buf, 12 + nvert * 12, nface * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geometry.setIndex(new THREE.BufferAttribute(faces, 1));
  geometry.computeVertexNormals();
  return geometry;
}

/** MuJoCo sizes are half-extents / radii; capsules and cylinders run along local Z. */
function geometryFor(g, meshes) {
  const [a, b, c] = g.size;
  switch (g.type) {
    case "plane":
      return new THREE.PlaneGeometry(a > 0 ? 2 * a : 20, b > 0 ? 2 * b : 20);
    case "sphere":
      return new THREE.SphereGeometry(a, 24, 16);
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 24, 16).scale(a, b, c);
    case "capsule":
      return new THREE.CapsuleGeometry(a, 2 * b, 6, 16).rotateX(Math.PI / 2);
    case "cylinder":
      return new THREE.CylinderGeometry(a, a, 2 * b, 24).rotateX(Math.PI / 2);
    case "mesh": {
      const src = meshes.get(g.mesh);
      return src ? src.clone() : null;
    }
    default:
      return null; // hfield / sdf: not drawn
  }
}
