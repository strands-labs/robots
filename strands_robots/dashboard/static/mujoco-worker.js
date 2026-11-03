/**
 * MuJoCo off the main thread.
 *
 * Compiling a model is the expensive step: MuJoCo parses every mesh and computes a
 * convex hull per mesh, which for the SO-101 (18 meshes, 348k faces) is four seconds of
 * one WebAssembly call. Run on the page's thread that call froze every button on the
 * page until the arm appeared. Here the engine lives in a Worker: the page streams the
 * files, hands them over, and keeps responding while the model compiles; afterwards the
 * worker owns the physics and posts geom poses, and the page only renders.
 *
 * Protocol (main -> worker): init {url} · compile {id, sceneXml, files} · setQpos {qadr,
 * value, act} · setQposAll {qpos} · setQposByName {q, poses} · reset · physics {on} · dispose ·
 * standalone {on, hz, root, joints, defaultPose, actionScale, obsWidth, commandWidth, fallZ, port} ·
 * skill {defaultPose, actionScale} · command {vx, vy, wz} (standalone-loop.js; only when no server is reachable).
 * setQposByName is the dashboard's path: joint values keyed by joint NAME and free-base poses
 * keyed by BODY name ([x y z qw qx qy qz]); names resolve to qpos addresses once per compile. Worker -> main: ready ·
 * compiled {id, model, pose, pluginsStripped} · pose {xpos, xmat, qpos, time} · error {id?,
 * message}. The model message carries plain typed arrays, not embind handles, so the page
 * never touches the WebAssembly heap.
 *
 * The rest pose is the model's first keyframe when it has one (a humanoid's "stand"), else
 * qpos0; compile and reset both land there, and the snapshot carries qpos0 so the page can
 * tween from the default pose into the keyframe when the robot first appears.
 */

import { StandaloneLoop } from "./standalone-loop.js";

let mj = null;
let model = null;
let data = null;
const handles = [];
let physicsTimer = null;
let lastTick = 0;
let actuated = []; // [{qadr, act}] for joints driven by a joint transmission actuator
let qadrByJoint = new Map(); // joint name -> qposadr (setQposByName)
let freeByBody = new Map(); // body name -> qposadr of its free joint (setQposByName poses)
let loop = null; // the standalone control loop, when one runs

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const engine = { get mj() { return mj; }, get model() { return model; }, get data() { return data; }, rest: () => rest(), pose: () => pose(), post };

function standaloneStop() { loop?.stop(); loop = null; }

function fail(id, err) {
  post({ type: "error", id, message: String(err?.message || err) });
}

function pose(seq) {
  const xpos = Float64Array.from(data.geom_xpos);
  const xmat = Float64Array.from(data.geom_xmat);
  const qpos = Float64Array.from(data.qpos);
  // seq echoes the setQposByName message this pose answers (undefined for compile/reset poses).
  post({ type: "pose", xpos, xmat, qpos, time: data.time, seq }, [xpos.buffer, xmat.buffer, qpos.buffer]);
}

function stripPlugins(sceneXml, files, vfs) {
  const clean = (xml) => xml
    .replace(/<extension>[\s\S]*?<\/extension>/g, "")
    .replace(/<plugin\b[^>]*\/>/g, "")
    .replace(/<plugin\b[^>]*>[\s\S]*?<\/plugin>/g, "")
    .replace(/<(general|actuator|motor|position|velocity|intvelocity|damper|cylinder|muscle|adhesion)\b[^>]*\bplugin="[^"]*"[^>]*\/>/g, "")
    .replace(/<(general|actuator|motor|position|velocity|intvelocity|damper|cylinder|muscle|adhesion)\b[^>]*\bplugin="[^"]*"[^>]*>[\s\S]*?<\/\1>/g, "");
  const enc = new TextEncoder(), dec = new TextDecoder();
  for (const [path, buf] of files) {
    if (!path.endsWith(".xml")) continue;
    vfs.deleteFile(path);
    vfs.addBuffer(path, enc.encode(clean(dec.decode(buf))));
  }
  return clean(sceneXml);
}

/** Everything the renderer and the joint panel need, copied out of the model once. */
function snapshot() {
  const m = model;
  const names = [];
  for (let j = 0; j < m.njnt; j++) names.push(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT.value, j) || `joint_${j}`);
  const bodyNames = [];
  for (let b = 0; b < m.nbody; b++) bodyNames.push(mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY.value, b) || `body_${b}`);
  const G = mj.mjtGeom;
  return {
    ngeom: m.ngeom, nmesh: m.nmesh, njnt: m.njnt, nu: m.nu, nq: m.nq, nbody: m.nbody,
    geom_bodyid: Int32Array.from(m.geom_bodyid), body_names: bodyNames,
    geom_type: Int32Array.from(m.geom_type), geom_size: Float64Array.from(m.geom_size),
    geom_group: Int32Array.from(m.geom_group), geom_dataid: Int32Array.from(m.geom_dataid),
    geom_rgba: Float32Array.from(m.geom_rgba), geom_matid: Int32Array.from(m.geom_matid),
    mat_rgba: Float32Array.from(m.mat_rgba),
    mesh_vertadr: Int32Array.from(m.mesh_vertadr), mesh_vertnum: Int32Array.from(m.mesh_vertnum),
    mesh_faceadr: Int32Array.from(m.mesh_faceadr), mesh_facenum: Int32Array.from(m.mesh_facenum),
    mesh_vert: Float32Array.from(m.mesh_vert), mesh_face: Int32Array.from(m.mesh_face),
    jnt_type: Int32Array.from(m.jnt_type), jnt_qposadr: Int32Array.from(m.jnt_qposadr),
    jnt_range: Float64Array.from(m.jnt_range), jnt_names: names,
    nkey: m.nkey, qpos0: Float64Array.from(m.qpos0),
    key_qpos: m.nkey > 0 ? Float64Array.from(m.key_qpos).subarray(0, m.nq) : new Float64Array(0),
    actuator_trntype: Int32Array.from(m.actuator_trntype), actuator_trnid: Int32Array.from(m.actuator_trnid),
    enums: {
      geom: {
        MESH: G.mjGEOM_MESH.value, PLANE: G.mjGEOM_PLANE.value, SPHERE: G.mjGEOM_SPHERE.value,
        CAPSULE: G.mjGEOM_CAPSULE.value, CYLINDER: G.mjGEOM_CYLINDER.value, BOX: G.mjGEOM_BOX.value,
        ELLIPSOID: G.mjGEOM_ELLIPSOID.value,
      },
      HINGE: mj.mjtJoint.mjJNT_HINGE.value, SLIDE: mj.mjtJoint.mjJNT_SLIDE.value,
      BALL: mj.mjtJoint.mjJNT_BALL.value, FREE: mj.mjtJoint.mjJNT_FREE.value,
      TRN_JOINT: mj.mjtTrn.mjTRN_JOINT.value,
    },
  };
}

function compile({ id, sceneXml, files }) {
  // Free the previous world first: the WASM heap is finite and every recompile (add/remove a robot) used to leak
  // vfs + model + data — the fourth world in one page died with "Cannot enlarge memory" (gate 8 caught it).
  standaloneStop();
  setPhysics(false);
  for (const h of handles.reverse()) { try { h.delete(); } catch { /* already freed */ } }
  handles.length = 0;
  model = data = null;
  const vfs = new mj.MjVFS();
  handles.push(vfs);
  for (const [path, buf] of files) vfs.addBuffer(path, buf);
  let pluginsStripped = false;
  try {
    model = mj.MjModel.from_xml_string(sceneXml, vfs);
  } catch (err) {
    if (!/plugin/i.test(String(err?.message || err))) throw err;
    // The browser build ships no engine plugins (mujoco.pid, elasticity...). Drop the
    // <extension> block and plugin-driven actuators; the kinematics still render.
    model = mj.MjModel.from_xml_string(stripPlugins(sceneXml, files, vfs), vfs);
    pluginsStripped = true;
  }
  if (!model) throw new Error("MuJoCo returned no model");
  data = new mj.MjData(model);
  handles.push(model, data);
  rest();
  const snap = snapshot();
  actuated = [];
  for (let j = 0; j < model.njnt; j++) {
    for (let u = 0; u < model.nu; u++) {
      if (model.actuator_trntype[u] === snap.enums.TRN_JOINT && model.actuator_trnid[2 * u] === j) {
        actuated.push({ qadr: model.jnt_qposadr[j], act: u });
        break;
      }
    }
  }
  qadrByJoint = new Map();
  freeByBody = new Map();
  for (let j = 0; j < model.njnt; j++) {
    const name = snap.jnt_names[j];
    qadrByJoint.set(name, model.jnt_qposadr[j]);
    if (model.jnt_type[j] === snap.enums.FREE) {
      const body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, model.jnt_bodyid[j]);
      if (body) freeByBody.set(body, model.jnt_qposadr[j]);
    }
  }
  const transfer = Object.values(snap).filter((v) => ArrayBuffer.isView(v)).map((v) => v.buffer);
  post({ type: "compiled", id, model: snap, pluginsStripped }, transfer);
  pose();
}

/** Rest pose: keyframe 0 when the model declares one, else qpos0. Leaves data at mj_forward. */
function rest() {
  if (model.nkey > 0) {
    mj.mj_resetData(model, data);
    for (let i = 0; i < model.nq; i++) data.qpos[i] = model.key_qpos[i];
  } else {
    mj.mj_resetData(model, data);
  }
  mj.mj_forward(model, data);
}

function setPhysics(on) {
  if (physicsTimer) { clearInterval(physicsTimer); physicsTimer = null; }
  if (!on || !data) return;
  for (const { qadr, act } of actuated) data.ctrl[act] = data.qpos[qadr];
  lastTick = performance.now();
  physicsTimer = setInterval(() => {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastTick) / 1000);
    lastTick = now;
    const target = data.time + dt;
    let n = 0;
    while (data.time < target && n++ < 200) mj.mj_step(model, data);
    pose();
  }, 1000 / 60);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "init": {
        const mod = await import(msg.url);
        mj = await (mod.default || mod)();
        post({ type: "ready" });
        return;
      }
      case "compile":
        compile(msg);
        return;
      case "setQpos":
        if (!data) return;
        data.qpos[msg.qadr] = msg.value;
        if (msg.act >= 0) data.ctrl[msg.act] = msg.value;
        if (!physicsTimer) { data.qvel.fill(0); mj.mj_forward(model, data); pose(); }
        return;
      case "setQposAll":
        // A whole pose at once (the wake-up tween): one message per frame instead of one per joint.
        if (!data) return;
        for (let i = 0; i < msg.qpos.length && i < model.nq; i++) data.qpos[i] = msg.qpos[i];
        for (const { qadr, act } of actuated) data.ctrl[act] = data.qpos[qadr];
        if (!physicsTimer) { data.qvel.fill(0); mj.mj_forward(model, data); pose(); }
        return;
      case "setQposByName": {
        // One server frame: {q: {jointName: value}, poses: {bodyName: [x y z qw qx qy qz]}}.
        if (!data) return;
        let missing = 0;
        for (const name in msg.q || {}) {
          const adr = qadrByJoint.get(name);
          if (adr === undefined) { missing++; continue; }
          data.qpos[adr] = msg.q[name];
        }
        for (const body in msg.poses || {}) {
          const adr = freeByBody.get(body);
          const p = msg.poses[body];
          if (adr === undefined || !p || p.length < 7) { missing++; continue; }
          for (let i = 0; i < 7; i++) data.qpos[adr + i] = p[i];
        }
        for (const { qadr, act } of actuated) data.ctrl[act] = data.qpos[qadr];
        data.qvel.fill(0);
        mj.mj_forward(model, data);
        pose(msg.seq);
        if (missing) post({ type: "unresolved", count: missing });
        return;
      }
      case "reset":
        if (!data) return;
        rest();
        for (const { qadr, act } of actuated) data.ctrl[act] = data.qpos[qadr];
        if (loop) { loop.lastAction.fill(0); loop.settle(); }
        pose();
        return;
      case "physics":
        setPhysics(Boolean(msg.on));
        return;
      case "standalone":
        standaloneStop();
        if (!msg.on) return;
        if (!data) throw new Error("standalone: no model compiled");
        setPhysics(false);
        loop = new StandaloneLoop(engine, msg);
        return;
      case "skill":
        loop?.skill(msg);
        return;
      case "command":
        loop?.setCommand(msg);
        return;
      case "dispose":
        standaloneStop();
        setPhysics(false);
        for (const h of handles.reverse()) { try { h.delete(); } catch { /* already freed */ } }
        handles.length = 0;
        model = data = null;
        self.close();
        return;
    }
  } catch (err) {
    fail(msg.id, err);
  }
};
