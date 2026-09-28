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
 * value, act} · reset · physics {on} · dispose. Worker -> main: ready · compiled {id,
 * model, pose, pluginsStripped} · pose {xpos, xmat, qpos, time} · error {id?, message}.
 * The model message carries plain typed arrays, not embind handles, so the page never
 * touches the WebAssembly heap.
 */

let mj = null;
let model = null;
let data = null;
const handles = [];
let physicsTimer = null;
let lastTick = 0;
let actuated = []; // [{qadr, act}] for joints driven by a joint transmission actuator

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

function fail(id, err) {
  post({ type: "error", id, message: String(err?.message || err) });
}

function pose() {
  const xpos = Float64Array.from(data.geom_xpos);
  const xmat = Float64Array.from(data.geom_xmat);
  const qpos = Float64Array.from(data.qpos);
  post({ type: "pose", xpos, xmat, qpos, time: data.time }, [xpos.buffer, xmat.buffer, qpos.buffer]);
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
  const G = mj.mjtGeom;
  return {
    ngeom: m.ngeom, nmesh: m.nmesh, njnt: m.njnt, nu: m.nu, nq: m.nq,
    geom_type: Int32Array.from(m.geom_type), geom_size: Float64Array.from(m.geom_size),
    geom_group: Int32Array.from(m.geom_group), geom_dataid: Int32Array.from(m.geom_dataid),
    geom_rgba: Float32Array.from(m.geom_rgba), geom_matid: Int32Array.from(m.geom_matid),
    mat_rgba: Float32Array.from(m.mat_rgba),
    mesh_vertadr: Int32Array.from(m.mesh_vertadr), mesh_vertnum: Int32Array.from(m.mesh_vertnum),
    mesh_faceadr: Int32Array.from(m.mesh_faceadr), mesh_facenum: Int32Array.from(m.mesh_facenum),
    mesh_vert: Float32Array.from(m.mesh_vert), mesh_face: Int32Array.from(m.mesh_face),
    jnt_type: Int32Array.from(m.jnt_type), jnt_qposadr: Int32Array.from(m.jnt_qposadr),
    jnt_range: Float64Array.from(m.jnt_range), jnt_names: names,
    actuator_trntype: Int32Array.from(m.actuator_trntype), actuator_trnid: Int32Array.from(m.actuator_trnid),
    enums: {
      geom: {
        MESH: G.mjGEOM_MESH.value, PLANE: G.mjGEOM_PLANE.value, SPHERE: G.mjGEOM_SPHERE.value,
        CAPSULE: G.mjGEOM_CAPSULE.value, CYLINDER: G.mjGEOM_CYLINDER.value, BOX: G.mjGEOM_BOX.value,
        ELLIPSOID: G.mjGEOM_ELLIPSOID.value,
      },
      HINGE: mj.mjtJoint.mjJNT_HINGE.value, SLIDE: mj.mjtJoint.mjJNT_SLIDE.value,
      TRN_JOINT: mj.mjtTrn.mjTRN_JOINT.value,
    },
  };
}

function compile({ id, sceneXml, files }) {
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
  mj.mj_forward(model, data);
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
  const transfer = Object.values(snap).filter((v) => ArrayBuffer.isView(v)).map((v) => v.buffer);
  post({ type: "compiled", id, model: snap, pluginsStripped }, transfer);
  pose();
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
      case "reset":
        if (!data) return;
        mj.mj_resetData(model, data);
        mj.mj_forward(model, data);
        for (const { qadr, act } of actuated) data.ctrl[act] = data.qpos[qadr];
        pose();
        return;
      case "physics":
        setPhysics(Boolean(msg.on));
        return;
      case "dispose":
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
