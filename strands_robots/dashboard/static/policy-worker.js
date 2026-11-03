/* The policy off the main thread: one ONNX graph at a time on onnxruntime-web (vendored, wasm + simd, one thread).
 * Used only in standalone mode. Protocol (main -> worker): init {ortUrl, wasmDir, port} · load {modelUrl, sidecarUrl}
 * (a skill: swaps the graph; the port keeps answering on the old one until the new one is ready) · dispose.
 * The port is one end of a MessageChannel whose other end is the physics loop: {obs: Float32Array} in,
 * {action: Float32Array} out, so the physics never waits on the page's thread.
 * Worker -> main: inited · ready {obsWidth, actionWidth, meta, loadMs, modelBytes} (after each load) · error {message}.
 * The sidecar JSON carries what onnxruntime-web cannot read from the graph: joint_names, default_joint_pos,
 * action_scale, command_names.
 */
let ort = null, session = null, inputName = "obs", outputName = "actions";

async function fetchOk(url, as) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return as === "json" ? r.json() : r.arrayBuffer();
}

async function load({ modelUrl, sidecarUrl }) {
  const t0 = performance.now();
  const [model, sidecar] = await Promise.all([fetchOk(modelUrl), fetchOk(sidecarUrl, "json")]);
  const next = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const old = session;
  session = next; inputName = next.inputNames[0]; outputName = next.outputNames[0];
  try { await old?.release(); } catch { /* gone */ }
  self.postMessage({
    type: "ready", obsWidth: (sidecar.inputs?.[0]?.shape || []).at(-1) || 0, actionWidth: (sidecar.outputs?.[0]?.shape || []).at(-1) || 0,
    meta: sidecar.metadata || {}, loadMs: performance.now() - t0, modelBytes: model.byteLength,
  });
}

function serve(port) {
  port.onmessage = async (ev) => {
    try {
      const out = await session.run({ [inputName]: new ort.Tensor("float32", ev.data.obs, [1, ev.data.obs.length]) });
      port.postMessage({ action: Float32Array.from(out[outputName].data) });
    } catch (err) {
      port.postMessage({ error: `policy: ${err?.message || err}` });
    }
  };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      const mod = await import(msg.ortUrl);
      ort = mod.default ?? mod;
      ort.env.wasm.wasmPaths = msg.wasmDir;
      ort.env.wasm.numThreads = 1; // no cross-origin isolation on a static site → no SharedArrayBuffer
      ort.env.wasm.proxy = false;
      serve(msg.port);
      self.postMessage({ type: "inited" });
    } else if (msg.type === "load") {
      await load(msg);
    } else if (msg.type === "dispose") {
      try { await session?.release(); } catch { /* gone */ }
      self.close();
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message || err) });
  }
};
