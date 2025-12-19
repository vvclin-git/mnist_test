const CONFIG = {
  canvasSize: 280,
  downscaleSize: 28,
  brushSize: 20,
  colors: {
    bg: "#ffffff",
    ink: "#0f172a",
  },
  modelUrl: "./web_graph_model_CNN_48_80/model.json",
  autoPredictDelayMs: 1200,
  debug: {
    showPanel: true,
    showPreview: true,
    logTensor: false,
  },
};

const state = {
  model: null,
  isPointerDown: false,
  activePointerId: null,
  idleTimer: null,
  isPredicting: false,
  hasDrawing: false,
  hasPredicted: false,
  lastEvent: "-",
  lastInputStats: null,
  lastPredictTop3: null,
  lastPredictLatencyMs: null,
  lastInvert: null,
  lastSmoothingEnabled: null,
};

function getElements() {
  return {
    app: document.getElementById("app"),
    canvas: document.getElementById("canvas"),
    preview: document.getElementById("preview"),
    clearBtn: document.getElementById("clearBtn"),
    result: document.getElementById("result"),
    statusHint: document.getElementById("statusHint"),
    debugPanel: document.getElementById("debugPanel"),
  };
}

function setDebugValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function formatNumber(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatMillis(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "-";
}

function updateDebugStatic() {
  setDebugValue("dbgLocation", location.href);
  setDebugValue("dbgUserAgent", navigator.userAgent);
  setDebugValue("dbgPlatform", navigator.platform || "-");
  setDebugValue("dbgDpr", formatNumber(window.devicePixelRatio, 2));

  if (typeof tf !== "undefined") {
    setDebugValue("dbgTfVersion", tf.version?.tfjs || "-");
    setDebugValue("dbgBackend", tf.getBackend ? tf.getBackend() : "-");
    try {
      const env = tf.env();
      setDebugValue("dbgWebglVersion", env.get("WEBGL_VERSION") ?? "-");
      setDebugValue("dbgWebglFloat32Capable", String(env.get("WEBGL_RENDER_FLOAT32_CAPABLE")));
      setDebugValue("dbgWebglFloat32Enabled", String(env.get("WEBGL_RENDER_FLOAT32_ENABLED")));
    } catch (err) {
      setDebugValue("dbgWebglVersion", "-");
      setDebugValue("dbgWebglFloat32Capable", "-");
      setDebugValue("dbgWebglFloat32Enabled", "-");
    }
  }
}

function updateDebugCanvas(canvas) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  setDebugValue("dbgCanvasSize", `${canvas.width} x ${canvas.height}`);
  setDebugValue("dbgCanvasRect", `${rect.width.toFixed(1)} x ${rect.height.toFixed(1)}`);
  setDebugValue("dbgScaleX", formatNumber(scaleX, 4));
  setDebugValue("dbgScaleY", formatNumber(scaleY, 4));
  setDebugValue("dbgTouchAction", canvas.style.touchAction || "-");
  setDebugValue("dbgSmoothing", state.lastSmoothingEnabled == null ? "-" : String(state.lastSmoothingEnabled));
}

function updateDebugInput(stats, invert) {
  setDebugValue("dbgInvert", invert == null ? "-" : String(invert));
  if (!stats) {
    setDebugValue("dbgMin", "-");
    setDebugValue("dbgMax", "-");
    setDebugValue("dbgMean", "-");
    setDebugValue("dbgStd", "-");
    setDebugValue("dbgNonZeroRatio", "-");
    return;
  }
  setDebugValue("dbgMin", formatNumber(stats.min, 4));
  setDebugValue("dbgMax", formatNumber(stats.max, 4));
  setDebugValue("dbgMean", formatNumber(stats.mean, 4));
  setDebugValue("dbgStd", formatNumber(stats.std, 4));
  setDebugValue("dbgNonZeroRatio", formatNumber(stats.nonZeroRatio, 4));
}

function updateDebugPrediction(bestIdx, bestProb, top3, latencyMs) {
  setDebugValue("dbgPredClass", bestIdx == null ? "-" : String(bestIdx));
  setDebugValue("dbgConfidence", bestProb == null ? "-" : formatNumber(bestProb, 4));
  if (top3 && top3.length) {
    setDebugValue("dbgTop3", top3.map(item => `${item.idx}:${item.prob.toFixed(3)}`).join(", "));
  } else {
    setDebugValue("dbgTop3", "-");
  }
  setDebugValue("dbgLatency", formatMillis(latencyMs));
}

function updateDebugEvents() {
  setDebugValue("dbgPointerDown", String(state.isPointerDown));
  setDebugValue("dbgIdleActive", String(state.idleTimer != null));
  setDebugValue("dbgHasDrawing", String(state.hasDrawing));
  setDebugValue("dbgHasPredicted", String(state.hasPredicted));
  setDebugValue("dbgLastEvent", state.lastEvent || "-");
}

function getTopK(probs, k = 3) {
  const items = Array.from(probs, (prob, idx) => ({ idx, prob }));
  items.sort((a, b) => b.prob - a.prob);
  return items.slice(0, k);
}

async function loadModel() {
  state.model = await tf.loadGraphModel(CONFIG.modelUrl);
  console.log("GraphModel loaded");
  console.log("inputs:", state.model.inputs.map(x => x.name));
  console.log("outputs:", state.model.outputs.map(x => x.name));
  updateDebugStatic();
}

function primeCanvas(canvas, ctx) {
  canvas.width = CONFIG.canvasSize;
  canvas.height = CONFIG.canvasSize;
  ctx.fillStyle = CONFIG.colors.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = CONFIG.brushSize;
  ctx.lineCap = "round";
  ctx.strokeStyle = CONFIG.colors.ink;
  ctx.beginPath();
  state.hasDrawing = false;
  state.hasPredicted = false;
  updateDebugEvents();
}

function attachPointerHandlers(canvas, ctx) {
  canvas.style.touchAction = "none";
  updateDebugCanvas(canvas);

  canvas.addEventListener("pointerdown", (e) => {
    if (state.isPredicting) return;
    clearIdleTimer();
    if (state.hasPredicted) {
      primeCanvas(canvas, ctx);
      const { result, statusHint } = getElements();
      if (result) result.textContent = "-";
      if (statusHint) statusHint.textContent = "Idle";
    }
    state.isPointerDown = true;
    state.activePointerId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    state.lastEvent = "pointerdown";
    updateDebugEvents();
    const { statusHint } = getElements();
    if (statusHint) statusHint.textContent = "Drawing...";
    drawPoint(e, canvas, ctx);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!state.isPointerDown || e.pointerId !== state.activePointerId) return;
    state.lastEvent = "pointermove";
    updateDebugEvents();
    drawPoint(e, canvas, ctx, true);
  });

  const stopDrawing = (shouldSchedule) => {
    if (!state.isPointerDown) return;
    state.isPointerDown = false;
    state.activePointerId = null;
    ctx.beginPath();
    updateDebugEvents();
    if (!state.isPredicting && shouldSchedule && state.hasDrawing) schedulePredict();
  };

  canvas.addEventListener("pointerup", () => {
    state.lastEvent = "pointerup";
    stopDrawing(true);
  });
  canvas.addEventListener("pointercancel", () => {
    state.lastEvent = "pointercancel";
    stopDrawing(true);
  });
  canvas.addEventListener("lostpointercapture", () => {
    state.lastEvent = "lostpointercapture";
    stopDrawing(true);
  });
  canvas.addEventListener("pointerleave", () => {
    state.lastEvent = "pointerleave";
    stopDrawing(false);
  });
}

function drawPoint(e, canvas, ctx, connect = false) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  if (!connect) {
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
  ctx.lineTo(x, y);
  ctx.stroke();
  state.hasDrawing = true;
}

function clearIdleTimer() {
  if (state.idleTimer) {
    clearTimeout(state.idleTimer);
    state.idleTimer = null;
    updateDebugEvents();
  }
}

function schedulePredict() {
  clearIdleTimer();
  state.idleTimer = setTimeout(() => {
    const elements = getElements();
    predict(elements);
  }, CONFIG.autoPredictDelayMs);
  state.lastEvent = "idle-timer";
  const { statusHint } = getElements();
  if (statusHint) statusHint.textContent = "Predicting soon...";
  updateDebugEvents();
}

function getInputTensor(canvas, preview, { invert = true } = {}) {
  const tmp = document.createElement("canvas");
  tmp.width = CONFIG.downscaleSize;
  tmp.height = CONFIG.downscaleSize;
  const tctx = tmp.getContext("2d");
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(canvas, 0, 0, CONFIG.downscaleSize, CONFIG.downscaleSize);

  if (preview && CONFIG.debug.showPreview) {
    const pctx = preview.getContext("2d");
    pctx.imageSmoothingEnabled = false;
    pctx.clearRect(0, 0, preview.width, preview.height);
    pctx.drawImage(tmp, 0, 0, preview.width, preview.height);
  }

  const { data } = tctx.getImageData(0, 0, CONFIG.downscaleSize, CONFIG.downscaleSize);
  const arr = new Float32Array(CONFIG.downscaleSize * CONFIG.downscaleSize);
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumsq = 0;
  let nonZero = 0;

  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    let gray = (r + g + b) / 3 / 255;
    if (invert) gray = 1 - gray;
    arr[j] = gray;
    if (gray < min) min = gray;
    if (gray > max) max = gray;
    sum += gray;
    sumsq += gray * gray;
    if (gray > 0.1) nonZero += 1;
  }

  const mean = sum / arr.length;
  const variance = sumsq / arr.length - mean * mean;
  const std = Math.sqrt(Math.max(variance, 0));
  const tensor = tf.tensor(arr, [1, CONFIG.downscaleSize, CONFIG.downscaleSize, 1], "float32");
  if (CONFIG.debug.logTensor) tensor.array().then(val => console.log("input", val));
  return {
    tensor,
    stats: {
      min,
      max,
      mean,
      std,
      nonZeroRatio: nonZero / arr.length,
    },
    smoothingEnabled: tctx.imageSmoothingEnabled,
  };
}

function setPredicting(flag, elements) {
  state.isPredicting = flag;
  const { statusHint, canvas } = elements;
  if (statusHint) statusHint.textContent = flag ? "Predicting..." : "Idle";
  if (canvas) {
    canvas.style.pointerEvents = flag ? "none" : "auto";
    canvas.style.opacity = flag ? "0.75" : "1";
  }
}

async function predict(elements) {
  const { canvas, preview, result } = elements;
  if (!state.model || !canvas || state.isPredicting) return;
  clearIdleTimer();
  setPredicting(true, elements);
  result.textContent = "...";

  const results = [];
  try {
    const startedAt = performance.now();
    for (const invert of [true]) {
      const { tensor: input, stats, smoothingEnabled } = getInputTensor(canvas, preview, { invert });
      state.lastInputStats = stats;
      state.lastInvert = invert;
      state.lastSmoothingEnabled = smoothingEnabled;
      updateDebugInput(stats, invert);
      updateDebugCanvas(canvas);
      const inputName = state.model.inputs[0].name;
      const yRaw = state.model.execute({ [inputName]: input });
      const y = Array.isArray(yRaw) ? yRaw[0] : yRaw;

      try {
        const probs = await y.data();
        let bestIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[bestIdx]) bestIdx = i;
        }
        const top3 = getTopK(probs, 3);
        results.push({ invert, bestIdx, bestProb: probs[bestIdx], top3 });
      } finally {
        input.dispose();
        y.dispose();
        if (Array.isArray(yRaw)) yRaw.forEach(t => t.dispose());
      }
    }

    results.sort((a, b) => b.bestProb - a.bestProb);
    const best = results[0];
    result.textContent = String(best.bestIdx);
    state.hasPredicted = true;
    state.lastPredictTop3 = best.top3 || null;
    state.lastPredictLatencyMs = performance.now() - startedAt;
    updateDebugPrediction(best.bestIdx, best.bestProb, best.top3, state.lastPredictLatencyMs);
    updateDebugEvents();
    console.log("Chosen preprocessing:", best.invert ? "inverted" : "normal", best);
  } finally {
    setPredicting(false, elements);
  }
}

function init() {
  const { app, canvas, clearBtn, result, statusHint } = getElements();
  if (!canvas || !clearBtn || !result) return;

  if (app && (CONFIG.debug.showPreview || CONFIG.debug.showPanel)) {
    app.dataset.debug = "true";
  }

  const ctx = canvas.getContext("2d");
  primeCanvas(canvas, ctx);
  attachPointerHandlers(canvas, ctx);
  updateDebugStatic();
  updateDebugCanvas(canvas);
  updateDebugInput(state.lastInputStats, state.lastInvert);
  updateDebugPrediction(null, null, null, state.lastPredictLatencyMs);
  updateDebugEvents();
  window.addEventListener("resize", () => updateDebugCanvas(canvas));

  clearBtn.addEventListener("click", () => {
    primeCanvas(canvas, ctx);
    result.textContent = "-";
    clearIdleTimer();
    state.lastEvent = "clear";
    if (statusHint) statusHint.textContent = "Idle";
    updateDebugInput(null, state.lastInvert);
    updateDebugPrediction(null, null, null, state.lastPredictLatencyMs);
    updateDebugEvents();
  });
}

window.addEventListener("load", async () => {
  const elements = getElements();
  init();
  try {
    await loadModel();
    if (elements.statusHint) elements.statusHint.textContent = "Idle";
  } catch (err) {
    console.error("Error loading model", err);
    if (elements.statusHint) elements.statusHint.textContent = "Load failed";
  }
});
