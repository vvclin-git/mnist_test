const CONFIG = {
  canvasSize: 280,
  downscaleSize: 28,
  brushSize: 20,
  colors: {
    bg: "#ffffff",
    ink: "#0f172a",
  },
  modelUrl: "./web_graph_model_512_352/model.json",
  debug: {
    showPreview: false, // flip to true to show 28x28 preview
    logTensor: false,
  },
};

const state = {
  model: null,
  isDrawing: false,
  activePointerId: null,
};

function getElements() {
  return {
    app: document.getElementById("app"),
    canvas: document.getElementById("canvas"),
    preview: document.getElementById("preview"),
    predictBtn: document.getElementById("predictBtn"),
    clearBtn: document.getElementById("clearBtn"),
    result: document.getElementById("result"),
  };
}

async function loadModel() {
  state.model = await tf.loadGraphModel(CONFIG.modelUrl);
  console.log("GraphModel loaded");
  console.log("inputs:", state.model.inputs.map(x => x.name));
  console.log("outputs:", state.model.outputs.map(x => x.name));
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
}

function attachPointerHandlers(canvas, ctx) {
  canvas.addEventListener("pointerdown", (e) => {
    state.isDrawing = true;
    state.activePointerId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    drawPoint(e, canvas, ctx);
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!state.isDrawing || e.pointerId !== state.activePointerId) return;
    drawPoint(e, canvas, ctx, true);
  });

  const stopDrawing = () => {
    state.isDrawing = false;
    state.activePointerId = null;
    ctx.beginPath();
  };

  canvas.addEventListener("pointerup", stopDrawing);
  canvas.addEventListener("pointercancel", stopDrawing);
  canvas.addEventListener("pointerleave", stopDrawing);
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
}

function getInputTensor(canvas, preview, { invert = true } = {}) {
  const tmp = document.createElement("canvas");
  tmp.width = CONFIG.downscaleSize;
  tmp.height = CONFIG.downscaleSize;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(canvas, 0, 0, CONFIG.downscaleSize, CONFIG.downscaleSize);

  if (preview && CONFIG.debug.showPreview) {
    const pctx = preview.getContext("2d");
    pctx.clearRect(0, 0, preview.width, preview.height);
    pctx.drawImage(canvas, 0, 0, preview.width, preview.height);
  }

  const { data } = tctx.getImageData(0, 0, CONFIG.downscaleSize, CONFIG.downscaleSize);
  const arr = new Float32Array(CONFIG.downscaleSize * CONFIG.downscaleSize);

  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    let gray = (r + g + b) / 3 / 255;
    if (invert) gray = 1 - gray;
    arr[j] = gray;
  }

  const tensor = tf.tensor(arr, [1, CONFIG.downscaleSize, CONFIG.downscaleSize, 1], "float32");
  if (CONFIG.debug.logTensor) tensor.array().then(val => console.log("input", val));
  return tensor;
}

async function predict(canvas, preview, resultEl, predictBtn) {
  if (!state.model) return;
  if (predictBtn) predictBtn.disabled = true;
  resultEl.textContent = "...";

  const results = [];
  for (const invert of [true]) {
    const input = getInputTensor(canvas, preview, { invert });
    const inputName = state.model.inputs[0].name;
    const yRaw = state.model.execute({ [inputName]: input });
    const y = Array.isArray(yRaw) ? yRaw[0] : yRaw;

    try {
      const probs = await y.data();
      let bestIdx = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[bestIdx]) bestIdx = i;
      }
      results.push({ invert, bestIdx, bestProb: probs[bestIdx] });
    } finally {
      input.dispose();
      y.dispose();
      if (Array.isArray(yRaw)) yRaw.forEach(t => t.dispose());
    }
  }

  results.sort((a, b) => b.bestProb - a.bestProb);
  const best = results[0];
  resultEl.textContent = String(best.bestIdx);
  console.log("Chosen preprocessing:", best.invert ? "inverted" : "normal", best);
  if (predictBtn) predictBtn.disabled = false;
}

function init() {
  const { app, canvas, preview, predictBtn, clearBtn, result } = getElements();
  if (!canvas || !predictBtn || !clearBtn || !result) return;

  if (app && CONFIG.debug.showPreview) {
    app.dataset.debug = "true";
  }

  const ctx = canvas.getContext("2d");
  primeCanvas(canvas, ctx);
  attachPointerHandlers(canvas, ctx);

  clearBtn.addEventListener("click", () => {
    primeCanvas(canvas, ctx);
    result.textContent = "-";
  });

  predictBtn.addEventListener("click", () => predict(canvas, preview, result, predictBtn));
}

window.addEventListener("load", async () => {
  const elements = getElements();
  init();
  try {
    await loadModel();
    elements.predictBtn.disabled = false;
  } catch (err) {
    console.error("Error loading model", err);
    if (elements.predictBtn) elements.predictBtn.textContent = "Load failed";
  }
});
