// Phase 8: browser client for the FastAPI backend (same origin: served by api/main.py).
"use strict";

const API_BASE = "";                 // same server; e.g. "http://127.0.0.1:8000" if served elsewhere
const MAX_BYTES = 10 * 1024 * 1024;  // keep in sync with MAX_UPLOAD_BYTES in api/main.py
const ACCEPTED_TYPES = ["image/jpeg", "image/png"];

// Short title per traffic-light level; the advice text comes from the API (single source of truth).
const LEVEL_TITLES = {
  high: "Confianza alta",
  medium: "Confianza media",
  normal: "Sin signos claros",
  low: "Confianza baja",
};

const $ = (id) => document.getElementById(id);
const els = {
  status: $("api-status"), modelInfo: $("model-info"),
  dropzone: $("dropzone"), input: $("file-input"), empty: $("dropzone-empty"), preview: $("preview"),
  analyze: $("analyze-btn"), reset: $("reset-btn"), fileName: $("file-name"),
  placeholder: $("placeholder"), loading: $("loading"),
  errorBox: $("error-box"), errorTitle: $("error-title"), errorList: $("error-list"),
  result: $("result"), verdict: $("verdict"), level: $("verdict-level"), name: $("verdict-name"),
  advice: $("verdict-advice"), confidence: $("verdict-confidence"), warnings: $("warnings"),
  bars: $("bars"), original: $("original-img"), heatmap: $("heatmap-img"),
};

let selectedFile = null;
let previewUrl = null;

const percent = (value) => `${(value * 100).toFixed(1)} %`;

function show(panel) {
  // Only one of the result-side panels is visible at a time.
  for (const key of ["placeholder", "loading", "errorBox", "result"]) {
    els[key].hidden = key !== panel;
  }
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error(response.status);
    const info = await response.json();
    els.status.textContent = `Modelo listo (${info.model}, ${info.device})`;
    els.status.className = "status ok";
    els.modelInfo.textContent = `Modelo: ${info.model} · calibrado: ${info.calibrated ? "sí" : "no"}`;
  } catch {
    els.status.textContent = "No se pudo conectar con la API. ¿Está corriendo uvicorn?";
    els.status.className = "status down";
  }
}

function selectFile(file) {
  if (!file) return;
  if (!ACCEPTED_TYPES.includes(file.type)) {
    showError("Formato no válido", ["Usa una imagen JPG o PNG."]);
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("Imagen demasiado grande", ["El máximo es 10 MB."]);
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);  // free the previous preview
  previewUrl = URL.createObjectURL(file);
  els.preview.src = previewUrl;
  els.preview.hidden = false;
  els.empty.hidden = true;
  els.fileName.textContent = file.name;
  els.analyze.disabled = false;
  els.reset.hidden = false;
  show("placeholder");
}

function resetSelection() {
  selectedFile = null;
  els.input.value = "";
  els.preview.hidden = true;
  els.preview.removeAttribute("src");
  els.empty.hidden = false;
  els.fileName.textContent = "";
  els.analyze.disabled = true;
  els.reset.hidden = true;
  show("placeholder");
}

function showError(title, items = []) {
  els.errorTitle.textContent = title;
  els.errorList.replaceChildren(...items.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;  // textContent, never innerHTML: server text is not trusted as HTML
    return li;
  }));
  show("errorBox");
}

async function analyze() {
  if (!selectedFile) return;
  els.analyze.disabled = true;
  show("loading");

  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const response = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail;
      if (detail && typeof detail === "object" && !Array.isArray(detail)) {
        showError(detail.message, detail.issues || []);  // 422: the quality filter said why
      } else {
        showError(typeof detail === "string" ? detail : `Error ${response.status}`);
      }
      return;
    }
    renderResult(body);
  } catch {
    showError("No se pudo conectar con la API.", ["Revisa que el servidor esté corriendo."]);
  } finally {
    els.analyze.disabled = !selectedFile;
  }
}

function renderResult(data) {
  const { prediction } = data;
  els.verdict.className = `verdict ${prediction.level}`;
  els.level.textContent = LEVEL_TITLES[prediction.level] ?? prediction.level;
  els.name.textContent = prediction.name;
  // API message starts with the emoji light (e.g. "🟡 media — confirmar..."): drop it, the colour shows it.
  // For "high" the API only says "alta", already in the title.
  const advice = prediction.level === "high"
    ? "Igual debe confirmarlo un oftalmólogo."
    : prediction.message.replace(/^[^\p{L}]+/u, "");
  els.advice.textContent = advice.charAt(0).toUpperCase() + advice.slice(1);
  els.confidence.textContent = percent(prediction.confidence);

  els.warnings.replaceChildren(...data.warnings.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));

  els.bars.replaceChildren(...data.probabilities.map((item) => {
    const li = document.createElement("li");
    li.className = item.code === prediction.code ? "chosen" : "";
    const label = document.createElement("span");
    label.textContent = item.name;
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${Math.max(item.probability * 100, 0.5)}%`;
    track.append(fill);
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = percent(item.probability);
    li.append(label, track, pct);
    return li;
  }));

  els.original.src = previewUrl;
  els.heatmap.src = data.heatmap;
  show("result");
}

// ---------- events ----------
els.input.addEventListener("change", () => selectFile(els.input.files[0]));
els.analyze.addEventListener("click", analyze);
els.reset.addEventListener("click", resetSelection);

for (const type of ["dragenter", "dragover"]) {
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("dragging"));
}
els.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  selectFile(event.dataTransfer.files[0]);
});

checkHealth();
