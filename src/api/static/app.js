// ECG Anomaly Detection UI. Plain JS, no build step; talks to the same FastAPI app.
"use strict";

const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
// classic 3 x 4 printout order: I aVR V1 V4 / II aVL V2 V5 / III aVF V3 V6
const GRID_ORDER = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];
const FS = 100; // Hz, PTB-XL at 100 Hz
const MAX_ROWS = 6000; // one recording: up to 60 s at 100 Hz (PTB-XL ECGs are 10 s = 1,000 rows)
const CLASSES = {
  NORM: { name: "Normal ECG", color: "#12b76a" },
  MI: { name: "Myocardial infarction", color: "#f04438" },
  STTC: { name: "ST/T change", color: "#f79009" },
  CD: { name: "Conduction disturbance", color: "#2e90fa" },
  HYP: { name: "Hypertrophy", color: "#7a5af8" },
};

const $ = (id) => document.getElementById(id);
const state = { signal: null, selected: 1, important: new Set() };

// ---------- CSV parsing ----------
function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) throw new Error("The file is empty.");
  const first = lines[0].split(",").map((s) => s.trim());
  const isHeader = first.some((c) => c !== "" && isNaN(Number(c)));
  const header = isHeader ? first : first.map((_, i) => `col${i}`);
  const rows = (isHeader ? lines.slice(1) : lines).map((l) => l.split(","));

  // find the 12 lead columns
  let cols = null;
  const lower = header.map((h) => h.toLowerCase());
  const byChannel = LEADS.map((_, i) => lower.indexOf(`channel-${i}`));
  const byName = LEADS.map((n) => lower.indexOf(n.toLowerCase()));
  if (byChannel.every((i) => i >= 0)) cols = byChannel;
  else if (byName.every((i) => i >= 0)) cols = byName;
  else {
    const numeric = header.map((h, i) => i).filter((i) => !["ecg_id", "id", "target", "time"].includes(lower[i]));
    if (numeric.length === 12) cols = numeric;
  }
  if (!cols) {
    throw new Error("Could not find 12 lead columns. Use channel-0 … channel-11 or I, II, III, aVR, aVL, aVF, V1–V6.");
  }

  // exactly one recording per file
  const idCol = lower.indexOf("ecg_id");
  if (idCol >= 0) {
    const ids = new Set(rows.map((r) => (r[idCol] || "").trim()));
    if (ids.size > 1) {
      throw new Error(`This file contains ${ids.size} ECGs (ecg_id ${[...ids].slice(0, 3).join(", ")}${ids.size > 3 ? ", …" : ""}). Upload a file with a single ECG.`);
    }
  }
  if (rows.length > MAX_ROWS) {
    throw new Error(`This file has ${rows.length.toLocaleString()} rows, more than one ${MAX_ROWS / FS} s ECG at ${FS} Hz. Upload a single recording.`);
  }
  const signal = [];
  for (const r of rows) {
    const v = cols.map((c) => Number(r[c]));
    if (v.some((x) => !Number.isFinite(x))) throw new Error(`Row ${signal.length + 2} has a missing or non-numeric value.`);
    signal.push(v);
  }
  if (signal.length < 100) throw new Error(`Need at least 100 samples per lead, got ${signal.length}.`);

  // optional cardiologist label (e.g. the generated test samples) to compare against
  const labelCol = lower.indexOf("label");
  const label = labelCol >= 0 ? (rows[0][labelCol] || "").trim().toUpperCase() || null : null;
  const ecgId = idCol >= 0 ? Number(rows[0][idCol]) || null : null;
  return { signal, ecgId, label };
}

// ---------- drawing ----------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight || Number(canvas.getAttribute("height"));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

// Draws one lead on ECG paper at 25 mm/s. Gain is the largest of 40/20/10/5/2.5 mm/mV that
// keeps the trace inside the strip (the standard is 10; narrow screens get more).
function drawLead(canvas, values, { grid = true, lineWidth = 1.4 } = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const seconds = values.length / FS;
  const pxPerMm = w / (seconds * 25);
  const base = median(values);
  const span = Math.max(...values.map((v) => Math.abs(v - base))) || 1;
  let gain = 40; // mm per mV
  while (gain > 2.5 && span * gain * pxPerMm > h * 0.46) gain /= 2;
  if (!grid) gain = (h * 0.42) / (span * pxPerMm); // thumbnails: fit to box

  ctx.clearRect(0, 0, w, h);
  if (grid) {
    for (let mm = 0, x = 0; x <= w; mm++, x = mm * pxPerMm) {
      ctx.strokeStyle = mm % 5 === 0 ? css("--grid-major") : css("--grid-minor");
      ctx.lineWidth = mm % 5 === 0 ? 0.9 : 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let mm = 0, y = 0; y <= h; mm++, y = mm * pxPerMm) {
      ctx.strokeStyle = mm % 5 === 0 ? css("--grid-major") : css("--grid-minor");
      ctx.lineWidth = mm % 5 === 0 ? 0.9 : 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }
  const mid = h / 2;
  ctx.strokeStyle = css("--trace");
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = mid - (v - base) * gain * pxPerMm;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  return gain;
}

function leadValues(i) {
  return state.signal.map((row) => row[i]);
}

function renderViewer() {
  if (!state.signal) return;
  const gain = drawLead($("main-plot"), leadValues(state.selected), { lineWidth: 1.6 });
  $("lead-title").textContent = `Lead ${LEADS[state.selected]}`;
  document.querySelector(".viewer-head .muted").textContent =
    `25 mm/s · ${gain} mm/mV · click a lead below to enlarge`;
  document.querySelectorAll(".lead").forEach((btn) => {
    const i = Number(btn.dataset.lead);
    btn.classList.toggle("active", i === state.selected);
    btn.classList.toggle("important", state.important.has(LEADS[i]));
    drawLead(btn.querySelector("canvas"), leadValues(i), { grid: false, lineWidth: 1 });
  });
}

function buildLeadButtons() {
  const box = $("leads");
  box.innerHTML = "";
  GRID_ORDER.forEach((i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "lead";
    b.dataset.lead = i;
    b.setAttribute("aria-label", `Show lead ${LEADS[i]}`);
    b.innerHTML = `<canvas></canvas><span>${LEADS[i]}</span>`;
    b.addEventListener("click", () => { state.selected = i; renderViewer(); });
    box.appendChild(b);
  });
}

// ---------- loading ----------
function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

function loadSignal(signal, label, ecgId = null, trueLabel = null) {
  state.signal = signal;
  state.ecgId = ecgId;
  state.trueLabel = trueLabel;
  state.important = new Set();
  $("viewer").hidden = false;
  $("analyze").disabled = false;
  $("reset").hidden = false;
  $("file-meta").textContent = `${label} · ${signal.length.toLocaleString()} samples × 12 leads · ${(signal.length / FS).toFixed(1)} s` +
    (trueLabel ? ` · label ${trueLabel}` : "");
  showError("");
  resetResult();
  requestAnimationFrame(renderViewer);
}

async function loadFile(file) {
  try {
    if (!/\.csv$/i.test(file.name)) throw new Error("Please choose a .csv file.");
    const { signal, ecgId, label } = parseCsv(await file.text());
    loadSignal(signal, file.name, ecgId, label);
  } catch (e) {
    showError(e.message);
  }
}

async function loadSample(name) {
  try {
    const r = await fetch(`/static/${name}`);
    if (!r.ok) throw new Error(`Could not load sample (${r.status}).`);
    const body = await r.json();
    loadSignal(body.signal, `Sample ${name.includes("1") ? "A" : "B"}`, body.ecg_id ?? null, body.label ?? null);
  } catch (e) {
    showError(e.message);
  }
}

// ---------- analysis ----------
async function postJson(path, payload) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `${path} failed (${r.status})`);
  }
  return body;
}

function resetResult() {
  $("placeholder").hidden = false;
  $("busy").hidden = true;
  $("result").hidden = true;
  $("result-meta").textContent = "";
}

function bar(label, fraction, valueText, color) {
  const row = document.createElement("div");
  row.className = "bar";
  row.innerHTML = `<span class="label"></span><span class="track"><span class="fill"></span></span><span class="value"></span>`;
  row.querySelector(".label").textContent = label;
  row.querySelector(".value").textContent = valueText;
  const fill = row.querySelector(".fill");
  if (color) fill.style.setProperty("--c", color);
  requestAnimationFrame(() => { fill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`; });
  return row;
}

function renderResult(pred, expl, ms) {
  const code = pred.prediction;
  const cls = CLASSES[code] || { name: code, color: css("--muted") };
  $("dx-code").textContent = code;
  $("diagnosis").style.setProperty("--dx", cls.color);
  $("dx-name").textContent = cls.name;
  $("dx-binary").textContent = code === "NORM" ? "Normal vs abnormal: normal" : "Normal vs abnormal: abnormal, review recommended";
  $("dx-conf").textContent = `${(pred.confidence * 100).toFixed(1)}%`;

  const probs = $("probs");
  probs.innerHTML = "";
  Object.entries(pred.probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .forEach(([k, p]) => probs.appendChild(bar(k, p, `${(p * 100).toFixed(1)}%`, (CLASSES[k] || {}).color)));

  const imp = $("importance");
  imp.innerHTML = "";
  const entries = Object.entries(expl.lead_importance || {});
  const max = Math.max(...entries.map(([, v]) => v), 0);
  const ranked = [...entries].sort((a, b) => b[1] - a[1]);
  state.important = new Set(max > 0 ? ranked.slice(0, 3).filter(([, v]) => v > 0).map(([k]) => k) : []);
  if (max <= 0) {
    const p = document.createElement("p");
    p.className = "muted small";
    p.textContent = "Blanking any single lead does not lower the confidence: the prediction does not depend on one lead.";
    imp.appendChild(p);
  } else {
    ranked.forEach(([k, v]) => imp.appendChild(bar(k, v / max, `${(v * 100).toFixed(2)} pp`, state.important.has(k) ? null : css("--muted"))));
  }

  const check = $("check");
  if (state.trueLabel) {
    const ok = state.trueLabel === code;
    const name = (CLASSES[state.trueLabel] || {}).name || state.trueLabel;
    check.className = `check ${ok ? "ok" : "miss"}`;
    check.textContent = ok
      ? `✓ Matches the cardiologist label: ${state.trueLabel} (${name}).`
      : `✗ Cardiologist label is ${state.trueLabel} (${name}); the model predicted ${code}.`;
    check.hidden = false;
  } else {
    check.hidden = true;
  }

  $("explanation").textContent = pred.explanation || "";
  $("result-meta").textContent = `${ms} ms · ${pred.model_version}`;
  $("placeholder").hidden = true;
  $("busy").hidden = true;
  $("result").hidden = false;
  renderViewer();
}

async function analyze() {
  if (!state.signal) return;
  $("analyze").disabled = true;
  showError("");
  $("placeholder").hidden = true;
  $("result").hidden = true;
  $("busy").hidden = false;
  const payload = { signal: state.signal, ecg_id: state.ecgId ?? undefined };
  const t0 = performance.now();
  try {
    const [pred, expl] = await Promise.all([postJson("/predict", payload), postJson("/explain", payload)]);
    renderResult(pred, expl, Math.round(performance.now() - t0));
  } catch (e) {
    resetResult();
    showError(e.message);
  } finally {
    $("analyze").disabled = false;
    refreshStats();
  }
}

// ---------- monitoring ----------
function fmtUptime(s) {
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}

async function refreshStats() {
  const status = $("status");
  try {
    const r = await fetch("/stats", { cache: "no-store" });
    if (!r.ok) throw new Error();
    const s = await r.json();
    status.className = "status up";
    $("status-text").textContent = "Service healthy";
    $("k-count").textContent = s.request_count.toLocaleString();
    $("k-latency").textContent = `${Math.round(s.avg_latency_ms)} ms`;
    $("k-errors").textContent = `${s.client_errors_4xx} / ${s.server_errors_5xx}`;
    $("k-uptime").textContent = fmtUptime(s.uptime_s);
    $("k-model").textContent = s.model_version;
    const rows = Object.entries(s.per_endpoint || {});
    $("endpoints").innerHTML = rows.length
      ? rows.map(([k, v]) => `<tr><td><code>${k}</code></td><td>${v.count}</td><td>${Math.round(v.avg_latency_ms)}</td><td>${Math.round(v.max_latency_ms)}</td></tr>`).join("")
      : `<tr><td colspan="4" class="muted">No requests yet</td></tr>`;
  } catch {
    status.className = "status down";
    $("status-text").textContent = "Service unreachable";
  }
}

// ---------- model card ----------
const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);

async function loadModelInfo() {
  try {
    const r = await fetch("/model", { cache: "no-store" });
    if (!r.ok) throw new Error();
    const m = await r.json();
    const date = m.trained_at ? new Date(m.trained_at) : null;
    const dateText = date && !isNaN(date)
      ? date.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" })
      : "unknown";
    const est = { lgbm: "LightGBM", xgboost: "XGBoost", rf: "Random forest", extra_tree: "Extra trees" }[m.estimator] || m.estimator || "model";
    $("m-acc").textContent = pct(m.accuracy);
    $("m-f1").textContent = m.f1_macro == null ? "–" : m.f1_macro.toFixed(3);
    $("m-f1ab").textContent = m.f1_abnormal == null ? "–" : m.f1_abnormal.toFixed(3);
    $("m-date").textContent = dateText;
    $("m-est").textContent = est;
    $("m-test").textContent = m.n_test ? `${m.n_test.toLocaleString()} held-out PTB-XL ECGs (fold 10)` : "the held-out PTB-XL test fold (fold 10)";
    $("m-version").textContent = m.model_version;
    const hdr = $("hdr-model");
    hdr.textContent = `Accuracy ${pct(m.accuracy)} · updated ${date && !isNaN(date) ? date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "–"}`;
    hdr.hidden = false;
  } catch {
    $("m-version").textContent = "model info unavailable";
  }
}

// ---------- wiring ----------
function init() {
  buildLeadButtons();
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => e.target.files[0] && loadFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 1) return showError(`Drop one ECG file at a time (you dropped ${files.length}).`);
    if (files[0]) loadFile(files[0]);
  });
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => loadSample(b.dataset.sample)));
  $("analyze").addEventListener("click", analyze);
  $("reset").addEventListener("click", () => {
    state.signal = null;
    state.trueLabel = null;
    $("viewer").hidden = true;
    $("analyze").disabled = true;
    $("reset").hidden = true;
    $("file-meta").textContent = "";
    $("file-input").value = "";
    showError("");
    resetResult();
  });
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderViewer, 120); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderViewer);
  loadModelInfo();
  refreshStats();
  setInterval(refreshStats, 5000);
}

init();
