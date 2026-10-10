// NuScope Studio — UI logic (no build step, no framework).
"use strict";

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const api = (p, o) => fetch(p, o).then(r => { if (!r.ok) throw new Error(r.statusText); return r; });
const PALETTE = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#8DD3C7","#FB8072","#80B1D3"];
const hex2rgb = h => [1,3,5].map(i => parseInt(h.slice(i, i+2), 16));

const S = {
  project: null, vols: {}, base: null, ovl: null, info: {},
  pos: [0,0,0], mode: "explore", alpha: 0.45, win: [0,255],
  views: [], strokes: [], cls: [{name:"class 1"},{name:"class 2"},{name:"class 3"}], curCls: 1,
  sliceCache: new Map(), gl: null,
};

// ------------------------------------------------------------------ project
async function loadProject() {
  const p = await (await api("/api/project")).json();
  S.project = p; S.vols = Object.fromEntries(p.volumes.map(v => [v.id, v]));
  $("#project-name").textContent = `${p.name} · v${p.version}`;
  const vl = $("#volumes"); vl.innerHTML = "";
  for (const v of p.volumes) {
    const d = document.createElement("div");
    d.className = "item" + (v.id === S.base ? " base" : "") + (v.id === S.ovl ? " ovl" : "");
    d.innerHTML = `<span class="k ${v.kind}">${v.kind === "labels" ? "LBL" : v.kind === "element" ? "EDS" : "IMG"}</span>
      <span>${v.id}</span><span class="s">${v.shape.join("×")}</span>`;
    d.title = "click: show as base / labels go to overlay";
    d.onclick = () => v.kind === "labels" ? setOverlay(v.id) : setBase(v.id);
    vl.appendChild(d);
  }
  for (const [sel, kinds, none] of [["#base-sel", ["intensity","element"], false], ["#ovl-sel", ["labels"], true]]) {
    const el = $(sel), cur = sel === "#base-sel" ? S.base : S.ovl;
    el.innerHTML = (none ? `<option value="">— none —</option>` : "") +
      p.volumes.filter(v => kinds.includes(v.kind)).map(v => `<option ${v.id === cur ? "selected" : ""}>${v.id}</option>`).join("");
  }
  const list = (el, items, fn) => { el.innerHTML = ""; for (const t of items) { const d = document.createElement("div");
    d.className = "item"; d.textContent = t; d.onclick = () => fn(t); el.appendChild(d); } };
  list($("#tables"), p.tables, showTable);
  list($("#media"), p.media, m => window.open(`/api/media/${m}`, "_blank"));
  if (!S.base) { const first = p.volumes.find(v => v.kind !== "labels"); if (first) await setBase(first.id); }
  if (!S.ovl) { const l = p.volumes.find(v => v.kind === "labels" && !v.id.includes("object")); if (l) await setOverlay(l.id); }
}

async function setBase(id) {
  S.base = id; S.sliceCache.clear();
  const info = S.info[id] = await (await api(`/api/volume/${id}/info`)).json();
  S.pos = info.shape.map(n => n >> 1);
  drawHist(info); $("#vol-info").textContent = describe(info);
  for (const v of S.views) v.fit();
  refreshSlices(); load3D(); loadProject();
}
async function setOverlay(id) {
  S.ovl = id || null; S.sliceCache.clear();
  if (id) S.info[id] = await (await api(`/api/volume/${id}/info`)).json();
  refreshSlices(); load3D(); loadProject();
  if (id) labelLegend(S.info[id]);
}
function describe(i) {
  const e = i.extent_um.map(x => x.toFixed(2));
  return `${i.kind} · ${i.dtype}\n${i.shape.join(" × ")} voxels (z·y·x)\nvoxel ${i.voxel_nm.map(v => +v.toFixed(2)).join(" × ")} nm\nextent ${e.join(" × ")} µm`;
}
function labelLegend(info) {
  const cls = (info.meta && info.meta.classes) || {};
  const ids = Object.keys(cls);
  if (ids.length) S.cls = ids.map(k => ({ name: cls[k] }));
  renderClasses();
}

// ------------------------------------------------------------------ slices
async function getSlice(id, axis, index) {
  const w = id === S.base ? S.win : [0, 255];
  const key = `${id}|${axis}|${index}|${w}`;
  if (S.sliceCache.has(key)) return S.sliceCache.get(key);
  const p = api(`/api/volume/${id}/slice?axis=${axis}&index=${index}&lo=${w[0]}&hi=${w[1]}`).then(r => r.blob())
    .then(b => createImageBitmap(b, { colorSpaceConversion: "none", premultiplyAlpha: "none" }));
  S.sliceCache.set(key, p);
  if (S.sliceCache.size > 600) S.sliceCache.delete(S.sliceCache.keys().next().value);
  return p;
}
function lut() {
  const t = new Uint8ClampedArray(256 * 4);
  for (let i = 1; i < 256; i++) { const c = hex2rgb(PALETTE[(i - 1) % PALETTE.length]);
    t.set([...c, 255], i * 4); }
  return t;
}
const LUT = lut();

class SliceView {
  constructor(el) {
    this.el = el; this.axis = +el.dataset.axis; this.cv = el.querySelector("canvas");
    this.ctx = this.cv.getContext("2d"); this.zoom = 1; this.pan = [0, 0];
    this.off = document.createElement("canvas");
    this._events();
  }
  dims() { // displayed image (w,h) in voxels + physical aspect (pixel w,h in nm)
    const i = S.info[S.base]; if (!i) return null;
    const [nz, ny, nx] = i.shape, [vz, vy, vx] = i.voxel_nm;
    return [[nx, ny, vx, vy], [nx, nz, vx, vz], [ny, nz, vy, vz]][this.axis];
  }
  fit() {
    const d = this.dims(); if (!d) return;
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    const pw = d[0] * d[2], ph = d[1] * d[3];
    this.scale = Math.min(W / pw, H / ph) * 0.96; this.zoom = 1; this.pan = [0, 0];
  }
  toScreen(u, v) { const d = this.dims(), s = this.scale * this.zoom;
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    return [W/2 + this.pan[0] + (u - d[0]/2) * d[2] * s, H/2 + this.pan[1] + (v - d[1]/2) * d[3] * s]; }
  toVoxel(x, y) { const d = this.dims(), s = this.scale * this.zoom;
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    return [(x - W/2 - this.pan[0]) / (d[2] * s) + d[0]/2, (y - H/2 - this.pan[1]) / (d[3] * s) + d[1]/2]; }
  async draw() {
    const d = this.dims(); if (!d) return;
    const dpr = devicePixelRatio, W = this.cv.clientWidth, H = this.cv.clientHeight;
    if (this.cv.width !== W * dpr) { this.cv.width = W * dpr; this.cv.height = H * dpr; if (!this.scale) this.fit(); }
    const idx = S.pos[this.axis];
    const [img, lab] = await Promise.all([getSlice(S.base, this.axis, idx),
      S.ovl ? getSlice(S.ovl, this.axis, idx) : null]);
    if (S.pos[this.axis] !== idx) return;
    const c = this.ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = "#05070c"; c.fillRect(0, 0, W, H);
    const [x0, y0] = this.toScreen(0, 0), [x1, y1] = this.toScreen(d[0], d[1]);
    c.imageSmoothingEnabled = this.zoom < 2;
    c.drawImage(img, x0, y0, x1 - x0, y1 - y0);
    if (lab && S.alpha > 0) { c.globalAlpha = S.alpha; c.imageSmoothingEnabled = false;
      c.drawImage(colorize(lab, this.off), x0, y0, x1 - x0, y1 - y0); c.globalAlpha = 1; }
    if (this.axis === 0) this.drawStrokes(c);
    this.crosshair(c, d);
    this.el.querySelector(".idx").textContent = `${["z","y","x"][this.axis]} ${idx + 1}/${S.info[S.base].shape[this.axis]}`;
  }
  crosshair(c, d) {
    const cols = ["#ff6b6b", "#5eea7a", "#6aa8ff"], others = [[2, 1], [2, 0], [1, 0]][this.axis];
    const [ha, va] = others; // horizontal-axis voxel → which global axis; vertical
    const [sx] = this.toScreen(S.pos[ha] + 0.5, 0), [, sy] = this.toScreen(0, S.pos[va] + 0.5);
    c.lineWidth = 1; c.setLineDash([4, 4]);
    c.strokeStyle = cols[ha]; c.beginPath(); c.moveTo(sx, 0); c.lineTo(sx, this.cv.clientHeight); c.stroke();
    c.strokeStyle = cols[va]; c.beginPath(); c.moveTo(0, sy); c.lineTo(this.cv.clientWidth, sy); c.stroke();
    c.setLineDash([]);
    // scale bar
    const s = this.scale * this.zoom, nmPerPx = 1 / s, target = 120 * nmPerPx;
    const nice = [10,20,50,100,200,500,1000,2000,5000,10000,20000].filter(v => v <= target).pop() || 10;
    const L = nice * s, X = this.cv.clientWidth - 20 - L, Y = this.cv.clientHeight - 18;
    c.fillStyle = "#fff"; c.fillRect(X, Y, L, 3); c.font = "600 11px -apple-system";
    c.textAlign = "center"; c.fillText(nice >= 1000 ? `${nice/1000} µm` : `${nice} nm`, X + L/2, Y - 5);
  }
  drawStrokes(c) {
    for (const st of S.strokes) { if (st.z !== S.pos[0]) continue;
      c.strokeStyle = PALETTE[(st.class - 1) % PALETTE.length]; c.lineCap = c.lineJoin = "round";
      const d = this.dims(); c.lineWidth = st.radius * 2 * d[2] * this.scale * this.zoom;
      c.beginPath(); st.points.forEach(([y, x], i) => { const [sx, sy] = this.toScreen(x + .5, y + .5);
        i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); }); c.stroke(); }
  }
  _events() {
    let drag = null, stroke = null;
    const at = e => { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    this.cv.addEventListener("contextmenu", e => e.preventDefault());
    this.cv.addEventListener("pointerdown", e => {
      this.cv.setPointerCapture(e.pointerId);
      const [x, y] = at(e);
      if (e.button === 2 || e.button === 1 || e.shiftKey) { drag = { pan: [x, y] }; return; }
      if (S.mode === "segment" && this.axis === 0) {
        const [u, v] = this.toVoxel(x, y);
        stroke = { class: S.curCls, z: S.pos[0], radius: +$("#brush").value, points: [[v | 0, u | 0]] };
        S.strokes.push(stroke); this.draw(); return;
      }
      drag = { pick: true }; this.pick(x, y);
    });
    this.cv.addEventListener("pointermove", e => {
      const [x, y] = at(e);
      if (stroke) { const [u, v] = this.toVoxel(x, y); stroke.points.push([v | 0, u | 0]); this.draw(); return; }
      if (!drag) return;
      if (drag.pan) { this.pan[0] += x - drag.pan[0]; this.pan[1] += y - drag.pan[1]; drag.pan = [x, y]; this.draw(); }
      else if (drag.pick) this.pick(x, y);
    });
    this.cv.addEventListener("pointerup", () => { drag = null; if (stroke) { stroke = null; annStat(); } });
    this.cv.addEventListener("wheel", e => { e.preventDefault();
      if (e.ctrlKey || e.metaKey) { const f = Math.exp(-e.deltaY * 0.002); this.zoom = Math.max(0.5, Math.min(40, this.zoom * f)); this.draw(); return; }
      const n = S.info[S.base].shape[this.axis];
      S.pos[this.axis] = Math.max(0, Math.min(n - 1, S.pos[this.axis] + Math.sign(e.deltaY) * (e.shiftKey ? 10 : 1)));
      refreshSlices(); }, { passive: false });
    this.cv.addEventListener("dblclick", () => { this.fit(); this.draw(); });
  }
  pick(x, y) {
    const [u, v] = this.toVoxel(x, y), sh = S.info[S.base].shape;
    const [ha, va] = [[2, 1], [2, 0], [1, 0]][this.axis];
    S.pos[ha] = Math.max(0, Math.min(sh[ha] - 1, u | 0)); S.pos[va] = Math.max(0, Math.min(sh[va] - 1, v | 0));
    refreshSlices();
  }
}
function colorize(bmp, off) {
  off.width = bmp.width; off.height = bmp.height;
  const c = off.getContext("2d", { willReadFrequently: true }); c.drawImage(bmp, 0, 0);
  const im = c.getImageData(0, 0, bmp.width, bmp.height), d = im.data;
  for (let i = 0; i < d.length; i += 4) { const k = d[i] * 4;
    d[i] = LUT[k]; d[i+1] = LUT[k+1]; d[i+2] = LUT[k+2]; d[i+3] = LUT[k+3]; }
  c.putImageData(im, 0, 0); return off;
}
function refreshSlices() { for (const v of S.views) v.draw(); }

// ---------------------------------------------------------------------- 3D
async function fetch3D(id) {
  const buf = await (await api(`/api/volume/${id}/preview3d`)).arrayBuffer();
  const n = new DataView(buf).getUint32(0, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { ...hdr, bytes: new Uint8Array(buf, 4 + n) };
}
async function load3D() {
  if (!S.gl) return;
  $("#gl-msg").textContent = "loading 3D preview…";
  try {
    if (S.base) { const v = await fetch3D(S.base); S.gl.setVolume(v.shape, v.voxel_nm, v.bytes); }
    if (S.ovl) { const l = await fetch3D(S.ovl); S.gl.setLabels(l.shape, l.voxel_nm, l.bytes);
      const m = {}; for (let i = 1; i < 256; i++) m[i] = [...hex2rgb(PALETTE[(i-1) % PALETTE.length]), 255];
      S.gl.setLabelColors(m); } else S.gl.setLabels(null);
    $("#gl-msg").textContent = "";
  } catch (e) { $("#gl-msg").textContent = "3D: " + e.message; }
}

// --------------------------------------------------------------- histogram
function drawHist(info) {
  const c = $("#hist"), x = c.getContext("2d"), h = info.histogram.counts;
  const W = c.width = c.clientWidth * devicePixelRatio, H = c.height = 70 * devicePixelRatio;
  x.clearRect(0, 0, W, H); const m = Math.max(...h.slice(1)) || 1;
  x.fillStyle = "#4cc2ff";
  h.forEach((v, i) => { const bh = Math.min(1, Math.sqrt(v / m)) * (H - 4); x.fillRect(i * W / h.length, H - bh, W / h.length, bh); });
}

// ------------------------------------------------------------------- tables
let TABLE = null, SORT = null, SEL = null;
async function showTable(t) {
  TABLE = await (await api(`/api/table/${t}`)).json(); SORT = null; SEL = null;
  $("#drawer").classList.add("open"); $("#drawer-title").textContent = t; renderTable();
}
function renderTable() {
  const { columns, data } = TABLE; let rows = data.slice();
  if (SORT) rows.sort((a, b) => (a[SORT.i] > b[SORT.i] ? 1 : -1) * SORT.dir);
  const fmt = v => typeof v === "number" ? (Math.abs(v) >= 1000 || Number.isInteger(v) ? v.toLocaleString(undefined, {maximumFractionDigits: 1}) : v.toPrecision(4)) : v;
  $("#table-wrap").innerHTML = `<table><tr>${columns.map((c, i) => `<th data-i="${i}" class="${SEL === i ? "sel" : ""}">${c}</th>`).join("")}</tr>` +
    rows.slice(0, 1500).map(r => `<tr>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`).join("") + "</table>";
  $$("#table-wrap th").forEach(th => th.onclick = () => { const i = +th.dataset.i;
    SORT = { i, dir: SORT && SORT.i === i ? -SORT.dir : -1 }; SEL = i; renderTable(); colHist(i); });
}
function colHist(i) {
  const vals = TABLE.data.map(r => r[i]).filter(v => typeof v === "number" && isFinite(v));
  const c = $("#col-hist"), x = c.getContext("2d"); x.clearRect(0, 0, c.width, c.height);
  if (!vals.length) return;
  const lo = Math.min(...vals), hi = Math.max(...vals), nb = 30, b = new Array(nb).fill(0);
  vals.forEach(v => b[Math.min(nb - 1, ((v - lo) / (hi - lo || 1) * nb) | 0)]++);
  const m = Math.max(...b); x.fillStyle = "#E69F00";
  b.forEach((v, k) => { const h = v / m * (c.height - 30); x.fillRect(10 + k * (c.width - 20) / nb, c.height - 18 - h, (c.width - 20) / nb - 1, h); });
  x.fillStyle = "#8b95a8"; x.font = "11px -apple-system";
  x.fillText(TABLE.columns[i], 10, 12); x.fillText(lo.toPrecision(3), 10, c.height - 4);
  x.textAlign = "right"; x.fillText(hi.toPrecision(3), c.width - 10, c.height - 4); x.textAlign = "left";
}

// ----------------------------------------------------------------- ops / jobs
let OPS = [];
async function loadOps() {
  OPS = await (await api("/api/ops")).json();
  $("#op-sel").innerHTML = OPS.map((o, i) => `<option value="${i}">${o.name}</option>`).join("");
  $("#op-sel").onchange = opForm; opForm();
}
function opForm() {
  const o = OPS[+$("#op-sel").value]; $("#op-doc").textContent = o.summary;
  const volOpts = S.project.volumes.map(v => `<option>${v.id}</option>`).join("");
  $("#op-form").innerHTML = o.inputs.map(n => `<label>${n} <select data-in="${n}"><option value="">—</option>${volOpts}</select></label>`).join("") +
    Object.entries(o.params).map(([k, v]) => typeof v === "boolean"
      ? `<label class="chk"><input type="checkbox" data-p="${k}" ${v ? "checked" : ""}> ${k}</label>`
      : `<label>${k} <input type="text" data-p="${k}" value='${v === null ? "" : JSON.stringify(v)}'></label>`).join("");
  const first = $(`#op-form select[data-in="${o.inputs[0]}"]`); if (first && S.base) first.value = S.base;
}
async function runOp(op, inputs, params, save = true) {
  const r = await (await api("/api/run", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ op, inputs, params, save }) })).json();
  pollJobs(); return r.job;
}
$("#op-run").onclick = () => {
  const o = OPS[+$("#op-sel").value], inputs = {}, params = {};
  $$("#op-form [data-in]").forEach(s => { if (s.value) inputs[s.dataset.in] = s.value; });
  $$("#op-form [data-p]").forEach(el => { if (el.type === "checkbox") params[el.dataset.p] = el.checked;
    else if (el.value !== "") { try { params[el.dataset.p] = JSON.parse(el.value); } catch { params[el.dataset.p] = el.value; } } });
  runOp(o.name, inputs, params, $("#op-save").checked);
};
let polling = false;
async function pollJobs() {
  if (polling) return; polling = true;
  while (true) {
    const jobs = await (await api("/api/jobs")).json(), running = jobs.filter(j => j.status === "running");
    const pill = $("#jobs-pill");
    pill.textContent = running.length ? `⟳ ${running.map(j => j.op).join(", ")}` : (jobs[0] ? `${jobs[0].op}: ${jobs[0].status} (${jobs[0].seconds}s)` : "idle");
    pill.classList.toggle("busy", running.length > 0);
    if (jobs[0] && jobs[0].status === "error") pill.title = jobs[0].error;
    if (!running.length) { polling = false; await loadProject(); return; }
    await new Promise(r => setTimeout(r, 1000));
  }
}

// --------------------------------------------------------------- annotation
function renderClasses() {
  $("#classes").innerHTML = S.cls.map((c, i) => `<div class="cls ${S.curCls === i + 1 ? "on" : ""}" data-i="${i + 1}">
    <span class="sw" style="background:${PALETTE[i % PALETTE.length]}"></span><input value="${c.name}" data-n="${i}"></div>`).join("");
  $$("#classes .cls").forEach(d => d.onclick = () => { S.curCls = +d.dataset.i; renderClasses(); });
  $$("#classes input").forEach(inp => inp.onchange = () => S.cls[+inp.dataset.n].name = inp.value);
}
function annStat() {
  const by = {}; S.strokes.forEach(s => by[s.class] = (by[s.class] || 0) + 1);
  $("#ann-stat").textContent = `${S.strokes.length} strokes on ${new Set(S.strokes.map(s => s.z)).size} slices\n` +
    Object.entries(by).map(([k, n]) => `${S.cls[k - 1] ? S.cls[k - 1].name : k}: ${n}`).join(" · ");
}
$("#add-class").onclick = () => { S.cls.push({ name: `class ${S.cls.length + 1}` }); renderClasses(); };
$("#erase-strokes").onclick = () => { S.strokes.pop(); annStat(); refreshSlices(); };
$("#save-ann").onclick = async () => { await api(`/api/annotations/${S.base}`, { method: "POST",
  headers: { "content-type": "application/json" }, body: JSON.stringify(S.strokes) }); annStat(); };
$("#train").onclick = async () => {
  await $("#save-ann").onclick();
  const names = Object.fromEntries(S.cls.map((c, i) => [i + 1, c.name]));
  runOp("segment.random_forest", { volume: S.base }, { scribbles: "@annotations", classes: S.cls.length, class_names: names }, true);
};

// ------------------------------------------------------------------- wiring
function setMode(m) {
  S.mode = m; $$("#modes button").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  $("#panel-segment").hidden = m !== "segment"; $("#panel-recipe").hidden = m !== "recipe";
  $("#panel-3d").hidden = m === "recipe";
  if (m === "recipe") api("/api/recipe").then(r => r.text()).then(t => $("#recipe-text").textContent = t);
  if (m === "analyze") { $("#drawer").classList.add("open");
    const t = S.project.tables.find(t => t.includes("objects")) || S.project.tables[0]; if (t) showTable(t); }
  if (m === "segment") { api(`/api/annotations/${S.base}`).then(r => r.json()).then(s => { S.strokes = s; annStat(); refreshSlices(); }); renderClasses(); }
}
$$("#modes button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
$("#base-sel").onchange = e => setBase(e.target.value);
$("#ovl-sel").onchange = e => setOverlay(e.target.value);
$("#ovl-alpha").oninput = e => { S.alpha = +e.target.value; refreshSlices(); };
for (const id of ["#win-lo", "#win-hi"]) $(id).oninput = () => {
  S.win = [+$("#win-lo").value, +$("#win-hi").value];
  if (S.gl) { S.gl.state.wlo = S.win[0] / 255; S.gl.state.whi = S.win[1] / 255; S.gl.draw(); }
  refreshSlices(); };
$("#drawer-toggle").onclick = () => $("#drawer").classList.toggle("open");
$$("#mode3d button").forEach(b => b.onclick = () => { $$("#mode3d button").forEach(x => x.classList.toggle("on", x === b));
  S.gl.state.mode = +b.dataset.m; $("#mode3d-label").textContent = b.textContent.toLowerCase(); S.gl.draw(); });
$("#cmap").onchange = e => S.gl.setColormap(e.target.value);
$("#tf-thr").oninput = e => { S.gl.state.thr = +e.target.value; S.gl.draw(); };
$("#tf-dens").oninput = e => { S.gl.state.dens = +e.target.value; S.gl.draw(); };
$("#cut-faces").onchange = e => { S.gl.state.cutFaces = e.target.checked ? 1 : 0; S.gl.draw(); };
for (const a of ["x", "y", "z"]) for (const k of [0, 1]) $(`#c${a}${k}`).oninput = () => {
  const i = "xyz".indexOf(a); S.gl.state[k ? "clipMax" : "clipMin"][i] = +$(`#c${a}${k}`).value; S.gl.draw(); };
$("#snap").onclick = () => { const a = document.createElement("a"); a.href = $("#gl").toDataURL("image/png");
  a.download = "nuscope_3d.png"; a.click(); };
(function spin() { if ($("#autorotate").checked && S.gl) { S.gl.state.yaw += 0.006; S.gl.draw(); } requestAnimationFrame(spin); })();
window.addEventListener("resize", () => { S.views.forEach(v => { v.cv.width = 0; v.fit(); }); refreshSlices(); S.gl && S.gl.draw(); });

(async function init() {
  S.views = $$(".view[data-axis]").map(el => new SliceView(el));
  try { S.gl = new VolumeRenderer($("#gl")); } catch (e) { $("#gl-msg").textContent = e.message; }
  await loadProject(); await loadOps(); pollJobs(); renderClasses();
})();
