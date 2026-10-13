// NUScope-TOMO Studio — application shell: workspaces, search palette, jobs, toasts, theme.
"use strict";

const VIEWER_WS = { explore: "explore", wizard: "wizard", segment: "segment" };
const PAGES = {}; // name -> render function, filled by pages.js
const Shell = { ws: null, jobs: [], seen: new Map() };

// ------------------------------------------------------------------ workspaces
function showWs(name, { push = true } = {}) {
  if (!name) name = "home";
  Shell.ws = name;
  const viewer = name in VIEWER_WS;
  for (const sec of document.querySelectorAll("#stage > .ws")) sec.hidden = true;
  (viewer ? $("#ws-viewer") : $(`#ws-${name}`) || $("#ws-home")).hidden = false;
  $$("#rail button").forEach((b) => b.classList.toggle("on", b.dataset.ws === name));
  if (viewer) setMode(VIEWER_WS[name]);
  else if (PAGES[name] && S.project) PAGES[name]();
  if (push && location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
}
$$("#rail button").forEach((b) => b.addEventListener("click", () => showWs(b.dataset.ws)));
window.addEventListener("hashchange", () => showWs(location.hash.slice(1), { push: false }));

// ------------------------------------------------------------------ theme
function applyTheme(t) {
  if (t === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("nuscope-theme", t); } catch (e) { /* storage blocked */ }
  if (Shell.ws && !(Shell.ws in VIEWER_WS) && PAGES[Shell.ws]) PAGES[Shell.ws]();
}
$("#theme-btn").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
try { applyTheme(localStorage.getItem("nuscope-theme") || "dark"); } catch (e) { applyTheme("dark"); }

// ------------------------------------------------------------------ toasts
function toast(title, body = "", kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  const b = document.createElement("b"); b.textContent = title; t.appendChild(b);
  if (body) t.appendChild(document.createTextNode(body));
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "err" ? 9000 : 5000);
}

// ------------------------------------------------------------------ jobs
function jobRow(j) {
  const d = document.createElement("div");
  d.className = `job ${j.status}`;
  const i = document.createElement("i");
  const o = document.createElement("span"); o.className = "o"; o.textContent = j.op;
  const t = document.createElement("span"); t.className = "t";
  t.textContent = j.status === "running" ? `running ${Math.round(Date.now() / 1000 - j.t0)} s` : `${j.status} · ${j.seconds ?? "–"} s`;
  d.append(i, o, t);
  const x = document.createElement("span"); x.className = "x";
  if (j.status === "error") x.textContent = j.error;
  else if (j.outputs && j.outputs.length) x.textContent = `→ ${j.outputs.join(", ")}`;
  if (x.textContent) d.appendChild(x);
  return d;
}
function renderJobs(target, jobs, limit = 50) {
  const el = $(target); if (!el) return;
  el.replaceChildren();
  if (!jobs.length) { const e = document.createElement("div"); e.className = "empty"; e.textContent = "No jobs yet in this session."; el.appendChild(e); return; }
  for (const j of jobs.slice(0, limit)) el.appendChild(jobRow(j));
}
window.onJobs = (jobs) => {
  Shell.jobs = jobs;
  for (const j of jobs) {
    const before = Shell.seen.get(j.id);
    if (before === "running" && j.status === "done") toast(`${j.op} finished`, j.outputs && j.outputs.length ? ` ${j.outputs.length} result(s) saved` : "", "ok");
    if (before === "running" && j.status === "error") toast(`${j.op} failed`, ` ${j.error}`, "err");
    Shell.seen.set(j.id, j.status);
  }
  renderJobs("#jp-list", jobs);
  if (Shell.ws === "home") renderJobs("#home-jobs", jobs, 6);
  if (!jobs.some((j) => j.status === "running") && Shell.ws && !(Shell.ws in VIEWER_WS) && PAGES[Shell.ws]) PAGES[Shell.ws]();
};
$("#jobs-pill").addEventListener("click", () => { $("#jobs-panel").hidden = !$("#jobs-panel").hidden; renderJobs("#jp-list", Shell.jobs); });
$("#jp-close").addEventListener("click", () => { $("#jobs-panel").hidden = true; });

// Run an operation from a page and follow it (pages call this).
async function runAndFollow(op, inputs, params = {}) {
  const job = await runOp(op, inputs, params, true);
  toast(`${op} started`, " Results appear here when it finishes.");
  return job;
}

// ------------------------------------------------------------------ palette
const PAL = { items: [], idx: 0 };
const WS_NAMES = { home: "Project home", explore: "View volumes", wizard: "Prepare (FIB-SEM wizard)", segment: "Annotate & segment",
  measure: "Measurements", confidence: "Confidence", chemistry: "Chemistry", acquire: "Acquire (live & self-driving)", recipe: "Recipe" };
function paletteItems() {
  const items = Object.entries(WS_NAMES).map(([k, n]) => ({ kind: "Page", name: n, run: () => showWs(k) }));
  for (const v of (S.project?.volumes || [])) items.push({ kind: v.kind === "labels" ? "Labels" : "Volume", name: v.id, detail: v.shape.join("×"),
    run: () => { showWs("explore"); v.kind === "labels" ? setOverlay(v.id) : setBase(v.id); } });
  for (const o of OPS) items.push({ kind: "Operation", name: o.name, detail: o.summary, run: () => { showWs("explore"); selectOp(o.name); } });
  for (const t of (S.project?.tables || [])) items.push({ kind: "Table", name: t, run: () => { showWs("measure"); openTable(t); } });
  items.push({ kind: "Action", name: "Switch light / dark theme", run: () => $("#theme-btn").click() });
  items.push({ kind: "Action", name: "Stop the self-driving run", run: () => $("#ag-stop").click() });
  return items;
}
function renderPalette() {
  const q = $("#pal-input").value.trim().toLowerCase();
  const terms = q.split(/\s+/).filter(Boolean);
  PAL.items = paletteItems().filter((it) => terms.every((t) => `${it.kind} ${it.name} ${it.detail || ""}`.toLowerCase().includes(t))).slice(0, 60);
  PAL.idx = Math.min(PAL.idx, Math.max(0, PAL.items.length - 1));
  const list = $("#pal-list"); list.replaceChildren();
  PAL.items.forEach((it, i) => {
    const li = document.createElement("li");
    li.className = i === PAL.idx ? "on" : "";
    const k = document.createElement("span"); k.className = "k"; k.textContent = it.kind;
    const n = document.createElement("span"); n.textContent = it.name;
    li.append(k, n);
    if (it.detail) { const d = document.createElement("span"); d.className = "d"; d.textContent = it.detail; li.appendChild(d); }
    li.addEventListener("click", () => { closePalette(); it.run(); });
    list.appendChild(li);
  });
  list.children[PAL.idx]?.scrollIntoView({ block: "nearest" });
}
function openPalette() { $("#palette").hidden = false; $("#pal-input").value = ""; PAL.idx = 0; renderPalette(); $("#pal-input").focus(); }
function closePalette() { $("#palette").hidden = true; }
$("#palette-btn").addEventListener("click", openPalette);
$("#palette").addEventListener("click", (e) => { if (e.target.id === "palette") closePalette(); });
$("#pal-input").addEventListener("input", () => { PAL.idx = 0; renderPalette(); });
$("#pal-input").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { PAL.idx = Math.min(PAL.items.length - 1, PAL.idx + 1); renderPalette(); e.preventDefault(); }
  else if (e.key === "ArrowUp") { PAL.idx = Math.max(0, PAL.idx - 1); renderPalette(); e.preventDefault(); }
  else if (e.key === "Enter" && PAL.items[PAL.idx]) { const it = PAL.items[PAL.idx]; closePalette(); it.run(); }
  else if (e.key === "Escape") closePalette();
});
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); $("#palette").hidden ? openPalette() : closePalette(); }
  else if (e.key === "Escape") { closePalette(); $("#jobs-panel").hidden = true; }
});

// ------------------------------------------------------------------ boot
window.onStudioReady = () => showWs(location.hash.slice(1) || "home", { push: false });
// Show the requested page straight away; its data fills in once the project has loaded.
setTimeout(() => { if (!Shell.ws) showWs(location.hash.slice(1) || "home", { push: false }); });
