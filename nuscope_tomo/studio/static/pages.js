// NUScope-TOMO Studio — workspace pages (home, measure, confidence, chemistry, acquire, recipe).
"use strict";

const F = Charts.fmt;
const PH_COLOR = (name, i = 0) => Charts.PHASE[name] || Charts.SERIES[i % Charts.SERIES.length];
const PH_ORDER = ["pore", "dark phase", "grey phase", "bright phase"];
const major = (r, key) => (r[key] ?? 1) >= 0.01;
const byPhase = (a, b) => (PH_ORDER.indexOf(a.name) + 99 * (PH_ORDER.indexOf(a.name) < 0)) - (PH_ORDER.indexOf(b.name) + 99 * (PH_ORDER.indexOf(b.name) < 0));

// ------------------------------------------------------------------ data helpers
const TCACHE = new Map();
function latest(re) {
  const ts = (S.project?.tables || []).filter((t) => re.test(t));
  ts.sort((a, b) => (S.project.mtime?.[b] || 0) - (S.project.mtime?.[a] || 0));
  return ts[0] || null;
}
async function rows(name) {
  if (!name) return null;
  const key = `${name}@${S.project.mtime?.[name]}`;
  if (!TCACHE.has(key)) {
    TCACHE.set(key, api(`/api/table/${encodeURI(name)}?limit=500000`).then((r) => r.json())
      .then(({ columns, data }) => Object.assign(data.map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i]]))), { columns })));
  }
  return TCACHE.get(key);
}
const T = {
  phases: /(phase_metrics\.phases|phases_\w+_phases)\.csv$/,
  particles: /(particles_measured|objects_\w+)\.csv$/,
  uncertainty: /(beyond_uncertainty|uncertainty_\w+)\.csv$/,
  rve: /(beyond_rve|rve_\w+_table)\.csv$/,
  transport: /(\/transport|transport_\w+)\.csv$/,
  tpb: /(tpb\.table|tpb_\w+_table)\.csv$/,
  qc: /(beyond_qc_slices|inspect_\w+_slices)\.csv$/,
  qcFlags: /(beyond_qc_flags|inspect_\w+_flags)\.csv$/,
  qcSummary: /(beyond_qc_summary|inspect_\w+_summary)\.csv$/,
  advisor: /(advisor_coarsening_\d+|advise_\w+_coarsening)\.csv$/,
  advice: /(advise_\w+_recommendation|tables\/advisor)\.csv$/,
  live: /live\/live_history\.csv$/,
  fusion: /chemical_fusion_summary\.csv$/,
  ablation: /self_driving_ablation\.csv$/,
};
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "on") e.addEventListener("click", v); else if (v != null) e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
function tile(label, value, unit = "", sub = "") {
  const v = el("div", { class: "v" }, value);
  if (unit) v.append(el("small", {}, unit));
  return el("div", { class: "tile" }, el("div", { class: "l" }, label), v, sub ? el("div", { class: "s" }, sub) : null);
}
function needOverlay() {
  if (S.ovl) return true;
  toast("Choose a label overlay first", " Pick a segmented volume in the View workspace (Overlay).", "err");
  return false;
}
function actionButtons(host, list) {
  const h = $(host); h.replaceChildren();
  for (const [label, fn, primary] of list) h.append(el("button", { class: primary ? "primary" : "", on: fn }, label));
}
const shortPhase = (n) => (n || "").replace(" phase", "");

// ------------------------------------------------------------------ home
PAGES.home = async () => {
  const p = S.project; if (!p) return;
  const base = S.base && S.info[S.base];
  const hero = $("#home-hero"); hero.replaceChildren();
  const left = el("div", {}, el("h1", {}, p.name), el("p", { class: "who" }, `NUScope-TOMO ${p.version} · created by Roberto dos Reis (@rmsreis) · part of NUScope, NUANCE Center, Northwestern University`));
  if (base) {
    const dl = el("dl");
    const add = (k, v) => dl.append(el("dt", {}, k), el("dd", {}, v));
    add("Base volume", S.base);
    add("Voxels (z × y × x)", base.shape.map((n) => n.toLocaleString()).join(" × "));
    add("Voxel size", `${base.voxel_nm.map((v) => +(+v).toFixed(2)).join(" × ")} nm`);
    add("Extent", `${base.extent_um.map((v) => v.toFixed(2)).join(" × ")} µm`);
    add("Project", `${p.volumes.length} volumes · ${p.tables.length} tables`);
    left.append(dl);
  }
  const thumbs = el("div", { class: "thumbs" });
  if (base) for (const [ax, name] of [[0, "XY"], [1, "XZ"]]) {
    const img = el("img", { alt: `${name} slice of ${S.base}`, src: `/api/volume/${S.base}/slice?axis=${ax}&index=${base.shape[ax] >> 1}` });
    thumbs.append(el("figure", {}, img, el("figcaption", {}, name)));
  }
  hero.append(left, thumbs);

  const vols = p.volumes;
  const steps = [
    ["Import", "Raw slice stack in the project", vols.some((v) => v.kind !== "labels"), "explore"],
    ["Prepare", "Crop, align, destripe, denoise", vols.some((v) => /wizard|denoised|corrected|n2v/.test(v.id)), "wizard"],
    ["Segment", "Phases from a few scribbles", vols.some((v) => v.kind === "labels"), "segment"],
    ["Measure", "Fractions, particles, transport", !!(latest(T.phases) || latest(T.particles)), "measure"],
    ["Confidence", "Data health and error bars", !!(latest(T.uncertainty) || latest(T.rve) || latest(T.qc)), "confidence"],
    ["Acquire", "Live and self-driving runs", !!(latest(T.live) || latest(T.ablation)), "acquire"],
  ];
  const ol = $("#home-steps"); ol.replaceChildren();
  steps.forEach(([t, d, ok, ws], i) => ol.append(el("li", { on: () => showWs(ws), title: `Open ${t}` },
    el("span", { class: "n" }, String(i + 1).padStart(2, "0")), el("span", { class: "t" }, t), el("span", { class: "d" }, d),
    el("span", { class: `st ${ok ? "done" : "todo"}` }, ok ? "Done" : "To do"))));

  actionButtons("#home-actions", [
    ["Prepare a raw stack", () => showWs("wizard")],
    ["Annotate and segment", () => showWs("segment")],
    ["Check data health", () => showWs("confidence")],
    ["Run the self-driving twin", () => showWs("acquire")],
    ["Search everything (⌘K)", openPalette],
  ]);
  renderJobs("#home-jobs", Shell.jobs, 6);

  const tiles = $("#home-tiles"); tiles.replaceChildren();
  const [unc, ph, parts, tr, ab] = await Promise.all([rows(latest(T.uncertainty)), rows(latest(T.phases)), rows(latest(T.particles)),
    rows(latest(T.transport)), rows(latest(T.ablation))]);
  if (unc) for (const r of unc.filter((r) => r.metric === "volume_fraction" && major(r, "value")).sort(byPhase))
    tiles.append(tile(`${r.name} fraction`, F.pct(r.value), "", `95 % sampling ${F.pct(r.sampling_ci95_low)}–${F.pct(r.sampling_ci95_high)}`));
  else if (ph) for (const r of ph.filter((r) => r.volume_fraction > 0.01).sort(byPhase)) tiles.append(tile(`${r.name} fraction`, F.pct(r.volume_fraction)));
  if (parts) {
    const inner = parts.filter((r) => !r.touches_border);
    tiles.append(tile("Particles measured", inner.length.toLocaleString(), "", `fully inside, of ${parts.length.toLocaleString()} separated`));
  }
  if (tr) { const g = tr.find((r) => /grey/.test(r.name) && r.axis === "z"); if (g) tiles.append(tile("Grey-phase tortuosity, Z", F.num(g.tortuosity_factor))); }
  if (ab) {
    const a = ab.find((r) => /agent/i.test(r.policy)), f = ab.find((r) => /full|^fixed recipe$/i.test(r.policy));
    if (a) tiles.append(tile("Self-driving time to target", F.num(a.time_h), "h", f ? `fixed recipe ${F.num(f.time_h)} h (twin)` : "twin"));
  }
  if (!tiles.children.length) tiles.append(el("div", { class: "empty" }, "Results appear here once you segment and measure."));
};

// ------------------------------------------------------------------ measure
PAGES.measure = async () => {
  actionButtons("#measure-actions", [
    ["Phase metrics", () => needOverlay() && runAndFollow("analyze.phases", { labels: S.ovl }), true],
    ["Transport", () => needOverlay() && runAndFollow("analyze.transport", { labels: S.ovl })],
    ["Triple-phase boundary", () => needOverlay() && runAndFollow("analyze.tpb", { labels: S.ovl })],
  ]);
  const [unc, ph, parts, tr, tpb] = await Promise.all([rows(latest(T.uncertainty)), rows(latest(T.phases)), rows(latest(T.particles)),
    rows(latest(T.transport)), rows(latest(T.tpb))]);

  if (unc) {
    $("#m-frac-note").textContent = "Upper whisker: 95 % sampling range. Lower whisker: every phase boundary moved by one voxel.";
    Charts.bars($("#m-frac"), { f: (v) => F.pct(v), axisFmt: (v) => `${(100 * v).toFixed(0)} %`,
      rows: unc.filter((r) => r.metric === "volume_fraction" && major(r, "value")).sort(byPhase).map((r) => ({
        label: r.name, value: r.value, lo: r.sampling_ci95_low, hi: r.sampling_ci95_high, lo2: r.boundary_low, hi2: r.boundary_high,
        color: PH_COLOR(r.name), tip: [{ name: "Fraction", value: F.pct(r.value) },
          { name: "Sampling 95 %", value: `${F.pct(r.sampling_ci95_low)}–${F.pct(r.sampling_ci95_high)}` },
          { name: "Boundary ±1 voxel", value: `${F.pct(r.boundary_low)}–${F.pct(r.boundary_high)}` }] })) });
  } else if (ph) {
    $("#m-frac-note").textContent = "Run Confidence → Uncertainty to add error bars.";
    Charts.bars($("#m-frac"), { f: (v) => F.pct(v), axisFmt: (v) => `${(100 * v).toFixed(0)} %`,
      rows: ph.slice().sort(byPhase).map((r) => ({ label: r.name, value: r.volume_fraction, color: PH_COLOR(r.name) })) });
  } else Charts.empty($("#m-frac"), "Run Phase metrics on a label overlay.");

  const pt = $("#m-phases"); pt.replaceChildren();
  if (ph) {
    $("#m-phase-note").textContent = `From ${latest(T.phases)}`;
    const cols = [["name", "Phase", (v) => v], ["volume_fraction", "Fraction", (v) => F.pct(v)], ["specific_surface_um-1", "S/V (µm⁻¹)", (v) => F.num(v)],
      ["largest_cluster_frac", "Largest cluster", (v) => F.pct(v, 0)], ["tortuosity_z", "Tortuosity Z", (v) => F.num(v)],
      ["tortuosity_y", "Y", (v) => F.num(v)], ["tortuosity_x", "X", (v) => F.num(v)]].filter(([k]) => ph.columns.includes(k));
    const t = el("table", { class: "t" }, el("thead", {}, el("tr", {}, ...cols.map(([, h]) => el("th", {}, h)))));
    const tb = el("tbody");
    for (const r of ph.slice().sort(byPhase)) tb.append(el("tr", {}, ...cols.map(([k, , f]) => el("td", {}, r[k] == null ? "–" : f(r[k])))));
    t.append(tb); pt.append(t);
  } else { $("#m-phase-note").textContent = ""; pt.append(el("div", { class: "empty" }, "No phase metrics yet.")); }

  if (parts) {
    const inner = parts.filter((r) => !r.touches_border);
    $("#m-size-note").textContent = `${inner.length.toLocaleString()} particles fully inside the volume (border-touching ones are left out).`;
    Charts.hist($("#m-size"), { values: inner.map((r) => r.eq_diameter_nm), xLabel: "equivalent diameter (nm)", color: Charts.css("c2") });
    Charts.scatter($("#m-shape"), { logX: true, xLabel: "equivalent diameter (nm, log)", yLabel: "sphericity",
      pts: inner.map((r) => ({ x: r.eq_diameter_nm, y: r.sphericity, head: `Particle ${r.label}`,
        tip: [{ name: "Diameter", value: `${F.auto(r.eq_diameter_nm)} nm` }, { name: "Sphericity", value: F.num(r.sphericity) },
          { name: "Elongation", value: F.num(r.elongation) }] })) });
  } else { $("#m-size-note").textContent = ""; Charts.empty($("#m-size"), "No particle table yet."); Charts.empty($("#m-shape"), "No particle table yet."); }

  if (tr) {
    const names = [...new Set(tr.map((r) => r.name))].sort((a, b) => byPhase({ name: a }, { name: b }));
    const colors = { z: Charts.css("c5"), y: Charts.css("c6"), x: Charts.css("c7") };
    const host = $("#m-transport");
    Charts.grouped(host, { keys: ["z", "y", "x"], colors, logX: true, f: (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(2)),
      groups: names.map((n) => ({ label: n, items: tr.filter((r) => r.name === n).map((r) => ({ key: r.axis, value: r.tortuosity_factor,
        tip: [{ name: "Tortuosity", value: F.num(r.tortuosity_factor) }, { name: "D_eff / D₀", value: (+r.D_eff_ratio).toFixed(3) }] })) })) });
    Charts.legend(host, [{ name: "Z (milling)", color: colors.z }, { name: "Y", color: colors.y }, { name: "X", color: colors.x }]);
  } else Charts.empty($("#m-transport"), "Run Transport on a label overlay.");

  const tp = $("#m-tpb"); tp.replaceChildren();
  if (tpb && tpb[0]) {
    const r = tpb[0];
    tp.append(tile("Length density", F.num(r["tpb_density_um-2"]), "µm⁻²", r.phases || ""),
      tile("On connected networks", F.pct(r.active_fraction, 0)), tile("Total length", F.auto(r.tpb_length_um), "µm"));
  } else tp.append(el("div", { class: "empty" }, "Run Triple-phase boundary on a label overlay."));

  const sel = $("#tx-sel"), cur = sel.value;
  sel.replaceChildren(...(S.project.tables || []).map((t) => el("option", { value: t }, t)));
  sel.value = cur && S.project.tables.includes(cur) ? cur : (latest(T.particles) || S.project.tables[0] || "");
  sel.onchange = () => openTable(sel.value);
  $("#tx-search").oninput = renderTx;
  if (sel.value) openTable(sel.value, false);
};

const TX = { name: null, rows: null, sort: null, col: null };
async function openTable(name, setSel = true) {
  if (setSel && $("#tx-sel")) $("#tx-sel").value = name;
  TX.name = name; TX.rows = await rows(name); TX.sort = null; TX.col = null;
  renderTx();
}
function renderTx() {
  const host = $("#tx-table"); if (!host || !TX.rows) return;
  const cols = TX.rows.columns, q = ($("#tx-search").value || "").toLowerCase();
  let rs = q ? TX.rows.filter((r) => cols.some((c) => String(r[c]).toLowerCase().includes(q))) : TX.rows.slice();
  if (TX.sort) rs.sort((a, b) => ((a[TX.sort.c] > b[TX.sort.c]) - (a[TX.sort.c] < b[TX.sort.c])) * TX.sort.d);
  const f = (v) => (typeof v === "number" ? F.auto(v) : v == null ? "–" : String(v));
  const head = el("tr", {}, ...cols.map((c) => {
    const th = el("th", { class: c === TX.col ? "sel" : "", "aria-sort": TX.sort?.c === c ? (TX.sort.d > 0 ? "ascending" : "descending") : null }, c);
    th.addEventListener("click", () => { TX.sort = { c, d: TX.sort?.c === c ? -TX.sort.d : -1 }; TX.col = c; renderTx(); });
    return th;
  }));
  const tb = el("tbody");
  for (const r of rs.slice(0, 2000)) tb.append(el("tr", {}, ...cols.map((c) => el("td", {}, f(r[c])))));
  host.replaceChildren(el("table", { class: "t" }, el("thead", {}, head), tb));
  const hc = $("#tx-hist");
  const vals = TX.col ? rs.map((r) => r[TX.col]).filter((v) => typeof v === "number") : [];
  if (vals.length) Charts.hist(hc, { values: vals, xLabel: TX.col, color: Charts.css("c1") });
  else Charts.empty(hc, `${rs.length.toLocaleString()} rows. Click a numeric column to see its distribution.`);
}

// ------------------------------------------------------------------ confidence
PAGES.confidence = async () => {
  actionButtons("#conf-actions", [
    ["Check data health", () => S.base && runAndFollow("qc.inspect", { volume: S.base }), true],
    ["Uncertainty & volume needed", () => { if (needOverlay()) { runAndFollow("analyze.uncertainty", { labels: S.ovl }); runOp("analyze.rve", { labels: S.ovl }, {}, true); } }],
    ["Acquisition advice", () => needOverlay() && runAndFollow("advise.acquisition", { labels: S.ovl })],
  ]);
  const [qc, flags, summ, unc, rve, adv, advice, live] = await Promise.all([T.qc, T.qcFlags, T.qcSummary, T.uncertainty, T.rve, T.advisor, T.advice, T.live]
    .map((re) => rows(latest(re))));

  const host = $("#c-qc"); host.replaceChildren();
  if (qc) {
    const s = summ && summ[0];
    $("#c-qc-note").textContent = s ? `${s.flagged_slices} of ${s.slices} slices flagged · median SNR ${F.num(s.snr_median, 1)} · brightness drift ${F.num(s.brightness_drift_pct, 1)} % · `
      + `correlation length Z/XY ${F.num(s.z_over_xy)}${s.warning ? ` · ${s.warning}` : ""}` : "";
    const flagged = new Set((flags || []).map((r) => r.z));
    for (const [k, name, color] of [["mean", "Brightness", "c2"], ["snr", "SNR", "c3"], ["curtaining", "Curtaining index", "c1"]]) {
      const d = el("div", { class: "chart" }); host.append(d);
      Charts.lines(d, { height: 140, endLabels: false, xLabel: k === "curtaining" ? "slice" : null, yMin: Math.min(...qc.map((r) => r[k])), yf: (v) => F.auto(v),
        series: [{ name, color: Charts.css(color), pts: qc.map((r) => ({ x: r.z, y: r[k] })) },
          { name: "flagged", dots: true, color: Charts.css("err"), pts: qc.filter((r) => flagged.has(r.z) && (flags || []).some((f) => f.z === r.z && f.metric === k)).map((r) => ({ x: r.z, y: r[k] })) }]
          .filter((s) => s.pts.length) });
      const lab = el("div", { class: "note" }, name); host.insertBefore(lab, d);
    }
  } else { $("#c-qc-note").textContent = ""; Charts.empty(host, "Run Check data health on the base volume."); }

  if (unc) Charts.bars($("#c-unc"), { f: (v) => F.pct(v), axisFmt: (v) => `${(100 * v).toFixed(0)} %`,
    rows: unc.filter((r) => r.metric === "volume_fraction" && major(r, "value")).sort(byPhase).map((r) => ({ label: r.name, value: r.value, color: PH_COLOR(r.name),
      lo: r.sampling_ci95_low, hi: r.sampling_ci95_high, lo2: r.boundary_low, hi2: r.boundary_high })) });
  else Charts.empty($("#c-unc"), "Run Uncertainty on a label overlay.");

  if (rve) {
    const edge = Math.cbrt(rve[0].volume_um3), col = rve.columns.find((c) => c.startsWith("rve_edge"));
    const rv = rve.filter((r) => major(r, "fraction")), ref = rv.find((r) => /dark/.test(r.name)) || rv[0];
    $("#c-rve-note").textContent = `This analysed volume is equivalent to a ${edge.toFixed(1)} µm cube; it pins the ${ref.name} fraction to ±${(100 * ref.rel_error_full_volume).toFixed(0)} % (95 %).`;
    Charts.bars($("#c-rve"), { f: (v) => `${v.toFixed(1)} µm`, axisFmt: (v) => `${v} µm`, ref: { v: edge, label: `analysed ≈ ${edge.toFixed(1)} µm` },
      rows: rv.slice().sort(byPhase).map((r) => ({ label: r.name, value: r[col], color: PH_COLOR(r.name) })) });
  } else { $("#c-rve-note").textContent = ""; Charts.empty($("#c-rve"), "Run Uncertainty & volume needed."); }

  if (adv) {
    const ok = adv.filter((r) => r.max_rel_change <= 0.05);
    const best = ok.sort((a, b) => a.imaging_time_factor - b.imaging_time_factor || a.max_rel_change - b.max_rel_change)[0];
    const a = advice && advice[advice.length - 1];
    $("#c-adv-note").textContent = a?.advice || "Largest change in any result if the specimen had been sampled coarser. Outlined: cheapest sampling within 5 %.";
    Charts.heat($("#c-adv"), { cols: [...new Set(adv.map((r) => r.slice_nm))].sort((x, y) => x - y), rows: [...new Set(adv.map((r) => r.pixel_nm))].sort((x, y) => x - y),
      xLabel: "slice thickness (nm)", yLabel: "pixel (nm)",
      cells: adv.map((r) => ({ col: r.slice_nm, row: r.pixel_nm, v: r.max_rel_change, label: `${(100 * r.max_rel_change).toFixed(0)}%`, outline: r === best,
        head: `${r.pixel_nm} nm pixels · ${r.slice_nm} nm slices`, tip: [{ name: "Largest change", value: F.pct(r.max_rel_change) },
          { name: "In", value: r.worst_metric }, { name: "Imaging time", value: F.pct(r.imaging_time_factor, 0) }] })) });
  } else { $("#c-adv-note").textContent = ""; Charts.empty($("#c-adv"), "Run Acquisition advice on a label overlay."); }

  if (live) {
    const names = live.columns.filter((c) => c.startsWith("err_") && !/pore/.test(c)).map((c) => c.slice(4));
    Charts.lines($("#c-live"), { logY: true, xLabel: "milled depth (µm)", yf: (v) => `${v.toFixed(0)} %`, refY: { v: 5, label: "±5 %" },
      series: names.map((n, i) => ({ name: shortPhase(n), color: PH_COLOR(n, i), pts: live.filter((r) => r[`err_${n}`] != null).map((r) => ({ x: r.depth_um, y: 100 * r[`err_${n}`] })) })) });
  } else Charts.empty($("#c-live"), "No live acquisition yet. Run nuscope watch next to the microscope.");
};

// ------------------------------------------------------------------ chemistry
const CHEM = { metric: "rmse" };
const METHOD_SHORT = { "full map": "Full EDS map", "coarse 6 %, upsampled": "Coarse map, upsampled", "DKL sparse 10 % (SE context)": "DKL, SE context",
  "DKL sparse 10 % (SE + BSE context)": "DKL, SE + BSE context", "CNMF: coarse 6 % + BSE": "Pan-sharpening, BSE",
  "CNMF: coarse 6 % + BSE + fast 4 %": "Pan-sharpening, BSE + fast map", "CNMF: coarse 6 % + SE (no physical link)": "Pan-sharpening, SE (no link)" };
$$("#chem-metric button").forEach((b) => b.addEventListener("click", () => {
  CHEM.metric = b.dataset.m; $$("#chem-metric button").forEach((x) => x.classList.toggle("on", x === b)); PAGES.chemistry();
}));
$("#chem-open-op").addEventListener("click", () => { showWs("explore"); selectOp("chem.eds_tomography"); });
PAGES.chemistry = async () => {
  actionButtons("#chem-actions", [["Open EDS tomography", () => { showWs("explore"); selectOp("chem.eds_tomography"); }]]);
  const fus = await rows(latest(T.fusion));
  const notes = { rmse: "Composition RMSE against the true chemistry; lower is better.", phase_acc: "Share of EDS pixels given the right phase; higher is better.",
    hidden_contrast: "Share of the hidden region's true contrast recovered; 100 % is perfect." };
  $("#chem-note").textContent = `${notes[CHEM.metric]} Right column: EDS time as a share of a full map. Simulated EDS on the real segmentation.`;
  if (!fus) return Charts.empty($("#chem-chart"), "Run python examples/chemical_fusion.py to create the benchmark.");
  const scen = [...new Set(fus.map((r) => r.scenario))];
  const series = scen.map((s, i) => ({ name: s.replace(/^\w\s+/, "").replace(/\s*\(.*/, ""), full: s, color: Charts.SERIES[[1, 0][i] ?? i] }));
  const cats = [...new Set(fus.map((r) => r.method))];
  const pct = CHEM.metric !== "rmse";
  const host = $("#chem-chart");
  Charts.dots(host, { cats: cats.map((c) => METHOD_SHORT[c] || c), series, f: pct ? (v) => F.pct(v, 0) : (v) => v.toFixed(3), rightHead: "EDS time",
    get: (c, sname) => { const s = series.find((x) => x.name === sname); const m = cats.find((k) => (METHOD_SHORT[k] || k) === c);
      return fus.find((r) => r.method === m && r.scenario === s.full)?.[CHEM.metric]; },
    right: (c) => { const m = cats.find((k) => (METHOD_SHORT[k] || k) === c); return F.pct(fus.find((r) => r.method === m).eds_time, 0); },
    emph: (c) => /SE \+ BSE/.test(c) });
  Charts.legend(host, series.map((s) => ({ name: `${s.name} chemistry`, color: s.color })));
};

// ------------------------------------------------------------------ acquire
const AG = { job: null, timer: null };
function fillSelect(sel, opts, prefer) {
  const cur = $(sel).value;
  $(sel).replaceChildren(...opts.map((o) => el("option", { value: o }, o)));
  $(sel).value = opts.includes(cur) ? cur : (opts.find((o) => prefer.test(o)) || opts[0] || "");
}
$("#ag-run").addEventListener("click", async () => {
  const body = { policy: $("#ag-policy").value, clean: $("#ag-clean").value, labels: $("#ag-labels").value, model: $("#ag-model").value,
    target: +$("#ag-target").value, max_depth_um: +$("#ag-depth").value,
    phases: $("#ag-phases").value.split(",").map((s) => s.trim()).filter(Boolean) };
  if (!body.clean || !body.labels || !body.model) return toast("Missing input", " Choose a specimen volume, a phase map and a classifier.", "err");
  const r = await (await api("/api/agent/start", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })).json();
  AG.job = r.job; toast("Twin run started", " Follow it below. Nothing is sent to a microscope.");
  pollJobs(); followAgent();
});
$("#ag-stop").addEventListener("click", async () => { await api("/api/agent/stop", { method: "POST" }); toast("Stop requested", " The run ends after the current slice."); });
async function followAgent() {
  clearTimeout(AG.timer);
  if (!AG.job) return;
  let d;
  try { d = await (await api(`/api/agent/${AG.job}`)).json(); } catch (e) { return; }
  drawAgent(d);
  if (d.status === "running") AG.timer = setTimeout(followAgent, 1500);
}
function drawAgent(d) {
  const running = d.status === "running";
  $("#ag-run").disabled = running; $("#ag-stop").disabled = !running;
  const st = $("#ag-status");
  const tr = d.trace || [], last = tr[tr.length - 1];
  st.className = `status-line ${d.status === "done" ? "ok" : d.status === "error" ? "err" : ""}`;
  st.textContent = d.status === "error" ? `Failed: ${d.error}`
    : d.status === "done" ? `${d.summary.reached ? "Target reached" : d.summary.stopped ? "Stopped by operator" : "Finished without reaching the target"} · `
      + `${d.summary.time_h.toFixed(2)} h instrument time · ${d.summary.depth_um.toFixed(1)} µm · ${d.summary.slices} slices · ${d.summary.eds_pixels.toLocaleString()} EDS pixels`
    : last ? `Running · slice ${last.slice} · ${last.depth_um.toFixed(2)} µm · ${(d.clock_h || 0).toFixed(2)} h instrument time` : "Loading the specimen into the twin…";
  if (!tr.length) return;
  const errCols = Object.keys(tr.find((r) => Object.keys(r).some((k) => k.startsWith("err_"))) || {}).filter((k) => k.startsWith("err_"));
  const want = $("#ag-phases").value.split(",").map((s) => s.trim());
  const target = 100 * +$("#ag-target").value;
  Charts.lines($("#ag-prec"), { xLabel: "instrument time (h)", yf: (v) => `${v.toFixed(0)} %`, refY: { v: target, label: `±${target.toFixed(0)} %` },
    series: errCols.filter((c) => want.includes(c.slice(4))).map((c, i) => ({ name: shortPhase(c.slice(4)), color: PH_COLOR(c.slice(4), i),
      pts: tr.filter((r) => r[c] != null).map((r) => ({ x: r.t_h, y: 100 * r[c] })) })) });
  Charts.lines($("#ag-ctrl"), { xLabel: "milled depth (µm)", step: true, yf: (v) => F.auto(v), series: [
    { name: "thickness nm", color: Charts.css("c5"), pts: tr.map((r) => ({ x: r.depth_um, y: r.thickness_nm })) },
    { name: "dwell µs ×10", color: Charts.css("c1"), pts: tr.map((r) => ({ x: r.depth_um, y: 10 * r.dwell_us })) }] });
  if (d.ledger_h) {
    const names = { mill: "Milling", overhead: "Per-slice overhead", image: "Imaging", eds: "EDS", autofocus: "Autofocus" };
    const keys = Object.keys(names).filter((k) => k in d.ledger_h);
    const colors = Object.fromEntries(keys.map((k, i) => [k, Charts.SERIES[[4, 5, 1, 0, 2][i] ?? i]]));
    Charts.stacked($("#ag-ledger"), { keys, colors, names, f: (v) => `${v.toFixed(2)} h`,
      rows: [{ label: "This run", total: keys.reduce((a, k) => a + d.ledger_h[k], 0), parts: keys.map((k) => ({ key: k, value: d.ledger_h[k] })) }] });
  }
  $("#ag-calls").textContent = d.autoscript_calls ? `AutoScript calls recorded (dry-run): ${Object.entries(d.autoscript_calls).map(([k, n]) => `${k} ×${n}`).join(" · ")}` : "";
  const log = $("#ag-log"); log.replaceChildren();
  for (const x of (d.decisions || []).slice().reverse())
    log.append(el("div", {}, el("span", { class: "s" }, `slice ${x.slice}`), el("span", { class: `a ${x.action}` }, x.action), el("span", {}, x.reason)));
  if (!log.children.length) log.append(el("div", { class: "s" }, "No decisions yet."));
}
PAGES.acquire = async () => {
  const p = S.project;
  fillSelect("#ag-clean", p.volumes.filter((v) => v.kind === "intensity" && !v.id.startsWith("live/")).map((v) => v.id), /n2v|denoised/);
  fillSelect("#ag-labels", p.volumes.filter((v) => v.kind === "labels" && !v.id.startsWith("live/")).map((v) => v.id), /phases$/);
  fillSelect("#ag-model", p.models || [], /rf/);
  if (!AG.job) { const j = Shell.jobs.find((x) => x.kind === "agent"); if (j) AG.job = j.id; }
  if (AG.job) followAgent();
  else ["#ag-prec", "#ag-ctrl", "#ag-ledger"].forEach((s) => Charts.empty($(s), "Start a twin run to see it here."));

  const lv = $("#acq-live"); lv.replaceChildren();
  try {
    const L = await (await api("/api/live")).json();
    if (L.status === "none") lv.append(el("div", { class: "empty" }, "No live acquisition in this project yet."));
    else {
      const running = L.status === "running" && L.age_s < 120;
      lv.append(el("div", { class: "tiles small" }, tile("Status", running ? "Live" : "Finished"), tile("Slices", L.slices.toLocaleString()),
        tile("Milled", L.depth_um.toFixed(2), "µm"), tile("Verdict", L.can_stop ? "Can stop" : "Keep milling")));
      const t = el("table", { class: "t" }, el("thead", {}, el("tr", {}, el("th", {}, "Phase"), el("th", {}, "Fraction"), el("th", {}, "Error (95 %)"))));
      const tb = el("tbody"); for (const ph of L.phases || []) tb.append(el("tr", {}, el("td", {}, ph.name), el("td", {}, F.pct(ph.fraction)),
        el("td", { class: ph.target_reached ? "ok" : "no" }, `±${(100 * ph.rel_error_95).toFixed(1)} %`)));
      t.append(tb); lv.append(t);
    }
  } catch (e) { lv.append(el("div", { class: "empty" }, "Live status unavailable.")); }

  const ab = await rows(latest(T.ablation));
  if (ab) {
    $("#ag-ab-note").textContent = "Three policies on the twin of this sample, to the same ±20 % target on the dark and grey phases.";
    const names = { t_mill: "Milling", t_overhead: "Per-slice overhead", t_image: "Imaging", t_eds: "EDS", t_autofocus: "Autofocus" };
    const keys = Object.keys(names).filter((k) => ab.columns.includes(k));
    const colors = Object.fromEntries(keys.map((k, i) => [k, Charts.SERIES[[4, 5, 1, 0, 2][i] ?? i]]));
    Charts.stacked($("#ag-ablation"), { keys, colors, names, f: (v) => `${v.toFixed(2)} h`,
      rows: ab.map((r) => ({ label: r.policy.charAt(0).toUpperCase() + r.policy.slice(1), total: r.time_h, parts: keys.map((k) => ({ key: k, value: r[k] })) })) });
  } else { $("#ag-ab-note").textContent = ""; Charts.empty($("#ag-ablation"), "Run examples/self_driving.py to compare policies."); }
};

// ------------------------------------------------------------------ recipe
$("#recipe-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#recipe-text").textContent); toast("Recipe copied"); } catch (e) { toast("Copy failed", " Select the YAML below and copy it.", "err"); }
});
PAGES.recipe = async () => {
  const [txt, doc] = await Promise.all([api("/api/recipe").then((r) => r.text()), api("/api/recipe.json").then((r) => r.json())]);
  $("#recipe-text").textContent = txt;
  const ol = $("#recipe-steps"); ol.replaceChildren();
  const steps = doc.steps || [];
  if (!steps.length) ol.append(el("div", { class: "empty" }, "No steps yet. Every operation you run in Studio is added here."));
  for (const s of steps) {
    const ins = Object.entries(s.inputs || {}).map(([k, v]) => `${k}=${v}`).join(", ");
    const ps = Object.entries(s.params || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
    ol.append(el("li", {}, el("span", { class: "op" }, s.op), el("span", { class: "p" }, [ins, ps].filter(Boolean).join(" · ") || "default parameters")));
  }
};
