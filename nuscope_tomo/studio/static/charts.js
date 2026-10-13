// NUScope-TOMO Studio — small SVG chart kit (no dependencies, works offline).
"use strict";

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => `var(--${v})`;
  const SERIES = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"].map(css);
  const PHASE = { pore: css("c1"), "dark phase": css("c2"), "grey phase": css("c3"), "bright phase": css("c4") };

  const fmt = {
    pct: (v, d = 1) => `${(100 * v).toFixed(d)} %`,
    num: (v, d = 2) => (Math.abs(v) >= 1e4 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : (+v).toFixed(d)),
    auto: (v) => (v == null || !isFinite(v) ? "–" : Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
      : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3)),
  };

  // ------------------------------------------------------------ scales / ticks
  function niceStep(span, n) {
    const raw = span / Math.max(1, n), p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
    return (f >= 5 ? 10 : f >= 2 ? 5 : f >= 1 ? 2 : 1) * p;
  }
  function ticks(lo, hi, n = 5) {
    if (!(hi > lo)) return [lo];
    const s = niceStep(hi - lo, n), out = [];
    for (let v = Math.ceil(lo / s) * s; v <= hi + s * 1e-9; v += s) out.push(+v.toPrecision(12));
    return out;
  }
  function nice(lo, hi, n = 5) {
    if (!(hi > lo)) return [lo - 1, hi + 1];
    const s = niceStep(hi - lo, n);
    return [Math.floor(lo / s) * s, Math.ceil(hi / s) * s];
  }
  function linear([d0, d1], [r0, r1]) {
    const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.domain = [d0, d1]; f.ticks = (n) => ticks(d0, d1, n); f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
    return f;
  }
  function log([d0, d1], [r0, r1]) {
    const L0 = Math.log10(d0), L1 = Math.log10(d1);
    const f = (v) => r0 + ((Math.log10(Math.max(v, 1e-12)) - L0) / (L1 - L0 || 1)) * (r1 - r0);
    f.domain = [d0, d1];
    f.ticks = () => {
      const out = [];
      for (let e = Math.floor(L0); e <= Math.ceil(L1); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v); }
      return out;
    };
    f.invert = (p) => 10 ** (L0 + ((p - r0) / (r1 - r0)) * (L1 - L0));
    return f;
  }
  function band(keys, [r0, r1], pad = 0.2) {
    const step = (r1 - r0) / Math.max(1, keys.length), bw = step * (1 - pad);
    const f = (k) => r0 + keys.indexOf(k) * step + (step - bw) / 2;
    f.bw = bw; f.step = step; return f;
  }

  // ------------------------------------------------------------ svg helpers
  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, s, { anchor = "start", cls = null, dy = "0.35em", color = null, rotate = null } = {}) {
    const t = el("text", { x, y, dy, "text-anchor": anchor, class: cls }, parent);
    if (color) t.style.fill = color;
    if (rotate) t.setAttribute("transform", `rotate(${rotate} ${x} ${y})`);
    t.textContent = s;
    return t;
  }
  let measurer = null;
  function measure(strs) {
    if (!measurer) {
      measurer = el("svg", { width: 0, height: 0, class: "chart" });
      measurer.style.position = "absolute"; measurer.style.visibility = "hidden";
      document.body.appendChild(measurer);
    }
    const t = el("text", {}, measurer);
    t.style.fontSize = "11px";
    let m = 0;
    for (const s of strs) { t.textContent = s; m = Math.max(m, t.getComputedTextLength()); }
    t.remove();
    return Math.ceil(m);
  }
  function frame(host, h) {
    host.replaceChildren();
    const w = Math.max(260, host.clientWidth || 600);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, height: h, role: "img" }, host);
    return { svg, w, h };
  }
  function empty(host, msg) {
    host.replaceChildren();
    const d = document.createElement("div");
    d.className = "empty"; d.textContent = msg;
    host.appendChild(d);
  }
  function tip(host) {
    let t = host.querySelector(":scope > .tip");
    if (!t) { t = document.createElement("div"); t.className = "tip"; t.hidden = true; host.appendChild(t); }
    return {
      show(ev, head, rows) {
        t.replaceChildren();
        const h = document.createElement("div"); h.className = "h"; h.textContent = head; t.appendChild(h);
        for (const r of rows) {
          const d = document.createElement("div"); d.className = "r";
          if (r.color) { const i = document.createElement("i"); i.style.background = r.color; d.appendChild(i); }
          const n = document.createElement("span"); n.textContent = r.name;
          const b = document.createElement("b"); b.textContent = r.value;
          d.append(n, b); t.appendChild(d);
        }
        t.hidden = false;
        const box = host.getBoundingClientRect();
        const x = ev.clientX - box.left, y = ev.clientY - box.top;
        let left = x + 14; if (left + t.offsetWidth > box.width) left = x - t.offsetWidth - 14;
        t.style.left = `${Math.max(0, left)}px`;
        t.style.top = `${Math.max(0, Math.min(y - t.offsetHeight / 2, box.height - t.offsetHeight))}px`;
      },
      hide() { t.hidden = true; },
    };
  }
  function axisX(svg, x, y0, h, { fmt: f = fmt.auto, grid = true, n = 5, label = null, w = 0 } = {}) {
    const g = el("g", { class: grid ? "grid" : "axis" }, svg);
    for (const v of x.ticks(n)) {
      const px = x(v);
      if (grid) el("line", { x1: px, x2: px, y1: y0 - h, y2: y0 }, g);
      text(svg, px, y0 + 12, f(v), { anchor: "middle" });
    }
    if (!grid) el("line", { x1: x(x.domain[0]), x2: w, y1: y0, y2: y0 }, g);
    if (label) text(svg, (x(x.domain[0]) + x(x.domain[1])) / 2, y0 + 28, label, { anchor: "middle" });
  }
  function axisY(svg, y, x0, x1, { fmt: f = fmt.auto, n = 4, label = null } = {}) {
    const g = el("g", { class: "grid" }, svg);
    for (const v of y.ticks(n)) {
      const py = y(v);
      el("line", { x1: x0, x2: x1, y1: py, y2: py }, g);
      text(svg, x0 - 6, py, f(v), { anchor: "end" });
    }
    if (label) text(svg, x0 + 4, y(y.domain[1]) - 2, label, { dy: "-0.4em" });
  }
  function legend(host, items) {
    const d = document.createElement("div");
    d.className = "chart-legend";
    for (const it of items) {
      const s = document.createElement("span"), i = document.createElement("i");
      i.style.background = it.color;
      if (it.dashed) { i.style.background = "none"; i.style.borderTop = `2px dashed ${it.color}`; i.style.height = "0"; }
      s.append(i, document.createTextNode(it.name)); d.appendChild(s);
    }
    host.prepend(d);
  }

  // ------------------------------------------------------------ chart types
  // Horizontal bars with optional whiskers: rows [{label, value, lo, hi, lo2, hi2, color}]
  function bars(host, { rows, f = fmt.auto, axisFmt = f, height = null, ref = null, logX = false, x0 = 0 }) {
    if (!rows.length) return empty(host, "No data yet");
    const H = height || Math.max(120, 30 + rows.length * 38);
    const { svg, w } = frame(host, H);
    const ml = measure(rows.map((r) => r.label)) + 14, mr = measure(rows.map((r) => f(r.value))) + 16, mt = ref ? 20 : 6, mb = 24;
    const hi = Math.max(...rows.map((r) => Math.max(r.value, r.hi ?? -Infinity, r.hi2 ?? -Infinity)), ref ? ref.v : -Infinity);
    const x = logX ? log([Math.max(x0, 1e-9) || 1, hi * 1.08], [ml, w - mr]) : linear(nice(x0, hi, 5), [ml, w - mr]);
    const y = band(rows.map((r) => r.label), [mt, H - mb], 0.3);
    axisX(svg, x, H - mb, H - mb - mt, { fmt: axisFmt, n: Math.max(2, Math.floor((w - ml - mr) / 80)) });
    const t = tip(host);
    for (const r of rows) {
      const yc = y(r.label) + y.bw / 2, base = x(logX ? x.domain[0] : x0);
      text(svg, ml - 10, yc, r.label, { anchor: "end", cls: "strong" });
      const bh = Math.min(16, y.bw * (r.lo != null ? 0.45 : 0.8));
      const b = el("rect", { x: base, y: yc - bh / 2, width: Math.max(1, x(r.value) - base), height: bh, rx: 3 }, svg);
      b.style.fill = r.color || SERIES[1];
      const whisk = (lo, hi2, yy, color) => {
        if (lo == null) return;
        for (const [a1, a2, b1, b2] of [[x(lo), x(hi2), yy, yy], [x(lo), x(lo), yy - 4, yy + 4], [x(hi2), x(hi2), yy - 4, yy + 4]]) {
          const l = el("line", { x1: a1, x2: a2, y1: b1, y2: b2 }, svg); l.style.stroke = color; l.style.strokeWidth = 1.6;
        }
      };
      whisk(r.lo, r.hi, yc - bh / 2 - 6, css("text"));
      whisk(r.lo2, r.hi2, yc + bh / 2 + 6, css("muted"));
      text(svg, x(Math.max(r.value, r.hi ?? -Infinity, r.hi2 ?? -Infinity)) + 8, yc, f(r.value), { cls: "strong" });
      const hit = el("rect", { x: 0, y: y(r.label), width: w, height: y.bw, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (ev) => t.show(ev, r.label, r.tip || [{ name: "Value", value: f(r.value) }]));
      hit.addEventListener("pointerleave", t.hide);
    }
    if (ref) {
      const l = el("line", { x1: x(ref.v), x2: x(ref.v), y1: mt - 4, y2: H - mb, class: "ref" }, svg);
      const lw = measure([ref.label]);
      text(svg, Math.min(Math.max(ml, x(ref.v) - lw / 2), w - lw), mt - 12, ref.label, { cls: "strong" });
      void l;
    }
  }

  // Grouped horizontal bars: groups [{label, items:[{key, value, color}]}]
  function grouped(host, { groups, keys, colors, f = fmt.auto, logX = false, height = null }) {
    if (!groups.length) return empty(host, "No data yet");
    const H = height || Math.max(150, 30 + groups.length * 64);
    const { svg, w } = frame(host, H);
    const vals = groups.flatMap((g) => g.items.map((i) => i.value));
    const ml = measure(groups.map((g) => g.label)) + 14, mr = measure(vals.map(f)) + 12, mt = 4, mb = 24;
    const x = logX ? log([1, Math.max(...vals) * 1.1], [ml, w - mr]) : linear(nice(0, Math.max(...vals)), [ml, w - mr]);
    const y = band(groups.map((g) => g.label), [mt, H - mb], 0.22);
    axisX(svg, x, H - mb, H - mb - mt, { fmt: (v) => fmt.auto(v), n: 5 });
    const t = tip(host);
    for (const g of groups) {
      text(svg, ml - 10, y(g.label) + y.bw / 2, g.label, { anchor: "end", cls: "strong" });
      const yi = band(keys, [y(g.label), y(g.label) + y.bw], 0.15);
      for (const it of g.items) {
        const b = el("rect", { x: x(x.domain[0]), y: yi(it.key), width: Math.max(1, x(it.value) - x(x.domain[0])), height: yi.bw, rx: 2 }, svg);
        b.style.fill = colors[it.key];
        text(svg, x(it.value) + 5, yi(it.key) + yi.bw / 2, f(it.value));
        b.addEventListener("pointermove", (ev) => t.show(ev, `${g.label} · ${it.key}`, it.tip || [{ name: "Value", value: f(it.value) }]));
        b.addEventListener("pointerleave", t.hide);
      }
    }
  }

  // Lines: series [{name, color, pts:[{x,y}]}]
  function lines(host, { series, xLabel = null, yLabel = null, xf = fmt.auto, yf = fmt.auto, logY = false, refY = null, height = 230, step = false, endLabels = true, yMin = null }) {
    series = series.filter((s) => s.pts.length);
    if (!series.length) return empty(host, "No data yet");
    const { svg, w, h } = frame(host, height);
    const all = series.flatMap((s) => s.pts);
    const ends = series.filter((s) => !s.dots).map((s) => `${s.name} ${yf(s.pts[s.pts.length - 1].y)}`);
    const ml = measure(["00000"]) + 10, mr = endLabels ? measure(ends) + 14 : 10, mt = 14, mb = xLabel ? 38 : 24;
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y).concat(refY ? [refY.v] : []);
    const x = linear([Math.min(...xs), Math.max(...xs)], [ml, w - mr]);
    const y = logY ? log([Math.max(Math.min(...ys) * 0.8, 1e-6), Math.max(...ys) * 1.2], [h - mb, mt])
      : linear(nice(yMin ?? Math.min(0, Math.min(...ys)), Math.max(...ys)), [h - mb, mt]);
    axisY(svg, y, ml, w - mr, { fmt: yf, label: yLabel });
    axisX(svg, x, h - mb, 0, { fmt: xf, grid: false, label: xLabel, w: w - mr, n: Math.max(3, Math.floor((w - ml) / 80)) });
    if (refY) {
      el("line", { x1: ml, x2: w - mr, y1: y(refY.v), y2: y(refY.v), class: "ref" }, svg);
      if (refY.label) text(svg, w - mr - 4, y(refY.v) - 7, refY.label, { anchor: "end" });
    }
    const labels = [];
    for (const s of series) {
      if (s.dots) {
        for (const p of s.pts) { const c = el("circle", { cx: x(p.x), cy: y(p.y), r: 3.2 }, svg); c.style.fill = s.color; }
        continue;
      }
      let d = "";
      s.pts.forEach((p, i) => {
        const X = x(p.x), Y = y(p.y);
        if (i === 0) d += `M${X},${Y}`;
        else if (step) d += `H${X}V${Y}`;
        else d += `L${X},${Y}`;
      });
      const path = el("path", { d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      path.style.stroke = s.color;
      const e = s.pts[s.pts.length - 1];
      const c = el("circle", { cx: x(e.x), cy: y(e.y), r: 3 }, svg); c.style.fill = s.color;
      labels.push({ y: y(e.y), x: x(e.x) + 8, s: `${s.name} ${yf(e.y)}`, color: s.color });
    }
    if (endLabels) {
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
      for (const l of labels) text(svg, Math.min(l.x, w - mr + 8), l.y, l.s, { color: l.color });
    }
    const t = tip(host);
    const rule = el("line", { y1: mt, y2: h - mb, class: "ref" }, svg); rule.style.display = "none";
    const hit = el("rect", { x: ml, y: mt, width: w - ml - mr, height: h - mb - mt, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const box = svg.getBoundingClientRect(), px = ((ev.clientX - box.left) / box.width) * w, xv = x.invert(px);
      const rows = series.filter((s) => !s.dots).map((s) => {
        let best = s.pts[0];
        for (const p of s.pts) if (Math.abs(p.x - xv) < Math.abs(best.x - xv)) best = p;
        return { color: s.color, name: s.name, value: yf(best.y), x: best.x };
      });
      rule.style.display = ""; rule.setAttribute("x1", x(rows[0].x)); rule.setAttribute("x2", x(rows[0].x));
      t.show(ev, `${xLabel || "x"} ${xf(rows[0].x)}`, rows);
    });
    hit.addEventListener("pointerleave", () => { t.hide(); rule.style.display = "none"; });
  }

  // Histogram of values
  function hist(host, { values, xLabel = null, color = css("c2"), bins = 30, logX = false }) {
    values = values.filter((v) => isFinite(v) && (!logX || v > 0));
    if (!values.length) return empty(host, "No data yet");
    const { svg, w, h } = frame(host, 220);
    const lo = Math.min(...values), hi = Math.max(...values);
    const edges = [];
    for (let i = 0; i <= bins; i++) edges.push(logX ? 10 ** (Math.log10(lo) + (i / bins) * (Math.log10(hi) - Math.log10(lo))) : lo + (i / bins) * (hi - lo || 1));
    const counts = new Array(bins).fill(0);
    for (const v of values) {
      let k = logX ? Math.floor(((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1)) * bins) : Math.floor(((v - lo) / (hi - lo || 1)) * bins);
      counts[Math.min(bins - 1, Math.max(0, k))]++;
    }
    const ml = measure([String(Math.max(...counts))]) + 14, mr = 10, mt = 10, mb = xLabel ? 38 : 24;
    const x = logX ? log([lo, hi], [ml, w - mr]) : linear([lo, hi], [ml, w - mr]);
    const y = linear(nice(0, Math.max(...counts)), [h - mb, mt]);
    axisY(svg, y, ml, w - mr, { fmt: (v) => v.toFixed(0) });
    axisX(svg, x, h - mb, 0, { grid: false, label: xLabel, w: w - mr });
    const t = tip(host);
    counts.forEach((c, i) => {
      const r = el("rect", { x: x(edges[i]) + 0.5, width: Math.max(0.5, x(edges[i + 1]) - x(edges[i]) - 1), y: y(c), height: y(0) - y(c), rx: 1.5 }, svg);
      r.style.fill = color;
      r.addEventListener("pointermove", (ev) => t.show(ev, `${fmt.auto(edges[i])} – ${fmt.auto(edges[i + 1])}`, [{ name: "Count", value: c.toLocaleString() }]));
      r.addEventListener("pointerleave", t.hide);
    });
  }

  // Scatter with nearest-point tooltip: pts [{x, y, tip:[...], head}]
  function scatter(host, { pts, xLabel, yLabel, logX = false, color = css("c2"), height = 230 }) {
    pts = pts.filter((p) => isFinite(p.x) && isFinite(p.y) && (!logX || p.x > 0));
    if (!pts.length) return empty(host, "No data yet");
    const { svg, w, h } = frame(host, height);
    const ml = measure(["0.000"]) + 12, mr = 10, mt = 14, mb = 38;
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const x = logX ? log([Math.min(...xs) * 0.9, Math.max(...xs) * 1.1], [ml, w - mr]) : linear(nice(Math.min(...xs), Math.max(...xs)), [ml, w - mr]);
    const y = linear(nice(Math.min(...ys), Math.max(...ys)), [h - mb, mt]);
    axisY(svg, y, ml, w - mr, { label: yLabel });
    axisX(svg, x, h - mb, 0, { grid: false, label: xLabel, w: w - mr, fmt: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)) });
    const g = el("g", {}, svg);
    for (const p of pts) { const c = el("circle", { cx: x(p.x), cy: y(p.y), r: 2.3 }, g); c.style.fill = color; c.style.fillOpacity = 0.45; }
    const hl = el("circle", { r: 5, fill: "none", "stroke-width": 1.5 }, svg); hl.style.stroke = css("text"); hl.style.display = "none";
    const t = tip(host);
    const hit = el("rect", { x: ml, y: mt, width: w - ml - mr, height: h - mb - mt, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const box = svg.getBoundingClientRect(), px = ((ev.clientX - box.left) / box.width) * w, py = ((ev.clientY - box.top) / box.height) * h;
      let best = null, bd = Infinity;
      for (const p of pts) { const d = (x(p.x) - px) ** 2 + (y(p.y) - py) ** 2; if (d < bd) { bd = d; best = p; } }
      hl.style.display = ""; hl.setAttribute("cx", x(best.x)); hl.setAttribute("cy", y(best.y));
      t.show(ev, best.head || "", best.tip || []);
    });
    hit.addEventListener("pointerleave", () => { t.hide(); hl.style.display = "none"; });
  }

  // Heat map: cells [{col, row, v, label, tip, outline}]
  function heat(host, { cells, cols, rows, xLabel, yLabel, color = css("c5"), height = 230 }) {
    if (!cells.length) return empty(host, "No data yet");
    const { svg, w, h } = frame(host, height);
    const ml = measure(rows.map(String)) + 30, mr = 4, mt = 4, mb = 38;
    const x = band(cols, [ml, w - mr], 0.06), y = band(rows, [mt, h - mb], 0.06);
    const vmax = Math.max(...cells.map((c) => c.v)) || 1;
    const t = tip(host);
    for (const c of cells) {
      const r = el("rect", { x: x(c.col), y: y(c.row), width: x.bw, height: y.bw, rx: 3 }, svg);
      r.style.fill = color; r.style.fillOpacity = 0.08 + 0.82 * (c.v / vmax);
      if (c.outline) { r.style.stroke = css("text"); r.style.strokeWidth = 2.5; }
      text(svg, x(c.col) + x.bw / 2, y(c.row) + y.bw / 2, c.label, { anchor: "middle", cls: "strong" }).style.pointerEvents = "none";
      r.addEventListener("pointermove", (ev) => t.show(ev, c.head || "", c.tip || []));
      r.addEventListener("pointerleave", t.hide);
    }
    for (const c of cols) text(svg, x(c) + x.bw / 2, h - mb + 12, String(c), { anchor: "middle" });
    for (const r of rows) text(svg, ml - 6, y(r) + y.bw / 2, String(r), { anchor: "end" });
    if (xLabel) text(svg, (ml + w) / 2, h - 8, xLabel, { anchor: "middle" });
    if (yLabel) text(svg, 10, (mt + h - mb) / 2, yLabel, { anchor: "middle", rotate: -90 });
  }

  // Dot plot: rows of categories, each with dots per series; extra right column text
  function dots(host, { cats, series, get, f = fmt.auto, right = null, rightHead = null, zero = true, emph = null }) {
    if (!cats.length) return empty(host, "No data yet");
    const rh = 34, mt = 4, mb = 26, H = mt + cats.length * rh + mb;
    const { svg, w } = frame(host, H);
    const vals = cats.flatMap((c) => series.map((s) => get(c, s.name))).filter((v) => v != null);
    const ml = measure(cats) + 14, mr = rightHead ? measure([rightHead]) + 18 : 10;
    const lo = zero ? 0 : Math.min(...vals) * 0.95;
    const x = linear(nice(lo, Math.max(...vals) * 1.03), [ml, w - mr - 8]);
    const y = band(cats, [mt, H - mb], 0.1);
    axisX(svg, x, H - mb, H - mb - mt, { fmt: f });
    if (rightHead) text(svg, w - 2, H - mb + 12, rightHead, { anchor: "end" });
    const t = tip(host);
    for (const c of cats) {
      const yc = y(c) + y.bw / 2;
      const lab = text(svg, ml - 10, yc, c, { anchor: "end", cls: "strong" });
      if (emph && emph(c)) lab.style.fontWeight = 700;
      const vs = series.map((s) => ({ s, v: get(c, s.name) })).filter((d) => d.v != null);
      if (vs.length > 1) {
        const l = el("line", { x1: x(Math.min(...vs.map((d) => d.v))), x2: x(Math.max(...vs.map((d) => d.v))), y1: yc, y2: yc, "stroke-width": 2 }, svg);
        l.style.stroke = css("line-2");
      }
      for (const d of vs) {
        const dot = el("circle", { cx: x(d.v), cy: yc, r: 6, "stroke-width": 1.5 }, svg);
        dot.style.fill = d.s.color; dot.style.stroke = css("surface");
      }
      if (right) text(svg, w - 2, yc, right(c), { anchor: "end" });
      const hit = el("rect", { x: 0, y: y(c), width: w, height: y.bw, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (ev) => t.show(ev, c, vs.map((d) => ({ color: d.s.color, name: d.s.name, value: f(d.v) }))));
      hit.addEventListener("pointerleave", t.hide);
    }
  }

  // Stacked horizontal bars: rows [{label, parts:[{key, value}], total}]
  function stacked(host, { rows, keys, colors, f = fmt.auto, names = {} }) {
    if (!rows.length) return empty(host, "No data yet");
    const H = Math.max(120, 30 + rows.length * 40);
    const { svg, w } = frame(host, H);
    const ml = measure(rows.map((r) => r.label)) + 14, mr = measure(rows.map((r) => f(r.total))) + 14, mt = 4, mb = 24;
    const x = linear(nice(0, Math.max(...rows.map((r) => r.total))), [ml, w - mr]);
    const y = band(rows.map((r) => r.label), [mt, H - mb], 0.3);
    axisX(svg, x, H - mb, H - mb - mt, { fmt: f, n: 5 });
    const t = tip(host);
    for (const r of rows) {
      text(svg, ml - 10, y(r.label) + y.bw / 2, r.label, { anchor: "end", cls: "strong" });
      let acc = 0;
      for (const k of keys) {
        const v = r.parts.find((p) => p.key === k)?.value || 0;
        const b = el("rect", { x: x(acc), y: y(r.label), width: Math.max(0, x(acc + v) - x(acc)), height: y.bw }, svg);
        b.style.fill = colors[k]; acc += v;
      }
      text(svg, x(r.total) + 6, y(r.label) + y.bw / 2, f(r.total), { cls: "strong" });
      const hit = el("rect", { x: 0, y: y(r.label), width: w, height: y.bw, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (ev) => t.show(ev, `${r.label} · ${f(r.total)}`,
        keys.map((k) => ({ color: colors[k], name: names[k] || k, value: f(r.parts.find((p) => p.key === k)?.value || 0) }))));
      hit.addEventListener("pointerleave", t.hide);
    }
    legend(host, keys.map((k) => ({ name: names[k] || k, color: colors[k] })));
  }

  return { bars, grouped, lines, hist, scatter, heat, dots, stacked, legend, empty, fmt, SERIES, PHASE, css };
})();
