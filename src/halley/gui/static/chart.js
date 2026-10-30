// Line charts for Halley's GUI.
//
// lineChart(spec) returns a <figure> that draws one or more series over the months of a run:
// hairline grid and axes, 2px lines, a crosshair that snaps to the nearest month and lists every
// series there, a legend for two or more series (a button per series that hides or shows it,
// each keeping its colour), and a table view of the same numbers. The figure redraws itself
// when its width changes. Every label is set as text, never parsed as HTML.
//
// spec = {
//   title: string,            // what is plotted; a single series needs nothing else
//   unit: string,             // the values' unit, shown above the axis
//   x: number[],              // the months, ascending
//   xLabel: string,           // "month t", by default
//   series: [{ label, values, slot }],  // values: numbers, or strings/null where there is none
// }
// `slot` is the categorical colour 1 to 8; the caller assigns it per entity so that a series
// keeps its colour when others come and go.

const SVG_NS = "http://www.w3.org/2000/svg";
const PLOT_HEIGHT = 260;
const PAD = { top: 12, right: 16, bottom: 34, left: 12 };

function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function isNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

// A value as the page shows it: a finite number in full (the shortest text that reads back as
// the same float, as Python's repr), anything else as the engine gave it.
export function formatValue(v) {
  if (isNumber(v)) return String(v);
  if (v === null || v === undefined) return "n/a";
  return String(v);
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  const step = norm > 5 ? 10 : norm > 2 ? 5 : norm > 1 ? 2 : 1;
  return step * magnitude;
}

// An axis from lo to hi with about `count` steps of 1, 2 or 5 times a power of ten. A spread of
// less than a billionth of the values' size is float noise, not a variation: it is drawn as the
// constant it is, with a padded axis, rather than stretched over the whole height.
function niceScale(lo, hi, count) {
  const size = Math.max(Math.abs(lo), Math.abs(hi));
  if (hi - lo <= size * 1e-9) lo = hi = (lo + hi) / 2;
  if (!(hi > lo)) {
    const pad = lo === 0 ? 1 : Math.abs(lo) * 0.05;
    lo -= pad;
    hi += pad;
  }
  const step = niceStep(hi - lo, count);
  const start = Math.floor(lo / step + 1e-9) * step;
  const end = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  const n = Math.round((end - start) / step);
  for (let i = 0; i <= n; i++) ticks.push(Number((start + i * step).toPrecision(12)));
  return { lo: start, hi: end, step, ticks };
}

function monthTicks(lo, hi) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  // months are whole: a step of at least one, in years from four years up, each tick counted
  // from the first rather than summed, so no float residue reaches a label
  const step = span >= 48 ? 12 * niceStep(span / 12, 6) : Math.max(1, niceStep(span, 6));
  const first = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let i = 0; first + i * step <= hi; i++) ticks.push(first + i * step);
  return ticks;
}

// Axis labels: compact from a thousand up (1.5K, 2M), with as many decimals as the step needs.
function tickFormatter(ticks, step) {
  const top = Math.max(...ticks.map(Math.abs));
  const [unit, suffix] =
    top >= 1e12 ? [1e12, "T"] : top >= 1e9 ? [1e9, "B"] : top >= 1e6 ? [1e6, "M"] : top >= 1e3 ? [1e3, "K"] : [1, ""];
  const scaled = step / unit;
  if (scaled < 1e-6) {
    // as many significant digits as tell one tick from the next, up to the 17 a double holds
    const digits = Math.min(17, Math.max(2, Math.ceil(Math.log10(top / step)) + 1));
    return (v) => (v === 0 ? "0" : v.toExponential(digits - 1));
  }
  const decimals = Math.min(6, Math.max(0, -Math.floor(Math.log10(scaled) + 1e-9)));
  return (v) => (v === 0 ? "0" : (v / unit).toFixed(decimals) + suffix);
}

function textWidth(text) {
  const canvas = textWidth.canvas || (textWidth.canvas = document.createElement("canvas"));
  const ctx = canvas.getContext("2d");
  ctx.font = "12px system-ui, -apple-system, 'Segoe UI', sans-serif";
  return ctx.measureText(text).width;
}

function nearestIndex(x, value) {
  let lo = 0;
  let hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= value) lo = mid;
    else hi = mid;
  }
  return Math.abs(x[hi] - value) < Math.abs(x[lo] - value) ? hi : lo;
}

export function lineChart(spec) {
  const series = spec.series.map((s, i) => ({ ...s, slot: s.slot || (i % 8) + 1, hidden: false }));
  const x = spec.x;
  const xLabel = spec.xLabel || "month t";
  const figure = el("figure", { class: "chart", tabindex: "0" });
  const head = el("div", { class: "chart-head" });
  const title = el("figcaption", { class: "chart-title" }, el("span", { text: spec.title || "" }));
  if (spec.unit) title.append(el("span", { class: "chart-unit", text: spec.unit }));
  const viewChart = el("button", { type: "button", class: "seg on", "aria-pressed": "true", text: "Chart" });
  const viewTable = el("button", { type: "button", class: "seg", "aria-pressed": "false", text: "Table" });
  head.append(title, el("div", { class: "segmented", role: "group", "aria-label": "View" }, viewChart, viewTable));
  figure.append(head);

  const legend = el("div", { class: "legend" });
  if (series.length >= 2) {
    for (const s of series) {
      const button = el(
        "button",
        { type: "button", class: "legend-item", "aria-pressed": "true", title: "Hide or show this series" },
        el("span", { class: `key s${s.slot}` }),
        el("span", { text: s.label }),
      );
      button.addEventListener("click", () => {
        if (!s.hidden && series.filter((o) => !o.hidden).length === 1) return; // keep one
        s.hidden = !s.hidden;
        button.setAttribute("aria-pressed", String(!s.hidden));
        button.classList.toggle("off", s.hidden);
        draw();
      });
      legend.append(button);
    }
    figure.append(legend);
  }

  const plot = el("div", { class: "chart-plot" });
  const tooltip = el("div", { class: "tooltip", role: "status", hidden: true });
  plot.append(tooltip);
  const tableHost = el("div", { class: "chart-table", hidden: true });
  figure.append(plot, tableHost);

  let index = null; // the month the crosshair is on, as an index into x
  let geometry = null;

  function visible() {
    return series.filter((s) => !s.hidden);
  }

  function draw() {
    plot.querySelector("svg")?.remove();
    const width = Math.max(320, Math.floor(plot.clientWidth || 640));
    const shown = visible();
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of shown) {
      for (const v of s.values) {
        if (!isNumber(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    plot.querySelector(".empty")?.remove();
    if (x.length === 0 || lo > hi) {
      plot.append(el("p", { class: "empty", text: "No values to plot." }));
      return;
    }
    if (lo > 0 && lo <= 0.5 * hi) lo = 0; // a magnitude near its baseline: show the baseline
    if (hi < 0 && hi >= 0.5 * lo) hi = 0;
    const y = niceScale(lo, hi, 5);
    const fmt = tickFormatter(y.ticks, y.step);
    const labelWidth = Math.max(...y.ticks.map((t) => textWidth(fmt(t))));
    const left = PAD.left + Math.ceil(labelWidth) + 8;
    const plotW = width - left - PAD.right;
    const height = PAD.top + PLOT_HEIGHT + PAD.bottom;
    const x0 = x[0];
    const x1 = x[x.length - 1];
    const sx = (t) => left + (x1 === x0 ? plotW / 2 : ((t - x0) / (x1 - x0)) * plotW);
    const sy = (v) => PAD.top + PLOT_HEIGHT - ((v - y.lo) / (y.hi - y.lo)) * PLOT_HEIGHT;
    geometry = { left, plotW, sx, sy, width };

    const root = svg("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width,
      height,
      role: "img",
      "aria-label": `${spec.title || "Chart"}: ${shown.length} series over ${x.length} months; the table view lists every value`,
    });
    const grid = svg("g", { class: "grid" });
    for (const t of y.ticks) {
      const py = Math.round(sy(t)) + 0.5;
      grid.append(svg("line", { x1: left, x2: left + plotW, y1: py, y2: py, class: t === 0 ? "baseline" : "" }));
      const label = svg("text", { x: left - 8, y: py, class: "tick", "text-anchor": "end", "dominant-baseline": "middle" });
      label.textContent = fmt(t);
      grid.append(label);
    }
    for (const t of monthTicks(x0, x1)) {
      const px = Math.round(sx(t)) + 0.5;
      grid.append(svg("line", { x1: px, x2: px, y1: PAD.top + PLOT_HEIGHT, y2: PAD.top + PLOT_HEIGHT + 4, class: "axis" }));
      const label = svg("text", { x: px, y: PAD.top + PLOT_HEIGHT + 16, class: "tick", "text-anchor": "middle" });
      label.textContent = String(t);
      grid.append(label);
    }
    const axisTitle = svg("text", { x: left + plotW, y: height - 2, class: "tick", "text-anchor": "end" });
    axisTitle.textContent = xLabel;
    grid.append(axisTitle);
    root.append(grid);

    const lines = svg("g", { class: "lines" });
    for (const s of shown) {
      let d = "";
      let pen = false;
      s.values.forEach((v, i) => {
        if (!isNumber(v) || i >= x.length) {
          pen = false;
          return;
        }
        d += `${pen ? "L" : "M"}${sx(x[i]).toFixed(2)},${sy(v).toFixed(2)}`;
        pen = true;
      });
      if (x.length === 1 && isNumber(s.values[0])) {
        lines.append(svg("circle", { cx: sx(x[0]), cy: sy(s.values[0]), r: 4, class: `dot s${s.slot}` }));
      } else if (d) {
        lines.append(svg("path", { d, class: `line s${s.slot}` }));
      }
    }
    root.append(lines);

    const cross = svg("g", { class: "crosshair", visibility: "hidden" });
    cross.append(svg("line", { y1: PAD.top, y2: PAD.top + PLOT_HEIGHT }));
    for (const s of shown) cross.append(svg("circle", { r: 4, class: `dot s${s.slot}`, "data-slot": s.slot }));
    root.append(cross);

    const hit = svg("rect", { x: left, y: PAD.top, width: plotW, height: PLOT_HEIGHT, class: "hit" });
    hit.addEventListener("pointermove", (event) => {
      const box = root.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const t = x0 + ((px - left) / plotW) * (x1 - x0);
      show(nearestIndex(x, t));
    });
    hit.addEventListener("pointerleave", hide);
    root.append(hit);
    plot.append(root);
    if (index !== null) show(index);
  }

  function show(i) {
    index = Math.max(0, Math.min(x.length - 1, i));
    const root = plot.querySelector("svg");
    if (!root || !geometry) return;
    const cross = root.querySelector(".crosshair");
    const px = geometry.sx(x[index]);
    const line = cross.querySelector("line");
    line.setAttribute("x1", px);
    line.setAttribute("x2", px);
    const shown = visible();
    cross.querySelectorAll("circle").forEach((dot, k) => {
      const v = shown[k]?.values[index];
      if (isNumber(v)) {
        dot.setAttribute("cx", px);
        dot.setAttribute("cy", geometry.sy(v));
        dot.setAttribute("visibility", "visible");
      } else dot.setAttribute("visibility", "hidden");
    });
    cross.setAttribute("visibility", "visible");
    tooltip.replaceChildren(el("div", { class: "tooltip-head", text: `${xLabel} = ${x[index]}` }));
    for (const s of shown) {
      tooltip.append(
        el(
          "div",
          { class: "tooltip-row" },
          el("span", { class: `key s${s.slot}` }),
          el("strong", { class: "num", text: formatValue(s.values[index]) }),
          series.length > 1 ? el("span", { class: "muted", text: s.label }) : null,
        ),
      );
    }
    tooltip.hidden = false;
    const scale = root.getBoundingClientRect().width / geometry.width || 1;
    const room = plot.clientWidth;
    const tipWidth = tooltip.offsetWidth;
    let leftPx = px * scale + 14;
    if (leftPx + tipWidth > room) leftPx = px * scale - tipWidth - 14;
    tooltip.style.left = `${Math.max(0, leftPx)}px`;
    tooltip.style.top = `${PAD.top * scale}px`;
  }

  function hide() {
    tooltip.hidden = true;
    plot.querySelector(".crosshair")?.setAttribute("visibility", "hidden");
  }

  figure.addEventListener("keydown", (event) => {
    if (tableHost.hidden === false) return;
    const moves = { ArrowLeft: -1, ArrowRight: 1, PageUp: -12, PageDown: 12 };
    if (event.key in moves) show((index ?? 0) + moves[event.key]);
    else if (event.key === "Home") show(0);
    else if (event.key === "End") show(x.length - 1);
    else if (event.key === "Escape") hide();
    else return;
    event.preventDefault();
  });
  figure.addEventListener("focus", () => {
    if (index === null) show(0);
  });
  figure.addEventListener("blur", hide);

  function table() {
    const head = el("tr", {}, el("th", { scope: "col", text: xLabel }), series.map((s) => el("th", { scope: "col", text: s.label })));
    const rows = x.map((t, i) =>
      el("tr", {}, el("th", { scope: "row", class: "num", text: String(t) }), series.map((s) => el("td", { class: "num", text: formatValue(s.values[i]) }))),
    );
    return el("table", { class: "data" }, el("thead", {}, head), el("tbody", {}, rows));
  }

  viewChart.addEventListener("click", () => setView(false));
  viewTable.addEventListener("click", () => setView(true));
  function setView(asTable) {
    viewChart.classList.toggle("on", !asTable);
    viewTable.classList.toggle("on", asTable);
    viewChart.setAttribute("aria-pressed", String(!asTable));
    viewTable.setAttribute("aria-pressed", String(asTable));
    plot.hidden = asTable;
    legend.hidden = asTable;
    tableHost.hidden = !asTable;
    if (asTable && !tableHost.firstChild) tableHost.append(table());
    if (!asTable) draw();
  }

  // Draw once the figure is in the page, then again whenever its width changes. The first draw
  // does not wait for a resize notification: a page that is not being painted (a background
  // window) gets none, and the chart would stay empty.
  let lastWidth = 0;
  const redraw = () => {
    if (!plot.isConnected || plot.hidden) return;
    const width = Math.floor(plot.clientWidth);
    if (width === 0 || (width === lastWidth && plot.querySelector("svg"))) return;
    lastWidth = width;
    draw();
  };
  new ResizeObserver(redraw).observe(plot);
  let tries = 0;
  const first = () => {
    if (plot.isConnected) redraw();
    else if (tries++ < 50) setTimeout(first, 20);
  };
  setTimeout(first, 0);
  return figure;
}
