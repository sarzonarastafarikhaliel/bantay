"""Renders models/classifier_metrics.json into a single self-contained HTML
report of every classifier visualisation Chapter 4 needs.

Self-contained on purpose: the data is inlined into the page rather than fetched,
so the file opens from disk (file:// blocks fetch) and can be handed to a panel or
attached to the thesis without a server or a network round-trip. No CDN, no build
step - the charts are hand-drawn SVG.

    python tools/build_metrics_report.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "models/classifier_metrics.json"
OUT = ROOT / "Thesis Writing/BANTAY Classifier Metrics.html"

TEMPLATE = r"""<title>BANTAY Classifier Metrics</title>
<style>
  :root {
    color-scheme: light;
    --plane: #f9f9f7;  --surface: #fcfcfb;
    --ink: #0b0b0b;    --ink-2: #52514e;  --muted: #898781;
    --grid: #e1e0d9;   --axis: #c3c2b7;   --ring: rgba(11,11,11,.10);
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
    --good:#0ca30c; --critical:#d03b3b;
    --seq-0:#cde2fb; --seq-1:#9ec5f4; --seq-2:#6da7ec; --seq-3:#3987e5;
    --seq-4:#256abf; --seq-5:#184f95; --seq-6:#0d366b;
    --seq-ink-hi:#fcfcfb; --seq-ink-lo:#0b0b0b;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --plane:#0d0d0d; --surface:#1a1a19;
      --ink:#ffffff;   --ink-2:#c3c2b7;  --muted:#898781;
      --grid:#2c2c2a;  --axis:#383835;   --ring: rgba(255,255,255,.10);
      --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#ffffff;   --ink-2:#c3c2b7;  --muted:#898781;
    --grid:#2c2c2a;  --axis:#383835;   --ring: rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  }

  body { background: var(--plane); color: var(--ink);
         font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 26px; letter-spacing: -.02em; margin: 0 0 6px; }
  h2 { font-size: 18px; letter-spacing: -.01em; margin: 0 0 4px; }
  .sub { color: var(--ink-2); margin: 0 0 28px; max-width: 78ch; }
  .note { color: var(--muted); font-size: 12.5px; margin: 10px 0 0; max-width: 88ch; }
  .card { background: var(--surface); border: 1px solid var(--ring);
          border-radius: 12px; padding: 22px 22px 18px; margin: 0 0 22px; }
  .card > .cap { color: var(--ink-2); font-size: 13px; margin: 2px 0 18px; max-width: 84ch; }
  .grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 18px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
  .scroll { overflow-x: auto; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
           gap: 14px; margin: 0 0 22px; }
  .tile { background: var(--surface); border: 1px solid var(--ring);
          border-radius: 12px; padding: 16px 18px; }
  .tile .k { color: var(--muted); font-size: 11.5px; text-transform: uppercase;
             letter-spacing: .06em; }
  .tile .v { font-size: 27px; letter-spacing: -.02em; margin-top: 4px; }
  .tile .d { color: var(--ink-2); font-size: 12px; margin-top: 2px; }

  .legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 0 0 14px;
            font-size: 12.5px; color: var(--ink-2); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }

  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 9px 10px; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
  thead th { color: var(--muted); font-weight: 600; font-size: 11.5px;
             text-transform: uppercase; letter-spacing: .05em; white-space: nowrap; }
  caption { caption-side: top; text-align: left; color: var(--ink-2);
            font-size: 13px; padding: 0 0 12px; }

  .ctl { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 16px; }
  .ctl label { color: var(--muted); font-size: 11.5px; text-transform: uppercase;
               letter-spacing: .06em; margin-right: 4px; }
  button.seg { font: inherit; font-size: 12.5px; color: var(--ink-2); cursor: pointer;
               background: transparent; border: 1px solid var(--ring);
               border-radius: 7px; padding: 5px 11px; }
  button.seg[aria-pressed="true"] { background: var(--s1); border-color: var(--s1);
                                    color: #fff; }
  svg { display: block; overflow: visible; }
  .axis-t { fill: var(--muted); font-size: 11px; }
  .lbl { fill: var(--ink-2); font-size: 11.5px; }
  .val { fill: var(--ink); font-size: 11.5px; font-variant-numeric: tabular-nums; }
  .gl { stroke: var(--grid); stroke-width: 1; }
  .ax { stroke: var(--axis); stroke-width: 1; }
  .hit { fill: transparent; cursor: default; }
  #tip { position: fixed; z-index: 50; pointer-events: none; opacity: 0;
         transition: opacity .09s; background: var(--surface); color: var(--ink);
         border: 1px solid var(--ring); border-radius: 8px; padding: 8px 11px;
         font-size: 12.5px; box-shadow: 0 6px 22px rgba(0,0,0,.16); max-width: 280px; }
  #tip b { font-weight: 620; }
  #tip .m { color: var(--ink-2); }
  @media print { .card { break-inside: avoid; } body { background: #fff; } }
</style>

<div class="wrap">
  <h1>BANTAY incident classifier &mdash; full metric comparison</h1>
  <p class="sub" id="sub"></p>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>The six metrics, per candidate</h2>
    <p class="cap">One panel per metric because the scales differ &mdash; eval loss is in nats
      and unbounded, the other five are proportions. Bars are sorted best-first within each
      panel, and each model keeps its colour across every chart on this page.</p>
    <div class="ctl">
      <label>Grain</label>
      <button class="seg" data-grain="type" aria-pressed="true">Incident type (35-way)</button>
      <button class="seg" data-grain="group" aria-pressed="false">Category group (7-way)</button>
      <label style="margin-left:14px">Averaging</label>
      <button class="seg" data-avg="weighted" aria-pressed="true">Weighted</button>
      <button class="seg" data-avg="macro" aria-pressed="false">Macro</button>
    </div>
    <div class="legend" id="legend-models"></div>
    <div class="grid3" id="panels"></div>
  </div>

  <div class="card">
    <h2>Full metric table</h2>
    <p class="cap">The same numbers as the panels above, both averagings side by side.
      Best value in each column is bold.</p>
    <div class="scroll"><table id="tbl"></table></div>
    <p class="note" id="tbl-note"></p>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>ROC curves</h2>
      <p class="cap">Micro-averaged one-vs-rest: every record &times; class cell is one binary
        decision, pooled into a single curve per model. The dashed diagonal is chance.</p>
      <div class="legend" id="legend-roc"></div>
      <div id="roc"></div>
    </div>
    <div class="card">
      <h2>Reliability (calibration)</h2>
      <p class="cap">Mean top-1 probability against the share actually correct, in probability
        bins. Points below the dashed diagonal are overconfident: the model asserts more
        certainty than its hit rate earns.</p>
      <div class="legend" id="legend-cal"></div>
      <div id="cal"></div>
    </div>
  </div>

  <div class="card">
    <h2>Per-group F1</h2>
    <p class="cap">F1 for each Category Group, per model. Darker is higher. This is the grain
      the dashboard's PNP-tier and KP-referability panels are read at.</p>
    <div id="heat"></div>
  </div>

  <div class="card">
    <h2>Category-group confusion</h2>
    <p class="cap">Rows are the gold group, columns the predicted group; the outlined diagonal
      is correct. Counts, not rates &mdash; several groups carry only a handful of dev-split
      records.</p>
    <div class="ctl" id="cm-ctl"><label>Model</label></div>
    <div id="cm"></div>
  </div>

  <div class="card">
    <h2>Confidence, correct vs. incorrect</h2>
    <p class="cap">Distribution of each model's top-1 probability, split by whether that
      prediction was right. A useful confidence signal would push the two distributions apart;
      overlapping mass at the high end is confidently-wrong output, which is what a
      confirm-or-correct workflow has to absorb.</p>
    <div class="legend">
      <span><i class="swatch" style="background:var(--good)"></i>Correct</span>
      <span><i class="swatch" style="background:var(--critical)"></i>Incorrect</span>
    </div>
    <div id="conf"></div>
  </div>

  <p class="note" id="method"></p>
</div>
<div id="tip" role="status" aria-live="polite"></div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const SERIES = ["--s1","--s2","--s3","--s4","--s5"];
const SEQ = ["--seq-0","--seq-1","--seq-2","--seq-3","--seq-4","--seq-5","--seq-6"];
// Colour follows the entity: fixed at load, by the model's identity, never by its
// rank inside whichever panel happens to be on screen.
const COLOR = {};
DATA.models.forEach((m, i) => COLOR[m.tag] = "var(" + SERIES[i % SERIES.length] + ")");

const pct = v => (v * 100).toFixed(1) + "%";
const f3  = v => v.toFixed(3);
const el  = (t, a, p) => { const n = document.createElementNS(NS, t);
  for (const k in (a || {})) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; };
const tip = document.getElementById("tip");
function bindTip(node, html) {
  const move = e => {
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX + 14, innerWidth - r.width - 10) + "px";
    tip.style.top  = Math.max(8, e.clientY - r.height - 12) + "px";
  };
  node.addEventListener("pointerenter", e => { tip.innerHTML = html;
                                               tip.style.opacity = 1; move(e); });
  node.addEventListener("pointermove", move);
  node.addEventListener("pointerleave", () => tip.style.opacity = 0);
}
const byDesc = (f) => DATA.models.slice().sort((a, b) => f(b) - f(a))[0];
const byAsc  = (f) => DATA.models.slice().sort((a, b) => f(a) - f(b))[0];

/* ---------- header ---------- */
const best = DATA.models[0];
document.getElementById("sub").textContent =
  "Accuracy, precision, recall, F1, AUC-ROC and cross-entropy eval loss for all " +
  DATA.models.length + " locally-run candidates, measured on the same " + DATA.n +
  "-record " + DATA.split + " split of the Barangay Anunas 2023 blotter corpus, at both " +
  "the 35-way incident-type grain and the 7-way Category Group grain.";

const bestGroup = byDesc(m => m.group.accuracy);
const bestLoss  = byAsc(m => m.type.eval_loss);
const bestAuc   = byDesc(m => m.type.auc_macro);
const tiles = [
  ["Best type accuracy", pct(best.type.accuracy), best.label],
  ["Best group accuracy", pct(bestGroup.group.accuracy), bestGroup.label],
  ["Lowest eval loss", f3(bestLoss.type.eval_loss) + " nats", bestLoss.label],
  ["Highest AUC-ROC", f3(bestAuc.type.auc_macro), bestAuc.label + " (macro OvR)"],
  ["Records scored", String(DATA.n), DATA.split + " split, " + DATA.types.length + " types"],
];
document.getElementById("tiles").innerHTML = tiles.map(t =>
  '<div class="tile"><div class="k">' + t[0] + '</div><div class="v">' + t[1] +
  '</div><div class="d">' + t[2] + "</div></div>").join("");

["legend-models", "legend-roc", "legend-cal"].forEach(id => {
  document.getElementById(id).innerHTML = DATA.models.map(m =>
    '<span><i class="swatch" style="background:' + COLOR[m.tag] + '"></i>' +
    m.label + "</span>").join("");
});

/* ---------- the six metrics, as small multiples ---------- */
const METRICS = [
  ["accuracy",  "Accuracy",  pct, false],
  ["precision", "Precision", pct, false],
  ["recall",    "Recall",    pct, false],
  ["f1",        "F1-score",  pct, false],
  ["auc",       "AUC-ROC",   f3,  false],
  ["eval_loss", "Eval loss (nats)", f3, true],
];
let grain = "type", avg = "weighted";

function metricValue(m, key) {
  const g = m[grain];
  if (key === "accuracy")  return g.accuracy;
  if (key === "eval_loss") return g.eval_loss;
  if (key === "auc")       return avg === "macro" ? g.auc_macro : g.auc_weighted;
  return g[key + "_" + avg];
}

function drawPanels() {
  const host = document.getElementById("panels");
  host.innerHTML = "";
  METRICS.forEach(spec => {
    const key = spec[0], title = spec[1], fmt = spec[2], lowerBetter = spec[3];
    const rows = DATA.models.map(m => ({ m: m, v: metricValue(m, key) }))
                            .sort((a, b) => lowerBetter ? a.v - b.v : b.v - a.v);
    const W = 340, rowH = 34, top = 34, H = top + rows.length * rowH + 24;
    const L = 8, R = W - 60;
    const max = key === "eval_loss" ? Math.max(...rows.map(r => r.v)) * 1.15 || 1 : 1;
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%",
                            height: H, role: "img",
                            "aria-label": title + " by model" });
    el("text", { x: L, y: 11, class: "lbl" }, svg).textContent = title;

    [0, .5, 1].forEach(t => {
      const x = L + (R - L) * t;
      el("line", { x1: x, x2: x, y1: top - 10, y2: top + rows.length * rowH - 8,
                   class: t === 0 ? "ax" : "gl" }, svg);
      el("text", { x: x, y: H - 8, class: "axis-t", "text-anchor": "middle" }, svg)
        .textContent = key === "eval_loss" ? (max * t).toFixed(1) : fmt(t);
    });
    rows.forEach((r, i) => {
      const y = top + i * rowH, h = rowH - 16;
      const w = Math.max(2, (R - L) * (r.v / max));
      el("text", { x: L + 1, y: y - 4, class: "lbl" }, svg).textContent = r.m.label;
      el("rect", { x: L, y: y, width: w, height: h, rx: 4,
                   fill: COLOR[r.m.tag] }, svg);
      el("text", { x: L + w + 8, y: y + h - 3, class: "val" }, svg).textContent = fmt(r.v);
      const hit = el("rect", { x: 0, y: y - 15, width: W, height: rowH, class: "hit" }, svg);
      bindTip(hit, "<b>" + r.m.label + "</b><br>" + title + ": <b>" + fmt(r.v) + "</b>" +
        '<br><span class="m">' +
        (grain === "type" ? "35-way incident type" : "7-way category group") +
        (key === "accuracy" || key === "eval_loss" ? "" : " · " + avg) +
        " · n=" + DATA.n + "</span>");
    });
    const box = document.createElement("div");
    box.appendChild(svg);
    host.appendChild(box);
  });
}

document.querySelectorAll("[data-grain]").forEach(b => b.onclick = () => {
  grain = b.dataset.grain;
  document.querySelectorAll("[data-grain]").forEach(o =>
    o.setAttribute("aria-pressed", String(o === b)));
  drawPanels(); drawTable();
});
document.querySelectorAll("[data-avg]").forEach(b => b.onclick = () => {
  avg = b.dataset.avg;
  document.querySelectorAll("[data-avg]").forEach(o =>
    o.setAttribute("aria-pressed", String(o === b)));
  drawPanels(); drawTable();
});

/* ---------- table view: the fallback that carries every number ---------- */
function drawTable() {
  const g = grain;
  const cols = [
    ["Accuracy",      m => m[g].accuracy,           pct, false],
    ["Precision (W)", m => m[g].precision_weighted, pct, false],
    ["Recall (W)",    m => m[g].recall_weighted,    pct, false],
    ["F1 (W)",        m => m[g].f1_weighted,        pct, false],
    ["Precision (M)", m => m[g].precision_macro,    pct, false],
    ["Recall (M)",    m => m[g].recall_macro,       pct, false],
    ["F1 (M)",        m => m[g].f1_macro,           pct, false],
    ["AUC-ROC (macro)", m => m[g].auc_macro,        f3,  false],
    ["AUC-ROC (micro)", m => m[g].auc_micro,        f3,  false],
    ["Eval loss",     m => m[g].eval_loss,          f3,  true],
    ["Brier",         m => m[g].brier,              f3,  true],
  ];
  const rows = DATA.models.slice().sort((a, b) => b[g].accuracy - a[g].accuracy);
  const bests = cols.map(c => {
    const vals = rows.map(c[1]);
    return c[3] ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
  });
  document.getElementById("tbl").innerHTML =
    "<caption>" + (g === "type" ? "35-way incident type" : "7-way Category Group") +
    " · n=" + DATA.n +
    " · (W) support-weighted, (M) macro over classes</caption>" +
    "<thead><tr><th>Candidate</th>" + cols.map(c => "<th>" + c[0] + "</th>").join("") +
    "</tr></thead><tbody>" +
    rows.map(m => "<tr><td>" + m.label + "</td>" + cols.map((c, i) => {
      const v = c[1](m);
      return "<td" + (Math.abs(v - bests[i]) < 1e-9 ? ' style="font-weight:650"' : "") +
             ">" + c[2](v) + "</td>";
    }).join("") + "</tr>").join("") + "</tbody>";
  document.getElementById("tbl-note").textContent =
    "AUC-ROC (M) averages only the " + rows[0][g].auc_classes_scored + " of " +
    (g === "type" ? DATA.types.length : DATA.groups.length) + " classes that have at least " +
    "one positive and one negative example in this split; a class with no dev-split example " +
    "has no defined ROC curve and is skipped rather than scored as 0 or 0.5. Eval loss is " +
    "mean categorical cross-entropy in nats with class probabilities floored at " +
    DATA.prob_floor + "; Brier is reported beside it because it is bounded and does not " +
    "depend on that floor.";
}

/* ---------- shared line chart (ROC + calibration) ---------- */
function lineChart(host, series, opts) {
  const W = 470, H = 330, L = 46, R = W - 14, T = 14, B = H - 40;
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
                          role: "img", "aria-label": opts.ylab + " against " + opts.xlab });
  const X = v => L + (R - L) * v, Y = v => B - (B - T) * v;
  for (let i = 0; i <= 4; i++) {
    const t = i / 4;
    el("line", { x1: L, x2: R, y1: Y(t), y2: Y(t), class: i ? "gl" : "ax" }, svg);
    el("text", { x: L - 9, y: Y(t) + 4, class: "axis-t", "text-anchor": "end" }, svg)
      .textContent = t.toFixed(2);
    el("text", { x: X(t), y: B + 18, class: "axis-t", "text-anchor": "middle" }, svg)
      .textContent = t.toFixed(2);
  }
  el("line", { x1: L, x2: L, y1: T, y2: B, class: "ax" }, svg);
  el("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), class: "gl",
               "stroke-dasharray": "4 4" }, svg);
  el("text", { x: (L + R) / 2, y: H - 5, class: "axis-t", "text-anchor": "middle" }, svg)
    .textContent = opts.xlab;
  el("text", { x: 12, y: (T + B) / 2, class: "axis-t", "text-anchor": "middle",
               transform: "rotate(-90 12 " + ((T + B) / 2) + ")" }, svg)
    .textContent = opts.ylab;

  series.forEach(s => {
    const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]) + " " + Y(p[1])).join(" ");
    el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2,
                 "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    if (opts.markers) s.pts.forEach((p, i) => {
      el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 5, fill: s.color,
                     stroke: "var(--surface)", "stroke-width": 2 }, svg);
      bindTip(el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 13, class: "hit" }, svg),
              s.tipAt(i));
    });
    if (s.tip) bindTip(el("path", { d: d, fill: "none", stroke: "transparent",
                                    "stroke-width": 14, class: "hit" }, svg), s.tip);
  });
  host.appendChild(svg);
}

lineChart(document.getElementById("roc"), DATA.models.map(m => ({
  color: COLOR[m.tag],
  pts: m.type.roc.fpr.map((x, i) => [x, m.type.roc.tpr[i]]),
  tip: "<b>" + m.label + "</b><br>Micro-average AUC: <b>" + f3(m.type.roc.auc) + "</b>" +
       '<br><span class="m">Macro OvR ' + f3(m.type.auc_macro) +
       " · 35-way · n=" + DATA.n + "</span>",
})), { xlab: "False positive rate", ylab: "True positive rate" });

lineChart(document.getElementById("cal"), DATA.models.map(m => ({
  color: COLOR[m.tag],
  pts: m.calibration.map(b => [b.conf, b.acc]),
  tipAt: i => { const b = m.calibration[i], gap = b.conf - b.acc;
    return "<b>" + m.label + "</b><br>Bin " + b.lo + "–" + b.hi + ": <b>" + b.n +
      "</b> record" + (b.n === 1 ? "" : "s") + "<br>Mean confidence <b>" + f3(b.conf) +
      "</b> · accuracy <b>" + pct(b.acc) + "</b>" +
      '<br><span class="m">gap ' + (gap >= 0 ? "+" : "") + f3(gap) + " — " +
      (gap > 0 ? "overconfident" : "underconfident") + "</span>"; },
})), { xlab: "Mean predicted probability", ylab: "Observed accuracy", markers: true });

/* ---------- per-group F1 heatmap ---------- */
(function heatmap() {
  const rows = DATA.models, cols = DATA.groups;
  const cw = 116, ch = 40, L = 212, T = 76;
  const W = L + cols.length * cw + 10, H = T + rows.length * ch + 46;
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H,
                          role: "img", "aria-label": "F1 per category group, per model" });
  const step = v => SEQ[Math.min(SEQ.length - 1,
                                 Math.max(0, Math.round(v * (SEQ.length - 1))))];
  cols.forEach((c, j) => {
    const x = L + j * cw + cw / 2 - 6;
    el("text", { x: x, y: T - 12, class: "axis-t",
                 transform: "rotate(-30 " + x + " " + (T - 12) + ")" }, svg).textContent = c;
  });
  rows.forEach((m, i) => {
    el("text", { x: L - 12, y: T + i * ch + ch / 2 + 4, class: "lbl",
                 "text-anchor": "end" }, svg).textContent = m.label;
    cols.forEach((c, j) => {
      const v = m.group.per_class_f1[c] || 0, sup = m.group.per_class_support[c] || 0;
      const g = el("g", null, svg);
      el("rect", { x: L + j * cw + 1, y: T + i * ch + 1, width: cw - 3, height: ch - 3,
                   rx: 4, fill: "var(" + step(v) + ")" }, g);
      el("text", { x: L + j * cw + cw / 2, y: T + i * ch + ch / 2 + 4,
                   "text-anchor": "middle", class: "val",
                   fill: v > .55 ? "var(--seq-ink-hi)" : "var(--seq-ink-lo)" }, g)
        .textContent = v.toFixed(2);
      bindTip(el("rect", { x: L + j * cw, y: T + i * ch, width: cw, height: ch,
                           class: "hit" }, g),
        "<b>" + m.label + "</b><br>" + c + "<br>F1 <b>" + v.toFixed(3) + "</b>" +
        '<br><span class="m">' + sup + " gold record" + (sup === 1 ? "" : "s") +
        " in this split</span>");
    });
  });
  el("text", { x: L - 12, y: H - 15, class: "axis-t", "text-anchor": "end" }, svg)
    .textContent = "F1";
  SEQ.forEach((s, i) => el("rect", { x: L + i * 26, y: H - 24, width: 24, height: 9,
                                     rx: 2, fill: "var(" + s + ")" }, svg));
  el("text", { x: L, y: H - 3, class: "axis-t" }, svg).textContent = "0.0";
  el("text", { x: L + SEQ.length * 26, y: H - 3, class: "axis-t",
               "text-anchor": "end" }, svg).textContent = "1.0";
  const box = document.createElement("div");
  box.className = "scroll"; box.appendChild(svg);
  document.getElementById("heat").appendChild(box);
})();

/* ---------- category-group confusion matrix ---------- */
(function confusion() {
  const ctl = document.getElementById("cm-ctl"), host = document.getElementById("cm");
  let pick = DATA.models[0];
  DATA.models.forEach(m => {
    const b = document.createElement("button");
    b.className = "seg"; b.textContent = m.label;
    b.setAttribute("aria-pressed", String(m === pick));
    b.onclick = () => { pick = m;
      ctl.querySelectorAll("button").forEach(o =>
        o.setAttribute("aria-pressed", String(o === b)));
      draw(); };
    ctl.appendChild(b);
  });
  function draw() {
    host.innerHTML = "";
    const g = DATA.groups, cm = pick.group_confusion;
    const max = Math.max(1, ...cm.map(r => Math.max.apply(null, r)));
    const cw = 106, ch = 40, L = 202, T = 80;
    const W = L + g.length * cw + 10, H = T + g.length * ch + 26;
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H,
                            role: "img", "aria-label": "Confusion matrix, " + pick.label });
    const step = v => SEQ[Math.min(SEQ.length - 1,
                                   Math.round((v / max) * (SEQ.length - 1)))];
    el("text", { x: L, y: 14, class: "axis-t" }, svg).textContent = "predicted →";
    el("text", { x: L - 12, y: T - 44, class: "axis-t", "text-anchor": "end" }, svg)
      .textContent = "gold ↓";
    g.forEach((c, j) => {
      const x = L + j * cw + cw / 2 - 6;
      el("text", { x: x, y: T - 10, class: "axis-t",
                   transform: "rotate(-30 " + x + " " + (T - 10) + ")" }, svg).textContent = c;
    });
    g.forEach((r, i) => {
      el("text", { x: L - 12, y: T + i * ch + ch / 2 + 4, class: "lbl",
                   "text-anchor": "end" }, svg).textContent = r;
      g.forEach((c, j) => {
        const v = cm[i][j], grp = el("g", null, svg);
        el("rect", { x: L + j * cw + 1, y: T + i * ch + 1, width: cw - 3, height: ch - 3,
                     rx: 4, fill: v ? "var(" + step(v) + ")" : "transparent",
                     stroke: v ? "none" : "var(--grid)" }, grp);
        if (i === j) el("rect", { x: L + j * cw + 1, y: T + i * ch + 1, width: cw - 3,
                                  height: ch - 3, rx: 4, fill: "none", stroke: "var(--ink)",
                                  "stroke-width": 1.5, "stroke-opacity": .45 }, grp);
        el("text", { x: L + j * cw + cw / 2, y: T + i * ch + ch / 2 + 4,
                     "text-anchor": "middle", class: "val",
                     fill: v / max > .55 ? "var(--seq-ink-hi)" : "var(--seq-ink-lo)" }, grp)
          .textContent = v || "";
        bindTip(el("rect", { x: L + j * cw, y: T + i * ch, width: cw, height: ch,
                             class: "hit" }, grp),
          "<b>" + v + "</b> record" + (v === 1 ? "" : "s") + "<br>gold <b>" + r + "</b>" +
          "<br>predicted <b>" + c + "</b>" +
          (i === j ? '<br><span class="m">correct</span>' : ""));
      });
    });
    const box = document.createElement("div");
    box.className = "scroll"; box.appendChild(svg);
    host.appendChild(box);
  }
  draw();
})();

/* ---------- confidence distributions ---------- */
(function confidence() {
  const BINS = 5, wrap = document.createElement("div");
  wrap.className = "grid3";
  DATA.models.forEach(m => {
    const W = 340, H = 200, L = 34, R = W - 12, T = 28, B = H - 34;
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
                            role: "img",
                            "aria-label": "Confidence distribution, " + m.label });
    el("text", { x: 0, y: 11, class: "lbl" }, svg).textContent = m.label;
    const hist = arr => { const h = new Array(BINS).fill(0);
      arr.forEach(v => h[Math.min(BINS - 1, Math.floor(v * BINS))]++); return h; };
    const hc = hist(m.confidence.correct), hw = hist(m.confidence.wrong);
    const max = Math.max(1, Math.max.apply(null, hc), Math.max.apply(null, hw));
    el("line", { x1: L, x2: R, y1: B, y2: B, class: "ax" }, svg);
    [0, .5, 1].forEach(t => el("text", { x: L + (R - L) * t, y: B + 16, class: "axis-t",
                                         "text-anchor": "middle" }, svg)
      .textContent = t.toFixed(1));
    el("text", { x: L - 8, y: T + 4, class: "axis-t", "text-anchor": "end" }, svg)
      .textContent = max;
    el("text", { x: L - 8, y: B, class: "axis-t", "text-anchor": "end" }, svg)
      .textContent = "0";
    const bw = (R - L) / BINS, half = (bw - 6) / 2;
    for (let i = 0; i < BINS; i++) {
      [[hc, "var(--good)", "Correct", 0], [hw, "var(--critical)", "Incorrect", 1]]
        .forEach(spec => {
          const v = spec[0][i]; if (!v) return;
          const hgt = (B - T) * (v / max), x = L + i * bw + 2 + spec[3] * half;
          const g = el("g", null, svg);
          el("rect", { x: x, y: B - hgt, width: half, height: hgt, rx: 4,
                       fill: spec[1] }, g);
          bindTip(el("rect", { x: x, y: T, width: half, height: B - T, class: "hit" }, g),
            "<b>" + m.label + "</b><br>" + spec[2] + ": <b>" + v + "</b> record" +
            (v === 1 ? "" : "s") + '<br><span class="m">top-1 probability ' +
            (i / BINS).toFixed(1) + "–" + ((i + 1) / BINS).toFixed(1) + "</span>");
        });
    }
    el("text", { x: (L + R) / 2, y: H - 3, class: "axis-t", "text-anchor": "middle" }, svg)
      .textContent = "top-1 probability";
    const d = document.createElement("div"); d.appendChild(svg); wrap.appendChild(d);
  });
  document.getElementById("conf").appendChild(wrap);
})();

document.getElementById("method").innerHTML =
  "<b>Method.</b> Every candidate was re-queried through Ollama with the identical prompt, " +
  "identical JSON output schema and temperature 0 used for the accuracy comparison, with " +
  "per-token logprobs retained. A distribution over the " + DATA.types.length +
  " canonical incident types was reconstructed from those logprobs (the emitted label takes " +
  "its own sequence probability; alternatives at each token position that still spell a " +
  "valid type prefix take theirs), which is what makes AUC-ROC and cross-entropy measurable " +
  "at all — a hard predicted label alone supports neither. Category-group probabilities " +
  "are the sums of their member types'. Class probabilities are floored at " +
  DATA.prob_floor + "; the loss depends on that floor, and Brier, which does not, is " +
  "reported beside it. Generated by <code>tools/build_metrics_report.py</code> from " +
  "<code>models/classifier_metrics.json</code>.";

drawPanels(); drawTable();
</script>
"""


def main():
    data = json.loads(SRC.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print("wrote {} ({:,} bytes)".format(OUT, OUT.stat().st_size))


if __name__ == "__main__":
    main()
