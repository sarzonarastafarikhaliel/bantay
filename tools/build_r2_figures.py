"""Write "Thesis Writing/R2 Updated Figures.html": every Chapter 4 and Appendix I/J figure
whose numbers come from the classifier or OCR-arm evaluation, all on the 188 gold records.

Values come from results/pooled_188/{classifier_metrics_188,pooled_stats}.json
(tools/score_pooled_188.py, tools/pool_188.py), models/preds_gemini_classify_all188.csv
and models/gemini_arms_run_all188.json. The chart primitives are read from
"Thesis Writing/Chapter 4 Figures.html" so both pages draw the same way.

    python tools/pool_188.py && python tools/score_pooled_188.py && python tools/build_r2_figures.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.eval_gemini_classify import LABEL_BRIDGE

RES = ROOT / "results" / "pooled_188"
SRC_HTML = ROOT / "Thesis Writing" / "Chapter 4 Figures.html"
OUT = ROOT / "Thesis Writing" / "R2 Updated Figures.html"


def pct(x):
    return round(100 * x, 1)


def primitives():
    """The shared token + chart-function block of Chapter 4 Figures.html."""
    html = SRC_HTML.read_text(encoding="utf-8")
    a = html.index("/* ---------- tokens ---------- */")
    b = html.index("/* ---------- figure wrapper + PNG export ---------- */")
    return html[a:b]


def main():
    cm = json.loads((RES / "classifier_metrics_188.json").read_text(encoding="utf-8"))
    ps = json.loads((RES / "pooled_stats.json").read_text(encoding="utf-8"))
    models = []
    for m in cm["models"]:
        t, g = m["type"], m["group"]
        pooled = ps["models"][m["key"]]["type"]["pooled"]
        models.append({
            "acc": pct(t["accuracy"]), "gacc": pct(g["accuracy"]), "ci": [pct(v) for v in pooled["wilson"]],
            "lat": [m["latency_median_s"], m["latency_p95_s"]], "tps": m["throughput_median_toks"],
            "vram": m["peak_vram_mib"], "ram": m["peak_ram_mib"],
            "cs": [pct(m["code_switch"][k]["accuracy"]) for k in ("code_switched", "english_dominant", "tagalog_dominant")],
            "met_t": [pct(t[k]) for k in ("precision_weighted", "recall_weighted", "f1_weighted", "f1_macro")],
            "met_g": [pct(g[k]) for k in ("precision_weighted", "recall_weighted", "f1_weighted", "f1_macro")],
            "prob_t": [round(t["auc_macro"], 3), round(t["eval_loss"], 3), round(t["brier"], 3)],
            "prob_g": [round(g["auc_macro"], 3), round(g["eval_loss"], 3), round(g["brier"], 3)],
            "roc": {"fpr": t["roc"]["fpr"], "tpr": t["roc"]["tpr"], "auc": round(t["roc"]["auc"], 3)},
            "cal": [[c["conf"], c["acc"], c["n"]] for c in m["calibration"]], "p1": m["mean_top1"],
            "gf1": [g["per_class_f1"][grp] for grp in cm["groups"]],
        })
    gm = cm["models"][0]
    cs_n = [gm["code_switch"][k]["n"] for k in ("code_switched", "english_dominant", "tagalog_dominant")]
    gsup = [gm["group"]["per_class_support"][grp] for grp in cm["groups"]]
    err = gm["errors"]
    kp = {"n": err["n_errors"], "same": err["kp_same"], "excl": len(err["kp_excluded"]),
          "cross_other": err["kp_cross"] - len(err["kp_excluded"])}

    partners = []
    for key, by in ps["agreement_by_partner"].items():
        a = by["pooled"]
        partners.append({"rate": pct(a["n_agree"] / a["n"]), "nag": a["n_agree"], "ndis": a["n_disagree"],
                         "agree": pct(a["agree"]["acc"]), "dis": pct(a["disagree"]["acc"]), "wrong": a["agree_wrong"]})
    tiers = [[ps["agreement"]["pooled"]["three_tier"][k]["n"], pct(ps["agreement"]["pooled"]["three_tier"][k]["acc"])]
             for k in ("type_agree", "type_disagree_group_agree", "type_disagree_group_disagree")]
    paired = []
    for i in range(4):
        for grain in ("type", "group"):
            r = ps["paired"][grain][i]
            paired.append({"i": i + 1, "grain": grain.capitalize(), "b": r["b"], "c": r["c"],
                           "p": r["p_exact"], "holm": r["p_holm"]})

    gem = pd.read_csv(ROOT / "models" / "preds_gemini_classify_all188.csv")
    gem_acc = pct((gem["pred_type"] == gem["gold_type"].map(lambda t: LABEL_BRIDGE.get(t, t))).mean())

    arms_path = ROOT / "models" / "gemini_arms_run_all188.json"
    arms = json.loads(arms_path.read_text(encoding="utf-8")) if arms_path.exists() else None

    data = {"models": models, "csn": cs_n, "groups": cm["groups"], "gsup": gsup,
            "cm": gm["group_confusion"], "kp": kp, "partners": partners, "tiers": tiers,
            "paired": paired, "gem": gem_acc, "arms": arms}
    html = TEMPLATE.replace("/*PRIMITIVES*/", primitives()).replace("/*DATA*/", json.dumps(data))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  (OCR arms: {'included' if arms else 'MISSING - run tools/run_gemini_arms.py --split all --tag all188'})")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BANTAY Updated Figures</title>
<style>
  :root { color-scheme: light; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --rule:#e1e0d9; --s1:#2a78d6; }
  body { margin:0; background:#ffffff; color:var(--ink); font: 15px/1.55 "Times New Roman", Times, serif; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 32px 16px 80px; }
  .bar { font: 13px/1.5 "Segoe UI", Arial, sans-serif; color: var(--ink-2); border: 1px solid var(--rule);
         border-radius: 8px; padding: 12px 14px; margin: 0 0 28px; }
  .bar b { color: var(--ink); }
  .bar ul { margin: 6px 0 0; padding-left: 18px; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  h2 { font: 600 13px "Segoe UI", Arial, sans-serif; text-transform: uppercase; letter-spacing: .06em;
       color: var(--muted); margin: 44px 0 6px; border-top: 1px solid var(--rule); padding-top: 18px; }
  figure { margin: 0 0 44px; break-inside: avoid; page-break-inside: avoid; }
  .fignum { font-weight: 700; margin: 0; }
  .figtitle { font-style: italic; margin: 0 0 12px; }
  .where { font: 12px "Segoe UI", Arial, sans-serif; color: var(--muted); margin: -8px 0 10px; }
  .chart { overflow-x: auto; }
  .chart svg { display: block; max-width: 100%; height: auto; }
  .note { font-size: 13.5px; margin: 10px 0 0; }
  .tools { margin-top: 8px; font: 12px "Segoe UI", Arial, sans-serif; }
  .tools button { font: inherit; cursor: pointer; background: #fff; color: var(--ink-2);
                  border: 1px solid #c3c2b7; border-radius: 6px; padding: 4px 10px; }
  .tools button:hover { border-color: var(--s1); color: var(--s1); }
  details { font: 12.5px "Segoe UI", Arial, sans-serif; color: var(--ink-2); margin-top: 8px; }
  details table { border-collapse: collapse; margin-top: 6px; }
  details th, details td { border-bottom: 1px solid var(--rule); padding: 3px 10px 3px 0; text-align: left; font-variant-numeric: tabular-nums; }
  @media print { .bar, .tools, details { display: none; } }
</style>
</head>
<body>
<div class="wrap">
  <h1>Updated Figures: Classifier and OCR Results on All 188 Gold Records</h1>
  <div class="bar">
    Every figure here replaces the image of the same number in <b>BANTAY - Final Thesis Paper (188 records).docx</b>.
    Titles and Notes match the revised document. To replace a figure in Word:
    <ul>
      <li>Click <b>Save PNG</b> under the chart. It downloads at 3&times; resolution on a white background.</li>
      <li>In Word, right-click the old image, choose <b>Change Picture &rarr; From a File</b>, and pick the PNG. Position and size are kept.</li>
    </ul>
    Values come from <code>results/pooled_188/</code>, <code>models/preds_gemini_classify_all188.csv</code> and
    <code>models/gemini_arms_run_all188.json</code>. Rebuild with <code>python tools/build_r2_figures.py</code>.
  </div>
  <div id="body"></div>
</div>

<script>
const D = /*DATA*/;
/*PRIMITIVES*/

/* ---------- extra charts ---------- */
function ciplot(rows,{min=30,max=90,ticks=[30,40,50,60,70,80,90],labelW=210,width=800,rightW=190}={}){
  const x0=labelW, x1=width-rightW, sx=v=>x0+(v-min)/(max-min)*(x1-x0), rh=34, top=6;
  let b='', y=top;
  for(const r of rows){
    const cy=y+rh/2;
    b += T(x0-10,cy+4,r.label,{a:'end'});
    b += L(sx(r.ci[0]),cy,sx(r.ci[1]),cy,r.color,'stroke-width="2.5"') + L(sx(r.ci[0]),cy-6,sx(r.ci[0]),cy+6,r.color,'stroke-width="2"') + L(sx(r.ci[1]),cy-6,sx(r.ci[1]),cy+6,r.color,'stroke-width="2"');
    b += `<g><title>${esc(r.label)}: ${r.v}% (95% CI ${r.ci[0]}–${r.ci[1]}%)</title>${marker(sx(r.v),cy,6.5,r.mk,r.color)}</g>`;
    b += T(x1+16,cy+4,`${r.v.toFixed(1)}% (${r.ci[0]}–${r.ci[1]})`,{c:'v'});
    y += rh;
  }
  const bottom=y+2;
  const g = ticks.map(t=>L(sx(t),top,sx(t),bottom,C.grid)+T(sx(t),bottom+15,t+'%',{a:'middle',c:'t'})).join('');
  return { w:width, h:bottom+38, b:g+b+T((x0+x1)/2,bottom+33,'Incident-type accuracy (Wilson 95% CI)',{a:'middle',c:'t'}) };
}
function mcnemarExact(rows){
  const labelW=250, cx=labelW+130, k=4.4, colP=740, colH=810, rh=24, top=30;
  let b = T(cx-8,14,'◀ Checkpoint-only correct (c)',{a:'end',c:'t'}) + T(cx+8,14,'Gemma-4-only correct (b) ▶',{c:'t'})
        + T(colP,14,'Exact p',{a:'end',c:'t'}) + T(colH,14,'Holm p',{a:'end',c:'t'});
  const pv = p => p < 0.001 ? '<.001' : p.toFixed(3).replace(/^0/,'');
  rows.forEach((r,i)=>{
    const y=top+i*rh, cy=y+rh/2;
    if(i%2===0){ b += T(0,cy+4,MODELS[r.i].n); if(i) b += L(0,y-1,colH+4,y-1,C.grid); }
    b += T(labelW-8,cy+4,r.grain,{a:'end',c:'t'});
    b += R(cx-r.c*k,cy-7,r.c*k,14,C.s2,`${MODELS[r.i].n} (${r.grain}): checkpoint-only correct ${r.c}`) + T(cx-r.c*k-6,cy+4,r.c,{a:'end',c:'v'});
    b += R(cx,cy-7,r.b*k,14,C.s1,`${MODELS[r.i].n} (${r.grain}): Gemma-4-only correct ${r.b}`) + T(cx+r.b*k+6,cy+4,r.b,{c:'v'});
    b += T(colP,cy+4,pv(r.p),{a:'end',c:'v'}) + T(colH,cy+4,pv(r.holm),{a:'end',c:r.holm<0.05?'b':'v'});
  });
  const bottom=top+rows.length*rh;
  b += L(cx,top-4,cx,bottom,C.ink2);
  return { w:colH+8, h:bottom+4, b };
}

/* ---------- figure wrapper + PNG export ---------- */
function section(t){ document.getElementById('body').insertAdjacentHTML('beforeend',`<h2>${esc(t)}</h2>`); }
function figure(n, title, where, chart, note, table){
  const pad=14, W=chart.w+pad*2, H=chart.h+pad*2;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${-pad} ${-pad} ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Figure ${n}. ${esc(title)}"><style>${CSS}</style><rect x="${-pad}" y="${-pad}" width="${W}" height="${H}" fill="#ffffff"/>${chart.b}</svg>`;
  const tbl = table ? `<details><summary>Data table</summary><table><tr>${table[0].map(h=>`<th>${esc(h)}</th>`).join('')}</tr>${table.slice(1).map(r=>`<tr>${r.map(c=>`<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</table></details>` : '';
  document.getElementById('body').insertAdjacentHTML('beforeend',`<figure id="fig-${n}"><p class="fignum">Figure ${n}</p><p class="figtitle">${esc(title)}</p><p class="where">${esc(where)}</p>
    <div class="chart">${svg}</div>${note?`<p class="note"><i>Note.</i> ${esc(note)}</p>`:''}
    <div class="tools"><button data-fig="${n}">Save PNG</button></div>${tbl}</figure>`);
}
document.addEventListener('click',e=>{
  const n=e.target.dataset && e.target.dataset.fig; if(!n) return;
  const el=document.querySelector(`#fig-${n} svg`), vb=el.viewBox.baseVal, k=3, img=new Image();
  img.onload=()=>{ const c=document.createElement('canvas'); c.width=vb.width*k; c.height=vb.height*k;
    const x=c.getContext('2d'); x.fillStyle='#fff'; x.fillRect(0,0,c.width,c.height); x.drawImage(img,0,0,c.width,c.height);
    const a=document.createElement('a'); a.download=`Figure ${n}.png`; a.href=c.toDataURL('image/png'); a.click(); };
  img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(new XMLSerializer().serializeToString(el));
});

const M = D.models, G = M[0], V3 = M[1];
const tblModels = (head, f) => [['Candidate', ...head], ...M.map((m,i)=>[MODELS[i].n, ...f(m)])];

/* =====================================================================
   Chapter 4
   ===================================================================== */
section('Chapter 4, Section 4.1.3');

const GRAIN=[{name:'Incident type (35-way)',color:C.seq[5]},{name:'Category group (7-way)',color:C.seq[2]}];
figure('8', 'Incident-Type and Category Group Accuracy of the Classifier Candidates (n = 188)', 'Section 4.1.3.1, after Table 12.',
  vstack([ legend(GRAIN),
    hbars({ series:GRAIN, max:100, ticks:[0,20,40,60,80,100], tickFmt:v=>v+'%', fmt:pct, labelW:250, rightPad:56, barH:15, xLabel:'Accuracy',
      rows:[ ...M.map((m,i)=>({label:MODELS[i].n, vals:[m.acc, m.gacc]})),
        {label:'Gemini few-shot predecessor', sub:'Superseded; reference only', vals:[D.gem, null], colors:[C.neutral]} ] })
  ],10),
  `n = 188 records, with the same prompt, label set, and JSON schema for every candidate. 937 of the 940 local model-record pairs returned a schema-valid answer; the three that did not are scored as wrong. The Gemini predecessor is a different, cloud-hosted model; its type-level figure (${pct(D.gem)}) is shown as a design reference only.`,
  tblModels(['Type %','Group %'], m=>[m.acc,m.gacc]).concat([['Gemini few-shot predecessor', D.gem, '']]));

const CMX = D.cm, ROWSUM = CMX.map(r=>r.reduce((a,b)=>a+b,0)), DIAG = CMX.reduce((a,r,i)=>a+r[i],0);
figure('9', 'Category Group Confusion Matrix of Gemma 4 (n = 188)', 'Section 4.1.3.1, after Figure 8.',
  heat({ rowLabels:D.groups.map((g,i)=>`${g} (${ROWSUM[i]})`), colLabels:D.groups, m:CMX, labelW:210, cw:62, ch:34, headH:120, diag:true,
    colTitle:'Predicted group', shade:v=>v?C.seq[Math.min(6,Math.round(Math.sqrt(v/Math.max(...CMX.flat()))*6))]:null }),
  `Rows are the gold group, with support in parentheses. Columns are the predicted group. The outlined diagonal shows correct predictions: ${DIAG} of 188 (${(DIAG/188*100).toFixed(1)}%). Cells show counts, not rates. Shading uses a square-root scale so that small counts stay visible. Empty cells are zero.`,
  [['Gold group', ...D.groups], ...CMX.map((r,i)=>[D.groups[i], ...r])]);

figure('10', 'Reliability Diagram: Top-1 Confidence Against Observed Accuracy', 'Section 4.1.3.1, after Table 15.',
  vstack([
    xy({ width:560, height:440, points:true, xLabel:'Mean top-1 probability in bin', yLabel:'Share of predictions correct',
      series:MODELS.map((mo,i)=>({name:mo.n,color:mo.c,mk:mo.mk,w:1.25,dash:'3 3',pts:M[i].cal})) }),
    legend(MODELS.map((mo,i)=>({name:`${mo.n} (mean top-1 = ${M[i].p1.toFixed(3)})`,color:mo.c,mk:mo.mk})),{width:560})
  ],10),
  `Probability bins of width 0.2, n = 188. Marker area is proportional to the number of records in the bin. Points below the dashed diagonal are overconfident. Gemma 4 places all 188 predictions in the 0.8–1.0 bin at a mean confidence of 1.000, including the 59 incorrect ones. Only Gemma-SEA-LION-v3-9B-IT spreads its predictions across several confidence levels.`,
  [['Candidate','Bin mean confidence','Bin accuracy','n'], ...M.flatMap((m,i)=>m.cal.map(c=>[MODELS[i].n, c[0], c[1], c[2]]))]);

figure('11', 'Incident-Type Accuracy of the Five Candidates With 95% Confidence Intervals', 'Section 4.1.3.2, after Table 16.',
  ciplot(M.map((m,i)=>({label:MODELS[i].n, v:m.acc, ci:m.ci, color:MODELS[i].c, mk:MODELS[i].mk}))),
  `n = 188. Whiskers are Wilson 95% confidence intervals. Gemma 4 (${G.acc}%, ${G.ci[0]}–${G.ci[1]}%) and Gemma-SEA-LION-v3-9B-IT (${V3.acc}%, ${V3.ci[0]}–${V3.ci[1]}%) overlap and cannot be separated (exact McNemar p = .058; Table 19). The intervals of the three smaller checkpoints lie below Gemma 4's.`,
  tblModels(['Type %','Wilson 95% CI'], m=>[m.acc, `${m.ci[0]}–${m.ci[1]}`]));

const P = D.partners;
figure('12', 'Gemma 4 Accuracy When the Check Partner Agrees Versus Disagrees', 'Section 4.1.3.3, after Table 17.',
  vstack([
    legend([{name:'Gemma 4 accuracy when the partner disagrees',color:C.ink2,mk:'circle',hollow:true},{name:'Gemma 4 accuracy when the partner agrees',color:C.ink2,mk:'circle'}]),
    dumbbell({ min:40, max:90, ticks:[40,50,60,70,80,90], labelW:220, rightW:110,
      rows:P.map((p,i)=>({ label:MODELS[i+1].n, sub:`Agreement ${p.rate.toFixed(1)}% (n = ${p.nag})`, a:p.dis, b:p.agree, color:MODELS[i+1].c, right:`Gap ${(p.agree-p.dis).toFixed(1)} pts` })) })
  ],10),
  `n = 188 records per pairing. Agreement means the two models predict the same incident type. Gemma-SEA-LION-v3-9B-IT gives the widest separation (${P[0].agree}% vs. ${P[0].dis}%) and is the deployed check partner (Chapter 3, Section 3.3.2). ${P[0].wrong} of its ${P[0].nag} agreed suggestions are wrong.`,
  [['Check partner','n agree','Accuracy when agree %','n disagree','Accuracy when disagree %','Agreed but wrong'], ...P.map((p,i)=>[MODELS[i+1].n,p.nag,p.agree,p.ndis,p.dis,p.wrong])]);

/* =====================================================================
   Appendix I (OCR) and Appendix J
   ===================================================================== */
section('Appendix I');
if (D.arms) {
  const A = D.arms, ARMS=[{name:'Text arm (Vision text → Gemini repair)',color:C.s2},{name:'Image arm (page image + Vision text → Gemini repair, deployed)',color:C.s1}];
  const ok = a => (a.statuses.ok||0), pages = a => a.pages, rej = a => a.pages - ok(a);
  const armOpts={ series:ARMS, labelW:180, width:720, rightPad:90, barH:14 };
  figure('I4', 'Outcomes of the Two OCR Repair Arms on the 188 Pages', 'Appendix I.',
    vstack([ legend(ARMS),
      titled('(a) Page outcome (% of 188 pages)', hbars({...armOpts, max:100, ticks:[0,25,50,75,100], tickFmt:v=>v+'%',
        rows:[{label:'Completed (ok)',vals:[ok(A.text)/pages(A.text)*100, ok(A.image)/pages(A.image)*100],labs:[`${(ok(A.text)/pages(A.text)*100).toFixed(1)}% (${ok(A.text)})`,`${(ok(A.image)/pages(A.image)*100).toFixed(1)}% (${ok(A.image)})`]},
              {label:'Not completed',sub:'guard-rejected or error',vals:[rej(A.text)/pages(A.text)*100, rej(A.image)/pages(A.image)*100],labs:[`${(rej(A.text)/pages(A.text)*100).toFixed(1)}% (${rej(A.text)})`,`${(rej(A.image)/pages(A.image)*100).toFixed(1)}% (${rej(A.image)})`]}] }),180),
      titled('(b) Word-level changes (% of words)', hbars({...armOpts, max:35, ticks:[0,10,20,30], tickFmt:v=>v+'%', fmt:pct,
        rows:[{label:'Words repaired',vals:[A.text.fidelity.repaired_rate*100,A.image.fidelity.repaired_rate*100]},
              {label:'Words added',vals:[A.text.fidelity.added_rate*100,A.image.fidelity.added_rate*100]},
              {label:'Words lost',vals:[A.text.fidelity.lost_rate*100,A.image.fidelity.lost_rate*100]}] }),180),
      titled('(c) Average processing time', hbars({...armOpts, max:Math.ceil(Math.max(A.text.sec_per_page,A.image.sec_per_page)/2)*2+2, ticks:[0,2,4,6,8,10], tickFmt:v=>v+' s', fmt:v=>v.toFixed(1)+' s/page',
        rows:[{label:'Seconds per page',vals:[A.text.sec_per_page,A.image.sec_per_page]}] }),180)
    ],14),
    `n = 188 pages per arm. A page that is guard-rejected or errors keeps its raw Vision transcription. Total wall time was ${(A.text.wall_seconds/60).toFixed(1)} minutes for the text arm and ${(A.image.wall_seconds/60).toFixed(1)} minutes for the image arm.`,
    [['Arm','Pages','Completed','Not completed','Repaired %','Added %','Lost %','s/page'], ...[['Text',A.text],['Image',A.image]].map(([n,a])=>[n,a.pages,ok(a),rej(a),(a.fidelity.repaired_rate*100).toFixed(1),(a.fidelity.added_rate*100).toFixed(1),(a.fidelity.lost_rate*100).toFixed(1),a.sec_per_page])]);
  // PaddleOCR's 81.8 s/page was measured on all 188 pages (unchanged); the two cloud paths are this run's.
  figure('I5', 'Average Processing Time per Page by OCR Configuration', 'Appendix I.',
    hbars({ max:90, ticks:[0,15,30,45,60,75,90], tickFmt:v=>v+' s', fmt:v=>v.toFixed(1)+' s/page', labelW:250, rightPad:80, barH:20, xLabel:'Seconds per page',
      rows:[{label:'Cloud: Vision + Gemini, text arm',vals:[A.text.sec_per_page],colors:[C.s2]},
            {label:'Cloud: Vision + Gemini, image arm',sub:'Deployed default',vals:[A.image.sec_per_page],colors:[C.s1]},
            {label:'Local: PaddleOCR',sub:'CPU-only runtime, 188 pages in about 4 h 16 min',vals:[81.8],colors:[C.s3]}] }),
    'The cloud paths include the Gemini repair call. A GPU build of PaddleOCR exists but was not measured. At 81.8 seconds per page, the offline path suits background batch digitization, not interactive one-record intake.',
    [['Configuration','Seconds per page'],['Cloud text arm',A.text.sec_per_page],['Cloud image arm',A.image.sec_per_page],['Local PaddleOCR',81.8]]);
} else {
  document.getElementById('body').insertAdjacentHTML('beforeend','<p><b>Figures I4 and I5 not built:</b> models/gemini_arms_run_all188.json is missing.</p>');
}

section('Appendix J');
const one=(vals,extra={})=>MODELS.map((mo,i)=>({label:mo.n,vals:[vals[i]],colors:[mo.c],...(extra.f?extra.f(i):{})}));
const latMax = Math.ceil(Math.max(...M.map(m=>m.lat[1]))/15)*15;
figure('J1', 'Latency, Throughput, and Peak Memory of the Five Candidates', 'Appendix J.',
  vstack([
    hstack([
      titled('(a) Latency per record: median, whisker to P95', hbars({ labelW:200, width:470, rightPad:110, barH:15, max:latMax, ticks:[0,15,30,45].filter(t=>t<=latMax), tickFmt:v=>v+' s',
        rows:one(M.map(m=>m.lat[0]),{f:i=>({whisker:M[i].lat[1],labs:[`${M[i].lat[0]} s (P95 ${M[i].lat[1]})`]})}) }),200),
      titled('(b) Throughput, median', hbars({ labelW:8, width:230, hideLabels:true, rightPad:66, barH:15, max:90, ticks:[0,30,60,90],
        fmt:v=>v+' tok/s', rows:one(M.map(m=>m.tps)) }),8)
    ],14),
    hstack([
      titled('(c) Peak GPU memory (VRAM)', hbars({ labelW:200, width:470, rightPad:86, barH:15, max:6500, ticks:[0,2000,4000,6000], tickFmt:num,
        fmt:v=>num(v)+' MiB', ref:[{v:6144,label:'6 GB limit',a:'end'}], rows:one(M.map(m=>m.vram)) }),200),
      titled('(d) Peak system RAM', hbars({ labelW:8, width:230, hideLabels:true, rightPad:80, barH:15, max:32000, ticks:[0,15000,30000], tickFmt:v=>v/1000+'k',
        fmt:v=>num(v)+' MiB', rows:one(M.map(m=>m.ram)) }),8)
    ],14)
  ],20),
  `All values are from each candidate's runs over the 188 records. Peak VRAM and RAM are system-wide peaks sampled at about one-second resolution, not per-call figures. No candidate exceeded the deployment machine's 6 GB of GPU memory, so the selection was decided on accuracy.`,
  tblModels(['Latency median s','Latency P95 s','Throughput tok/s','Peak VRAM MiB','Peak RAM MiB'], m=>[m.lat[0],m.lat[1],m.tps,m.vram,m.ram]));

const LANG=[{name:`Code-switched (n = ${D.csn[0]})`,color:C.seq[6]},{name:`English-dominant (n = ${D.csn[1]})`,color:C.seq[4]},{name:`Tagalog-dominant (n = ${D.csn[2]})`,color:C.seq[1]}];
figure('J2', 'Incident-Type Accuracy by Narrative Language Profile', 'Appendix J.',
  vstack([ legend(LANG),
    hbars({ series:LANG, max:100, ticks:[0,20,40,60,80,100], tickFmt:v=>v+'%', fmt:pct, labelW:200, rightPad:56, barH:13, rowGap:14, xLabel:'Incident-type accuracy',
      rows:MODELS.map((mo,i)=>({label:mo.n,vals:M[i].cs})) })
  ],10),
  'n = 188 records, split with the function-word lexicon heuristic (Section 3.3.3). This split is diagnostic only. It is not a certified per-narrative language label.',
  tblModels(['Code-switched %','English-dominant %','Tagalog-dominant %'], m=>m.cs));

const MET=[{name:'Precision (weighted)',color:C.seq[6]},{name:'Recall (weighted)',color:C.seq[4]},{name:'F1 (weighted)',color:C.seq[3]},{name:'F1 (macro)',color:C.seq[1]}];
const metRows=(key)=>MODELS.map((mo,i)=>({label:mo.n,vals:M[i][key]}));
const metOpts={ series:MET, max:100, ticks:[0,25,50,75,100], tickFmt:v=>v+'%', fmt:pct, barH:11, inner:2, rowGap:14, rightPad:50 };
figure('J3', 'Label-Based Metrics per Candidate at Both Classification Grains', 'Appendix J.',
  vstack([ legend(MET,{width:760}),
    hstack([
      titled('(a) Incident type (35-way)', hbars({...metOpts, labelW:200, width:450, rows:metRows('met_t') }),200),
      titled('(b) Category group (7-way)', hbars({...metOpts, labelW:8, width:300, hideLabels:true, rows:metRows('met_g') }),8)
    ],12)
  ],10),
  'n = 188. Weighted = averaged by class support. Macro = unweighted mean over classes. At the incident-type grain, many of the 27 represented categories have only a few examples, which explains the gap between macro and weighted values.',
  tblModels(['Type P(W)','Type R(W)','Type F1(W)','Type F1(M)','Group P(W)','Group R(W)','Group F1(W)','Group F1(M)'], m=>[...m.met_t,...m.met_g]));

const PG=[{name:'Incident type',color:C.seq[5]},{name:'Category group',color:C.seq[2]}];
const pRows=(j)=>MODELS.map((mo,i)=>({label:mo.n,vals:[M[i].prob_t[j],M[i].prob_g[j]]}));
const pOpts={ series:PG, barH:12, inner:2, rowGap:14, rightPad:44 };
const lossMax = Math.ceil(Math.max(...M.map(m=>m.prob_t[1]))/4)*4;
figure('J4', 'Probability-Based Metrics per Candidate: AUC-ROC, Eval Loss, and Brier Score', 'Appendix J.',
  vstack([ legend(PG),
    hstack([
      titled('(a) AUC-ROC, macro (higher = better)', hbars({...pOpts, labelW:200, width:440, max:1, ticks:[0,.5,1], fmt:v=>v.toFixed(3), rows:pRows(0) }),200),
      titled('(b) Eval loss (lower = better)', hbars({...pOpts, labelW:8, width:220, hideLabels:true, max:lossMax, ticks:[0,lossMax/2,lossMax], fmt:v=>v.toFixed(2), rows:pRows(1) }),8),
      titled('(c) Brier (lower = better)', hbars({...pOpts, labelW:8, width:220, hideLabels:true, max:1.2, ticks:[0,.6,1.2], fmt:v=>v.toFixed(3), rows:pRows(2) }),8)
    ],10)
  ],10),
  'n = 188. Probabilities are reconstructed from token log-probabilities as described at the start of this appendix. Eval loss is mean cross-entropy at a 1 x 10^-6 probability floor. Gemma-SEA-LION-v3-9B-IT has the best AUC-ROC and eval loss even though it is less accurate. The bounded Brier score also favors Gemma-SEA-LION-v3-9B-IT at the incident-type grain, though narrowly, and places Gemma 4 first at the Category Group grain (Section 4.1.3.1).',
  tblModels(['Type AUC','Type loss','Type Brier','Group AUC','Group loss','Group Brier'], m=>[...m.prob_t,...m.prob_g]));

figure('J5', 'Micro-Averaged ROC Curves at the Incident-Type Grain', 'Appendix J.',
  vstack([
    xy({ width:560, height:440, xLabel:'False positive rate', yLabel:'True positive rate',
      series:MODELS.map((mo,i)=>({name:`${mo.n}: micro AUC ${M[i].roc.auc.toFixed(3)}`,color:mo.c,dash:mo.dash,w:i?1.75:2.75,pts:M[i].roc.fpr.map((f,j)=>[f,M[i].roc.tpr[j]])})) }),
    legend(MODELS.map((mo,i)=>({name:`${mo.n} (AUC ${M[i].roc.auc.toFixed(3)})`,color:mo.c,line:mo.dash,w:i?2:3})),{width:560})
  ],10),
  'One-vs-rest, micro-averaged: each record and class pair is one binary decision, pooled into one curve per model (n = 188). The dashed diagonal is chance. Micro AUC differs from the macro AUC-ROC in Figure J4 because it pools all classes before scoring.',
  tblModels(['Micro AUC'], m=>[m.roc.auc]));

figure('J6', 'Per-Group F1 Score for Each Candidate', 'Appendix J.',
  heat({ rowLabels:MODELS.map(m=>m.n), colLabels:D.groups, colSubs:D.gsup.map(n=>`n=${n}`), m:M.map(m=>m.gf1), labelW:210, cw:66, ch:32, headH:132,
    fmt:v=>v.toFixed(2), shade:v=>C.seq[Math.round(v*6)] }),
  'F1 for each Category Group (n = 188). Darker cells are higher. The dashboard\'s PNP-tier and KP-referability panels are read at this grain. Administrative/Referral has one record, so its 0.00 reflects a single miss.',
  [['Candidate', ...D.groups], ...M.map((m,i)=>[MODELS[i].n, ...m.gf1])]);

const TL = ['Type agrees ("high")','Type disagrees, group agrees (candidate "medium")','Type and group both disagree ("low")'];
figure('J7', 'Three-Tier Agreement Signal Tested and Rejected', 'Appendix J.',
  hbars({ max:100, ticks:[0,25,50,75,100], tickFmt:v=>v+'%', fmt:pct, labelW:330, rightPad:60, barH:20, xLabel:'Gemma 4 incident-type accuracy',
    rows:TL.map((l,i)=>({label:l, sub:`n = ${D.tiers[i][0]}`, vals:[D.tiers[i][1]], colors:[i?C.seq[2]:C.seq[5]]})) }),
  `Check partner: Gemma-SEA-LION-v3-9B-IT, n = 188. The candidate middle tier (${pct(D.tiers[1][1])}) does not separate from the low tier (${pct(D.tiers[2][1])}). The deployed signal is therefore binary: agree or disagree.`,
  [['Tier','n','Gemma 4 accuracy %'], ...TL.map((l,i)=>[l,D.tiers[i][0],D.tiers[i][1]])]);

figure('J8', 'McNemar Test of Gemma 4 Against Each SEA-LION Checkpoint', 'Appendix J.',
  vstack([
    legend([{name:'Records only Gemma 4 classified correctly (b)',color:C.s1},{name:'Records only the SEA-LION checkpoint classified correctly (c)',color:C.s2}],{width:820}),
    mcnemarExact(D.paired)
  ],12),
  'Exact McNemar test, two-sided, n = 188 paired records per comparison. Bars show the discordant records only. p values in the Holm column are adjusted across the four checkpoints within each grain (Table 19); bold values are below .05. Against Gemma-SEA-LION-v3-9B-IT the difference is not significant at either grain (type p = .058, group p = .080). The continuity-corrected version is in Table J2.',
  [['Checkpoint','Grain','b','c','Exact p','Holm p'], ...D.paired.map(r=>[MODELS[r.i].n,r.grain,r.b,r.c,r.p.toFixed(4),r.holm.toFixed(4)])]);

const K = D.kp, KPS=[{name:'Stays within the same KP tier',color:C.seq[2]},{name:'Crosses a KP tier, Excluded not involved',color:C.s4},{name:'Crosses a tier into or out of Excluded',color:C.s2}];
const kpp = v => `${v} (${(v/K.n*100).toFixed(1)}%)`;
figure('J9', `KP-Referability Impact of the Deployed Classifier's ${K.n} Errors`, 'Appendix J.',
  vstack([ legend(KPS),
    hbars({ stacked:true, series:KPS, max:K.n, ticks:[0,10,20,30,40,50,K.n].filter((t,i,a)=>t<=K.n&&a.indexOf(t)===i), labelW:130, rightPad:20, barH:30, xLabel:'Misclassified records',
      rows:[{label:'Gemma 4 errors',sub:`n = ${K.n}`,vals:[K.same,K.cross_other,K.excl],inner:[kpp(K.same),kpp(K.cross_other),kpp(K.excl)]}] })
  ],10),
  `Errors were mapped to KP tiers with the deployed lookup table (bantay/normalize.py). In total, ${K.cross_other+K.excl} errors (${((K.cross_other+K.excl)/K.n*100).toFixed(1)}%) cross a KP tier, and ${K.excl} of these (${(K.excl/K.n*100).toFixed(1)}% of all errors) involve the Excluded tier. No gold RA 9262 record was predicted into a mediable category; all nine gold RA 9262 records were predicted as RA 9262 (Table J4).`,
  [['Outcome','Errors'],['Same KP tier',K.same],['Crosses tier, Excluded not involved',K.cross_other],['Involves Excluded',K.excl]]);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
