"""Full Chapter 4 metric set on all 188 gold records: the 151- and 37-record
prediction files of each candidate are concatenated (records and probability rows
together) and scored with the same functions tools/score_classifier_metrics.py
uses, so the 188-record tables are computed exactly like the earlier ones.
Nothing is re-queried and nothing in models/ is written.

Also derives the deployed model's error analysis (per-type precision/recall,
KP-tier impact of its errors, largest confusion pairs) on the same 188 records.

    python tools/score_pooled_188.py   ->  results/pooled_188/classifier_metrics_188.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay.ml.sealion_classify import EVALUATED_MODELS
from bantay.normalize import get_category_group, get_kp_status
from tools.eval_classifier_metrics import PROB_FLOOR
from tools.pool_188 import KEYS, _safe
from tools.score_classifier_metrics import _calibration, _code_switch_bucket, _score

MODELS = ROOT / "models"
OUT = ROOT / "results" / "pooled_188" / "classifier_metrics_188.json"


def load(tag):
    """Concatenated predictions and probability rows for one candidate (188 rows)."""
    safe = _safe(tag)
    dfs, ps = [], []
    for suffix in ("", "_holdout"):
        dfs.append(pd.read_csv(MODELS / f"metrics_preds_{safe}{suffix}.csv"))
        ps.append(np.load(MODELS / f"metrics_probs_{safe}{suffix}.npy"))
    df = pd.concat(dfs, ignore_index=True)
    P = np.vstack(ps)
    assert len(df) == len(P) == 188 and df["source_file"].is_unique
    return df, P


def latency(df, manifests, tag):
    lat = df["latency_s"].dropna().values
    tps = (df["eval_count"] / df["eval_duration_s"]).replace([np.inf, -np.inf], np.nan).dropna()
    runs = [next(r for r in m["runs"] if r["tag"] == tag) for m in manifests]
    return {"seconds": round(sum(r["seconds"] for r in runs), 1),
            "latency_median_s": round(float(np.median(lat)), 2),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
            "throughput_median_toks": round(float(tps.median()), 1),
            "peak_vram_mib": max(r["peak_vram_mib"] for r in runs),
            "peak_ram_mib": max(r["peak_ram_mib"] for r in runs)}


def error_analysis(df, types):
    gold, pred = df["gold_type"], df["pred_type"].fillna("Others / Miscellaneous")
    wrong = df[gold != df["pred_type"]]
    kp = lambda t: get_kp_status(t)[0] if isinstance(t, str) else "Unclassified"
    same, cross, excl = 0, 0, []
    for r in wrong.itertuples():
        g, p = kp(r.gold_type), kp(r.pred_type)
        if g == p:
            same += 1
        else:
            cross += 1
        if "Excluded" in (g, p):
            excl.append({"source_file": r.source_file, "gold": r.gold_type, "gold_kp": g,
                         "pred": r.pred_type, "pred_kp": p})
    pairs = Counter(zip(wrong["gold_type"], wrong["pred_type"].fillna("(no answer)")))
    by_gold = Counter(wrong["gold_type"])
    present = sorted(set(gold) | set(pred), key=types.index)
    p, r, f, sup = precision_recall_fscore_support(gold, pred, labels=present, zero_division=0)
    n_pred = Counter(pred)
    per_type = [{"type": t, "gold": int(s), "predicted": int(n_pred.get(t, 0)),
                 "precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(c), 4)}
                for t, a, b, c, s in zip(present, p, r, f, sup)]
    per_type.sort(key=lambda d: (-d["gold"], d["type"]))
    return {"n_errors": int(len(wrong)), "kp_same": same, "kp_cross": cross,
            "kp_excluded": excl, "top_pairs": [[g, pr, n] for (g, pr), n in pairs.most_common(10)],
            "errors_by_gold": by_gold.most_common(10), "per_type": per_type}


def main():
    manifests = [json.loads((MODELS / f).read_text())
                 for f in ("metrics_manifest.json", "metrics_manifest_holdout.json")]
    types, groups = manifests[0]["types"], manifests[0]["groups"]
    g_index = {g: i for i, g in enumerate(groups)}
    M = np.zeros((len(types), len(groups)))
    for i, t in enumerate(types):
        M[i, g_index[get_category_group(t)]] = 1.0

    gold = pd.read_csv(ROOT / "data/raw/bantay_ml_ready_gold.csv")
    bucket_of = {sf: _code_switch_bucket(n) for sf, n in zip(gold["source_file"], gold["Narrative/Summary"])}

    out = {"n": 188, "prob_floor": PROB_FLOOR, "types": types, "groups": groups, "models": []}
    for m in EVALUATED_MODELS:
        df, P = load(m["tag"])
        t_idx = np.array([types.index(t) for t in df.gold_type])
        g_idx = np.array([g_index[g] for g in df.gold_group])
        pred_t = df.pred_type.fillna("Others / Miscellaneous").values
        pred_g = df.pred_group.fillna("Catch-all").values
        correct = pred_t == df.gold_type.values
        bucket = df["source_file"].map(bucket_of).fillna("unclear")
        cm = np.zeros((len(groups), len(groups)), dtype=int)
        for gg, pg in zip(df.gold_group, pred_g):
            cm[g_index[gg], g_index[pg]] += 1
        entry = {"tag": m["tag"], "key": KEYS[m["tag"]], "label": m["label"],
                 **latency(df, manifests, m["tag"]),
                 "n_unparsed": int(df.pred_type.isna().sum()),
                 "n_errors_transport": int(df["error"].notna().sum()),
                 "n_reconstructed": int(df.reconstructed.sum()),
                 "type": _score(df.gold_type.values, pred_t, t_idx, P, types),
                 "group": _score(df.gold_group.values, pred_g, g_idx, P @ M, groups),
                 "calibration": _calibration(P, t_idx),
                 "code_switch": {b: {"n": int((bucket == b).sum()),
                                     "accuracy": round(float(correct[(bucket == b).values].mean()), 4)}
                                 for b in sorted(bucket.unique())},
                 "mean_top1": round(float(P.max(axis=1).mean()), 4),
                 "group_confusion": cm.tolist()}
        if m["tag"] == EVALUATED_MODELS[0]["tag"]:
            entry["errors"] = error_analysis(df, types)
        out["models"].append(entry)
        print(f"{m['label']:<36} acc {entry['type']['accuracy']:.3f}  group {entry['group']['accuracy']:.3f}"
              f"  F1w {entry['type']['f1_weighted']:.3f}  AUC {entry['type']['auc_macro']:.3f}"
              f"  loss {entry['type']['eval_loss']:.3f}  brier {entry['type']['brier']:.3f}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
