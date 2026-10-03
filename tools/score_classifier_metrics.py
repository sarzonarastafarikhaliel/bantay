"""Scores the probability matrices written by tools/eval_classifier_metrics.py
into the six metrics Chapter 4 reports, and emits models/classifier_metrics.json.

Split from the collection script on purpose: the Ollama sweep costs about an
hour of wall time, so scoring must be re-runnable against the saved .npy
matrices without re-querying a single model.

Two grains are scored from the same run: the 35-way canonical incident type, and
the 7-way Category Group obtained by summing each type's probability into its
group (the probability of a partition is the sum of its members' - no second
model call is needed and no independence assumption is made).

    python tools/score_classifier_metrics.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay.ml.preprocess import ENGLISH_STOP_WORDS, TAGALOG_STOP_WORDS
from bantay.ml.sealion_classify import EVALUATED_MODELS
from bantay.normalize import get_category_group
from tools.eval_classifier_metrics import PROB_FLOOR

MODELS = ROOT / "models"
LABELS = {m["tag"]: m["label"] for m in EVALUATED_MODELS}

# No language-ID library is installed (and adding one for a 27-class, ~150
# record slice is not worth the dependency), so code-switch is bucketed with
# the same Tagalog/English function-word lists preprocess.py already uses for
# stop-word removal - function words are a reliable per-language signal even
# in a text that borrows content words freely, which is exactly what Taglish
# does. This is a lexicon heuristic, not a language-ID model: report it as
# such, not as a validated code-switch detector.
_WORD_RE = re.compile(r"[a-zA-Z']+")
_CODE_SWITCH_MIN_HITS = 2


def _code_switch_bucket(text):
    tokens = _WORD_RE.findall(str(text).lower())
    tl = sum(t in TAGALOG_STOP_WORDS for t in tokens)
    en = sum(t in ENGLISH_STOP_WORDS for t in tokens)
    if tl >= _CODE_SWITCH_MIN_HITS and en >= _CODE_SWITCH_MIN_HITS:
        return "code_switched"
    if tl >= _CODE_SWITCH_MIN_HITS:
        return "tagalog_dominant"
    if en >= _CODE_SWITCH_MIN_HITS:
        return "english_dominant"
    return "unclear"


def _ovr_auc(y_idx, P, n_classes):
    """Macro one-vs-rest AUC over the classes that are actually scorable.

    sklearn's roc_auc_score(multi_class="ovr") requires every column to appear in
    y_true, which a 35-way taxonomy on 58 records never satisfies - 14 types have
    no dev-split example at all. A class with no positive (or no negative) has an
    undefined ROC curve, so it is skipped rather than scored as 0 or 0.5, both of
    which would silently move the macro average. The scored count is reported
    alongside so the denominator is never implicit.
    """
    per_class, skipped = {}, 0
    for c in range(n_classes):
        pos = (y_idx == c)
        if pos.sum() == 0 or pos.all():
            skipped += 1
            continue
        fpr, tpr, _ = roc_curve(pos.astype(int), P[:, c])
        per_class[c] = float(auc(fpr, tpr))
    if not per_class:
        return float("nan"), float("nan"), {}, skipped
    vals = list(per_class.values())
    macro = float(np.mean(vals))
    w = np.array([(y_idx == c).sum() for c in per_class], dtype=float)
    weighted = float(np.average(vals, weights=w)) if w.sum() else macro
    return macro, weighted, per_class, skipped


def _micro_roc(y_idx, P):
    """Micro-average ROC curve: every (record, class) cell is one binary decision.
    This is what the report plots - it is defined even when individual classes are
    not, and it pools all n x k cells into a single curve."""
    y_bin = np.zeros_like(P, dtype=int)
    y_bin[np.arange(len(y_idx)), y_idx] = 1
    fpr, tpr, _ = roc_curve(y_bin.ravel(), P.ravel())
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 250)).astype(int))
    return {"fpr": [round(float(fpr[i]), 5) for i in keep],
            "tpr": [round(float(tpr[i]), 5) for i in keep],
            "auc": float(auc(fpr, tpr))}


def _score(y_true, y_pred, y_idx, P, classes):
    idx_n = len(classes)
    row = {"n": int(len(y_true)), "accuracy": float((y_true == y_pred).mean())}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, labels=classes, zero_division=0)
        row["precision_" + avg] = float(p)
        row["recall_" + avg] = float(r)
        row["f1_" + avg] = float(f)

    macro, weighted, _, skipped = _ovr_auc(y_idx, P, idx_n)
    row["auc_macro"], row["auc_weighted"] = macro, weighted
    row["auc_classes_scored"] = idx_n - skipped
    row["roc"] = _micro_roc(y_idx, P)
    row["auc_micro"] = row["roc"]["auc"]

    p_true = P[np.arange(len(y_idx)), y_idx]
    row["eval_loss"] = float(-np.log(np.clip(p_true, PROB_FLOOR, 1.0)).mean())
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    row["brier"] = float(((P - onehot) ** 2).sum(axis=1).mean())
    row["zero_one_loss"] = 1.0 - row["accuracy"]
    row["mean_p_top1"] = float(P.max(axis=1).mean())
    row["mean_p_true"] = float(p_true.mean())

    _, _, f_per, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=classes, zero_division=0)
    row["per_class_f1"] = {c: round(float(v), 4) for c, v in zip(classes, f_per)}
    row["per_class_support"] = {c: int(v) for c, v in zip(classes, sup)}
    return row


def _calibration(P, y_idx, bins=5):
    """Reliability curve: mean top-1 probability against the share actually correct."""
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if j == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                    "n": int(m.sum()), "conf": round(float(conf[m].mean()), 4),
                    "acc": round(float(correct[m].mean()), 4)})
    return out


def main():
    manifest = json.loads((MODELS / "metrics_manifest.json").read_text())
    types, groups = manifest["types"], manifest["groups"]
    g_index = {g: i for i, g in enumerate(groups)}
    # Type column -> group column, so a group distribution is one matrix product.
    M = np.zeros((len(types), len(groups)))
    for i, t in enumerate(types):
        M[i, g_index[get_category_group(t)]] = 1.0

    out = {"split": manifest["split"], "n": manifest["n"], "prob_floor": PROB_FLOOR,
           "types": types, "groups": groups, "models": []}

    gold = pd.read_csv(MODELS.parent / "data/raw/bantay_ml_ready_gold.csv")
    gold = gold.rename(columns={"Source File/Page": "source_file"})
    bucket_of = {sf: _code_switch_bucket(nar) for sf, nar in
                zip(gold["source_file"], gold["Narrative/Summary"])}

    for run in manifest["runs"]:
        df = pd.read_csv(MODELS / "metrics_preds_{}.csv".format(run["safe"]))
        P = np.load(MODELS / "metrics_probs_{}.npy".format(run["safe"]))
        t_idx = np.array([types.index(t) for t in df.gold_type])
        g_idx = np.array([g_index[g] for g in df.gold_group])
        Pg = P @ M

        pred_t = df.pred_type.fillna("Others / Miscellaneous").values
        pred_g = df.pred_group.fillna("Catch-all").values
        correct = (pred_t == df.gold_type.values)
        bucket = df["source_file"].map(bucket_of).fillna("unclear")
        code_switch = {
            b: {"n": int((bucket == b).sum()),
                "accuracy": round(float(correct[bucket == b].mean()), 4)}
            for b in sorted(bucket.unique()) if (bucket == b).sum()}

        entry = {"tag": run["tag"], "label": LABELS.get(run["tag"], run["tag"]),
                 "seconds": run["seconds"],
                 "latency_median_s": run.get("latency_median_s"),
                 "latency_p95_s": run.get("latency_p95_s"),
                 "throughput_median_toks": run.get("throughput_median_toks"),
                 "peak_vram_mib": run.get("peak_vram_mib"),
                 "peak_ram_mib": run.get("peak_ram_mib"),
                 "n_unparsed": int(df.pred_type.isna().sum()),
                 "n_reconstructed": int(df.reconstructed.sum()),
                 "type": _score(df.gold_type.values, pred_t, t_idx, P, types),
                 "group": _score(df.gold_group.values, pred_g, g_idx, Pg, groups),
                 "calibration": _calibration(P, t_idx),
                 "code_switch": code_switch,
                 "confidence": {
                     "correct": [round(float(x), 4) for x in
                                 P.max(axis=1)[P.argmax(axis=1) == t_idx]],
                     "wrong": [round(float(x), 4) for x in
                               P.max(axis=1)[P.argmax(axis=1) != t_idx]]}}
        cm = np.zeros((len(groups), len(groups)), dtype=int)
        for gg, pg in zip(df.gold_group, pred_g):
            cm[g_index[gg], g_index[pg]] += 1
        entry["group_confusion"] = cm.tolist()
        out["models"].append(entry)

    out["models"].sort(key=lambda m: -m["type"]["accuracy"])
    (MODELS / "classifier_metrics.json").write_text(json.dumps(out, indent=2))

    hdr = "{:<34} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8}".format(
        "model", "acc", "prec", "rec", "F1(W)", "F1(M)", "AUC", "loss")
    print(hdr)
    print("-" * len(hdr))
    for m in out["models"]:
        t = m["type"]
        print("{:<34} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>6.1%} {:>7.3f} {:>8.3f}".format(
            m["label"][:34], t["accuracy"], t["precision_weighted"],
            t["recall_weighted"], t["f1_weighted"], t["f1_macro"], t["auc_macro"],
            t["eval_loss"]))

    hdr2 = "{:<34} {:>10} {:>10} {:>12} {:>10} {:>10}".format(
        "model", "lat p50", "lat p95", "tok/s(med)", "VRAM MiB", "RAM MiB")
    print("\n" + hdr2)
    print("-" * len(hdr2))
    for m in out["models"]:
        print("{:<34} {:>9.2f}s {:>9.2f}s {:>12} {:>10} {:>10}".format(
            m["label"][:34], m["latency_median_s"] or 0.0, m["latency_p95_s"] or 0.0,
            m["throughput_median_toks"] or "n/a", m["peak_vram_mib"] or "n/a",
            m["peak_ram_mib"] or "n/a"))
        cs = m["code_switch"]
        print("  code-switch: " + ", ".join(
            f"{b}={cs[b]['accuracy']:.1%}(n={cs[b]['n']})" for b in cs))
    print("\nwrote models/classifier_metrics.json")


if __name__ == "__main__":
    main()
