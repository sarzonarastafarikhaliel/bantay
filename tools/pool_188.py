"""Review item R2: score the five classifier candidates on all 188 gold records.

The 151/37 dev/holdout split gains nothing for a classifier that is never trained, so
this pools the two. Nothing is re-queried: tools/eval_classifier_metrics.py already wrote
one dev and one holdout prediction file per candidate under the frozen prompt, label set,
schema and temperature 0.0, and together they cover the 188 records exactly. This script
only reads those files, the gold file and data/gold_split.csv, and writes to
results/pooled_188/. The existing dev/holdout outputs in models/ are left untouched.

Statistics: Wilson 95% intervals, a 10,000-resample bootstrap as a cross-check, exact
McNemar tests (two-sided binomial on the discordant pairs) of Gemma 4 against each
SEA-LION checkpoint with Holm's correction, and a paired bootstrap of the difference.

    python tools/pool_188.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay.ml.sealion_classify import CHECK_MODEL, EVALUATED_MODELS, _DEFAULT_MODEL
from bantay.normalize import get_category_group
from tools.eval_gemini_classify import LABEL_BRIDGE

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "results" / "pooled_188"
SEED = 20261010
N_BOOT = 10_000
Z95 = 1.959963984540054

# Short column prefixes for the pooled CSV, in EVALUATED_MODELS order.
KEYS = {
    "gemma4:e4b": "gemma4_e4b",
    "aisingapore/Gemma-SEA-LION-v3-9B-IT:q4_k_m": "sealion_v3_9b_it",
    "aisingapore/llama-sea-lion-v3.5-8b-r": "llama_sealion_v3_5_8b_r",
    "aisingapore/Gemma-SEA-LION-v4-4B-VL": "sealion_v4_4b_vl",
    "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT": "sealion_v4_5_e2b_it",
}
LABELS = {KEYS[m["tag"]]: m["label"] for m in EVALUATED_MODELS}
DEPLOYED = KEYS[_DEFAULT_MODEL]
CHECK = KEYS[CHECK_MODEL]

# Published agreement figures (thesis Table J4) the reused files must reproduce.
PUBLISHED_AGREEMENT = {"dev": {"n_agree": 81, "agree_wrong": 12, "n_disagree": 70},
                       "holdout": {"n_agree": 30, "agree_wrong": 5, "n_disagree": 7}}


def _safe(tag):
    """File-name stem, same transform as tools/eval_classifier_metrics.run_model."""
    return tag.replace("/", "_").replace(":", "_").replace(".", "_").lower()


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(correct, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% interval of the mean of a 0/1 vector."""
    x = np.asarray(correct, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return (float(lo), float(hi))


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, seed=SEED):
    """Accuracy difference a - b with a percentile 95% interval, resampling records
    (both models' outcomes on a record move together, which keeps the pairing)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (float(a.mean() - b.mean()), float(lo), float(hi))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p: binomial test of b against b + c at p = 0.5."""
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def holm(pvals):
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def pool(gold, split, preds):
    """Wide per-record table. gold: source_file, gold_type; split: source_file, split;
    preds: {key: DataFrame with source_file, pred_type, pred_group}. An unparsed reply
    (pred NaN) is never equal to a gold label, so it scores as wrong."""
    out = gold.merge(split, on="source_file", how="left", validate="1:1")
    assert out["split"].notna().all(), "gold record missing from the split file"
    out["gold_group"] = out["gold_type"].map(get_category_group)
    for key, df in preds.items():
        assert df["source_file"].is_unique, key
        assert set(df["source_file"]) == set(out["source_file"]), key
        m = out[["source_file"]].merge(df[["source_file", "pred_type", "pred_group"]],
                                       on="source_file", how="left")
        out[f"{key}_predicted_type"] = m["pred_type"].values
        out[f"{key}_predicted_group"] = m["pred_group"].values
        out[f"{key}_correct_type"] = m["pred_type"].values == out["gold_type"].values
        out[f"{key}_correct_group"] = m["pred_group"].values == out["gold_group"].values
    cols = ["source_file", "split", "gold_type", "gold_group"] + [
        c for c in out.columns if c not in ("source_file", "split", "gold_type", "gold_group")]
    return (out[cols].rename(columns={"source_file": "record_id"})
            .sort_values("record_id").reset_index(drop=True))


def load_pooled():
    """Read gold, split and the ten existing prediction files (read-only) and pool them.
    Also returns the record ids of transport failures per model."""
    gold = pd.read_csv(ROOT / "data/raw/bantay_ml_ready_gold.csv")
    gold = gold.rename(columns={"Source File/Page": "source_file"})
    gold["gold_type"] = gold["Incident Type"].map(lambda t: LABEL_BRIDGE.get(t, t))
    split = pd.read_csv(ROOT / "data/gold_split.csv")
    gold_of = dict(zip(gold["source_file"], gold["gold_type"]))

    preds, errors = {}, {}
    for m in EVALUATED_MODELS:
        parts = []
        for suffix, name in (("", "dev"), ("_holdout", "holdout")):
            df = pd.read_csv(MODELS_DIR / f"metrics_preds_{_safe(m['tag'])}{suffix}.csv")
            assert set(df["source_file"]) == set(split.loc[split["split"] == name, "source_file"]), \
                (m["tag"], name)
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        assert (df["gold_type"] == df["source_file"].map(gold_of)).all(), \
            f"{m['tag']}: gold label in prediction file differs from gold file"
        key = KEYS[m["tag"]]
        preds[key] = df
        errors[key] = sorted(df.loc[df["error"].notna(), "source_file"])
    return pool(gold[["source_file", "gold_type"]], split, preds), errors


def _acc(correct):
    k, n = int(np.sum(correct)), len(correct)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "acc": k / n if n else float("nan"), "wilson": [lo, hi]}


def mcnemar_cc(b, c):
    """Continuity-corrected McNemar chi-square (1 df) and p, the form the thesis's
    dev-split tables use; reported beside the exact test for comparability."""
    if b + c == 0:
        return 0.0, 1.0
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return chi, float(chi2.sf(chi, 1))


def agreement(df, check=CHECK):
    """Deployed model's accuracy split by whether the check partner names the same type."""
    g, c = df[f"{DEPLOYED}_predicted_type"], df[f"{check}_predicted_type"]
    ok = df[f"{DEPLOYED}_correct_type"].values
    agree = (g == c).values
    group_agree = (df[f"{DEPLOYED}_predicted_group"] == df[f"{check}_predicted_group"]).values
    mid, low = ~agree & group_agree, ~agree & ~group_agree
    return {
        "n": len(df), "n_agree": int(agree.sum()), "n_disagree": int((~agree).sum()),
        "agree": _acc(ok[agree]), "disagree": _acc(ok[~agree]),
        "agree_wrong": int((agree & ~ok).sum()), "flagged_share": float((~agree).mean()),
        "confusion": {"agree_correct": int((agree & ok).sum()),
                      "agree_wrong": int((agree & ~ok).sum()),
                      "disagree_correct": int((~agree & ok).sum()),
                      "disagree_wrong": int((~agree & ~ok).sum())},
        "three_tier": {"type_agree": _acc(ok[agree]),
                       "type_disagree_group_agree": _acc(ok[mid]),
                       "type_disagree_group_disagree": _acc(ok[low])},
    }


def summarise(pooled, errors):
    subsets = {"pooled": pooled, "dev": pooled[pooled["split"] == "dev"],
               "holdout": pooled[pooled["split"] == "holdout"]}
    keys = list(LABELS)
    stats = {"seed": SEED, "n_boot": N_BOOT, "n": len(pooled), "models": {}, "paired": {},
             "agreement": {}, "sensitivity": {}, "transport_errors": errors}
    for key in keys:
        entry = {"label": LABELS[key]}
        for grain in ("type", "group"):
            col = pooled[f"{key}_correct_{grain}"].values
            entry[grain] = {name: _acc(sub[f"{key}_correct_{grain}"].values)
                            for name, sub in subsets.items()}
            entry[grain]["pooled"]["bootstrap"] = list(bootstrap_ci(col))
        stats["models"][key] = entry

    for grain in ("type", "group"):
        a = pooled[f"{DEPLOYED}_correct_{grain}"].values
        rows = []
        for key in keys[1:]:
            o = pooled[f"{key}_correct_{grain}"].values
            b, c = int((a & ~o).sum()), int((~a & o).sum())
            diff, lo, hi = paired_bootstrap_diff(a, o)
            chi, p_cc = mcnemar_cc(b, c)
            rows.append({"other": key, "b": b, "c": c, "p_exact": mcnemar_exact(b, c),
                         "chi2_cc": chi, "p_cc": p_cc, "diff": diff, "diff_ci": [lo, hi]})
        for row, adj, adj_cc in zip(rows, holm([r["p_exact"] for r in rows]),
                                    holm([r["p_cc"] for r in rows])):
            row["p_holm"], row["p_cc_holm"] = adj, adj_cc
        stats["paired"][grain] = rows

    for name, sub in subsets.items():
        stats["agreement"][name] = agreement(sub)
    # Every SEA-LION checkpoint as a would-be check partner (thesis Tables 17 and J4).
    stats["agreement_by_partner"] = {
        key: {name: agreement(sub, key) for name, sub in subsets.items()} for key in keys[1:]}

    # Sensitivity: drop each model's transport failures (ReadTimeout etc.) rather than
    # scoring them wrong, and re-run that model's paired test on the remaining records.
    for key, ids in errors.items():
        if not ids:
            continue
        keep = ~pooled["record_id"].isin(ids)
        sub = pooled[keep]
        a = sub[f"{DEPLOYED}_correct_type"].values
        o = sub[f"{key}_correct_type"].values
        b, c = int((a & ~o).sum()), int((~a & o).sum())
        stats["sensitivity"][key] = {"excluded": ids, "type": _acc(o),
                                     "deployed_type": _acc(a), "b": b, "c": c,
                                     "p_exact": mcnemar_exact(b, c)}
    return stats


def check_against_published(stats):
    """The reused files must reproduce the figures already in the thesis."""
    for name, fn in (("dev", "classifier_metrics.json"), ("holdout", "classifier_metrics_holdout.json")):
        published = json.loads((MODELS_DIR / fn).read_text())
        for m in published["models"]:
            mine = stats["models"][KEYS[m["tag"]]]
            for grain in ("type", "group"):
                assert abs(mine[grain][name]["acc"] - m[grain]["accuracy"]) < 1e-9, \
                    (m["tag"], name, grain)
        ag = stats["agreement"][name]
        for k, v in PUBLISHED_AGREEMENT[name].items():
            assert ag[k] == v, (name, k, ag[k], v)
    # Table J4, dev rows: n agree per partner.
    for key, n_agree in (("llama_sealion_v3_5_8b_r", 89), ("sealion_v4_4b_vl", 82),
                         ("sealion_v4_5_e2b_it", 74)):
        assert stats["agreement_by_partner"][key]["dev"]["n_agree"] == n_agree, key


def _pct(x):
    return f"{100 * x:.1f}%"


def _ci(lo_hi):
    return f"{100 * lo_hi[0]:.1f}–{100 * lo_hi[1]:.1f}"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tables_md(stats):
    m, out = stats["models"], []
    out.append("### Table A. Accuracy of the five candidates on all 188 gold records\n")
    out.append("| Candidate | Type acc. (k/188) | Wilson 95% CI | Bootstrap 95% CI "
               "| Group acc. (k/188) | Wilson 95% CI | Bootstrap 95% CI |")
    out.append("|---|---|---|---|---|---|---|")
    for key, e in m.items():
        t, g = e["type"]["pooled"], e["group"]["pooled"]
        out.append(f"| {e['label']} | {_pct(t['acc'])} ({t['k']}) | {_ci(t['wilson'])} "
                   f"| {_ci(t['bootstrap'])} | {_pct(g['acc'])} ({g['k']}) | {_ci(g['wilson'])} "
                   f"| {_ci(g['bootstrap'])} |")
    out.append(f"\n*Note.* n = 188 (151 dev + 37 holdout). Wilson score intervals; bootstrap "
               f"is the 2.5th–97.5th percentile of {stats['n_boot']:,} record resamples "
               f"(seed {stats['seed']}). Unparsed replies and transport failures are scored "
               f"as wrong, as in the published dev/holdout tables.\n")

    out.append("### Table B. Paired comparison of Gemma 4 with each SEA-LION checkpoint (n = 188)\n")
    out.append("| Comparison | Grain | b (Gemma 4 right, other wrong) | c (Gemma 4 wrong, other right) "
               "| Exact McNemar p | Holm-adjusted p | Difference (pp) | Paired bootstrap 95% CI (pp) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for grain in ("type", "group"):
        for r in stats["paired"][grain]:
            out.append(f"| Gemma 4 vs {m[r['other']]['label']} | {grain.capitalize()} | {r['b']} "
                       f"| {r['c']} | {_p(r['p_exact'])} | {_p(r['p_holm'])} "
                       f"| {100 * r['diff']:+.1f} | {100 * r['diff_ci'][0]:+.1f} to "
                       f"{100 * r['diff_ci'][1]:+.1f} |")
    out.append("\n*Note.* Exact (binomial) McNemar test on the discordant pairs, two-sided. "
               "Holm's correction across the four checkpoints within each grain. Positive "
               "difference = Gemma 4 more accurate.\n")

    for grain, title in (("type", "incident-type"), ("group", "Category Group")):
        out.append(f"### Table C{'1' if grain == 'type' else '2'}. Dev, holdout and pooled "
                   f"{title} accuracy\n")
        out.append("| Candidate | Dev (n = 151) | Holdout (n = 37) | Pooled (n = 188) |")
        out.append("|---|---|---|---|")
        for key, e in m.items():
            cells = [f"{_pct(e[grain][s]['acc'])} ({e[grain][s]['k']}; {_ci(e[grain][s]['wilson'])})"
                     for s in ("dev", "holdout", "pooled")]
            out.append(f"| {e['label']} | " + " | ".join(cells) + " |")
        out.append("\n*Note.* Accuracy (correct count; Wilson 95% CI).\n")

    out.append("### Table D. Gemma 4 accuracy by agreement with the check partner "
               "(Gemma-SEA-LION-v3-9B-IT)\n")
    out.append("| Records | n | Agree n | Accuracy when agree (Wilson 95% CI) | Disagree n "
               "| Accuracy when disagree (Wilson 95% CI) | Agreed but wrong | Flagged share |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name in ("dev", "holdout", "pooled"):
        a = stats["agreement"][name]
        out.append(f"| {name.capitalize()} | {a['n']} | {a['n_agree']} | {_pct(a['agree']['acc'])} "
                   f"({_ci(a['agree']['wilson'])}) | {a['n_disagree']} | {_pct(a['disagree']['acc'])} "
                   f"({_ci(a['disagree']['wilson'])}) | {a['agree_wrong']} of {a['n_agree']} "
                   f"({_pct(a['agree_wrong'] / a['n_agree'])}) | {_pct(a['flagged_share'])} |")
    a = stats["agreement"]["pooled"]
    cm = a["confusion"]
    out.append("\n**Confusion counts, pooled (n = 188).**\n")
    out.append("| | Gemma 4 correct | Gemma 4 wrong | Total |")
    out.append("|---|---|---|---|")
    out.append(f"| Models agree (\"high\") | {cm['agree_correct']} | {cm['agree_wrong']} | {a['n_agree']} |")
    out.append(f"| Models disagree (\"low\") | {cm['disagree_correct']} | {cm['disagree_wrong']} "
               f"| {a['n_disagree']} |")
    out.append(f"| Total | {cm['agree_correct'] + cm['disagree_correct']} "
               f"| {cm['agree_wrong'] + cm['disagree_wrong']} | {a['n']} |")
    out.append("\n**Three-tier check, pooled (n = 188).**\n")
    out.append("| Tier | n | Gemma 4 accuracy (Wilson 95% CI) |")
    out.append("|---|---|---|")
    for name, label in (("type_agree", "Type agrees"),
                        ("type_disagree_group_agree", "Type disagrees, Category Group agrees"),
                        ("type_disagree_group_disagree", "Type disagrees, Category Group disagrees")):
        t = a["three_tier"][name]
        out.append(f"| {label} | {t['n']} | {_pct(t['acc'])} ({_ci(t['wilson'])}) |")
    out.append("")

    if stats["sensitivity"]:
        out.append("### Sensitivity: transport failures excluded\n")
        for key, s in stats["sensitivity"].items():
            out.append(f"- {m[key]['label']}: {len(s['excluded'])} ReadTimeout record(s) "
                       f"({', '.join(s['excluded'])}) dropped. Type accuracy "
                       f"{_pct(s['type']['acc'])} ({s['type']['k']}/{s['type']['n']}; Wilson "
                       f"{_ci(s['type']['wilson'])}). Gemma 4 on the same {s['type']['n']}: "
                       f"{_pct(s['deployed_type']['acc'])}. Exact McNemar b = {s['b']}, "
                       f"c = {s['c']}, p = {_p(s['p_exact'])}.")
        out.append("")
    return "\n".join(out)


def main():
    pooled, errors = load_pooled()
    stats = summarise(pooled, errors)
    check_against_published(stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(OUT_DIR / "pooled_per_record.csv", index=False, encoding="utf-8")
    (OUT_DIR / "pooled_stats.json").write_text(json.dumps(stats, indent=2))
    tables = tables_md(stats)
    (OUT_DIR / "R2_tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/pooled_per_record.csv, pooled_stats.json, R2_tables.md")


if __name__ == "__main__":
    main()
