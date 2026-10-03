"""Full metric sweep (accuracy, precision, recall, F1, AUC-ROC, eval loss) for
the five Ollama classifier candidates already compared in Chapter 4, sec 4.3.2.

Why this exists: tools/eval_sealion_classify.py records only the hard predicted
label, so it can produce accuracy but not AUC-ROC or a cross-entropy loss - both
need a score per class, not a single winner. Ollama 0.12+ returns per-token
logprobs, so this rerun asks the identical prompt, identical JSON schema and
identical temperature=0 as the published comparison, and additionally keeps the
logprobs. From those a distribution over the 35 canonical incident types is
reconstructed (see _distribute), which makes AUC-ROC and log loss genuinely
measured quantities rather than a restatement of accuracy.

    python tools/eval_classifier_metrics.py            # all 5 candidates, dev split
    python tools/eval_classifier_metrics.py --limit 3  # smoke test
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BANTAY_OLLAMA", "1")

from bantay.ml.sealion_classify import TYPE_NAMES, _PROMPT, _SCHEMA, EVALUATED_MODELS
from bantay.normalize import get_category_group
from tools.eval_gemini_classify import LABEL_BRIDGE

HOST = os.environ.get("BANTAY_OLLAMA_HOST", "http://localhost:11434")
GROUP_NAMES = sorted({get_category_group(t) for t in TYPE_NAMES})
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_NAMES)}

# Numerical floor on any single class probability (see _distribute).
PROB_FLOOR = 1e-6


def _load_rows(split, limit):
    gold = pd.read_csv(ROOT / "data/raw/bantay_ml_ready_gold.csv").rename(
        columns={"Source File/Page": "source_file"})
    split_df = pd.read_csv(ROOT / "data/gold_split.csv")
    keep = set(split_df.loc[split_df.split == split, "source_file"])
    rows = gold[gold.source_file.isin(keep)].sort_values("source_file")
    return rows.head(limit) if limit else rows


def _token_spans(logprobs):
    """Char span of each token within the detokenised stream, and that stream.

    Aligning against message.content instead would be wrong for a thinking model:
    gemma4:e4b and Gemma-SEA-LION-v4.5-E2B-IT emit several hundred reasoning
    tokens before the JSON, and Ollama returns logprobs for the whole generation
    while content holds only the final JSON. Concatenating the tokens is the one
    reference text that always corresponds 1:1 to the logprobs array.
    """
    spans, pos, parts = [], 0, []
    for lp in logprobs:
        tok = lp.get("token", "")
        parts.append(tok)
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
    return spans, "".join(parts)


def _distribute(label, logprobs):
    """Probability over TYPE_NAMES reconstructed from the label's token logprobs.

    The emitted label takes its own sequence probability, exp(sum of its token
    logprobs). At every token position inside the label, each *alternative* token
    in top_logprobs that still spells a valid prefix of some other type is
    credited with that alternative's probability, split evenly across the types
    sharing that prefix - the standard beam-marginalisation approximation, since
    the alternatives' own continuations were never generated. Whatever mass is
    left over (alternatives that spell no valid type at all) is spread uniformly
    across the types nothing reached, so the vector is a proper distribution.
    Returns None if the token/char alignment fails.
    """
    spans, stream = _token_spans(logprobs)
    if not spans:
        return None
    needle = json.dumps(label)                      # the quoted value as emitted
    # rfind, not find: a thinking model names candidate types in its reasoning
    # before committing, and the answer is the last occurrence, not the first.
    start = stream.rfind(needle)
    if start < 0:
        return None
    lo, hi = start + 1, start + 1 + len(label)      # inside the quotes
    idxs = [i for i, (a, b) in enumerate(spans) if a < hi and b > lo]
    if not idxs:
        return None

    p = np.zeros(len(TYPE_NAMES))
    prefix, log_prefix = "", 0.0
    for i in idxs:
        tok = logprobs[i].get("token", "")
        for alt in (logprobs[i].get("top_logprobs") or []):
            a = alt.get("token", "")
            if a == tok:
                continue
            cand = [t for t in TYPE_NAMES if t.startswith(prefix + a)]
            if not cand:
                continue
            share = math.exp(log_prefix + alt["logprob"]) / len(cand)
            for t in cand:
                p[TYPE_INDEX[t]] += share
        prefix += tok
        log_prefix += logprobs[i]["logprob"]
        if len(prefix) >= len(label):
            break

    p[TYPE_INDEX[label]] = max(p[TYPE_INDEX[label]], math.exp(log_prefix))
    total = p.sum()
    if total > 1.0:                                  # approximation overshoot
        p /= total
        total = 1.0
    zeros = np.flatnonzero(p == 0)
    if len(zeros):
        p[zeros] = (1.0 - total) / len(zeros)
    # Greedy decoding routinely emits a label whose sequence probability rounds
    # to 1.0 in float64, leaving the other 34 types at exactly 0 and making
    # cross-entropy infinite for any such row that is wrong. Floor every class
    # at PROB_FLOOR so the loss stays finite; the floor caps a single confidently
    # wrong record's contribution at -ln(PROB_FLOOR) nats and has to be reported
    # alongside the loss, since the loss is not floor-independent.
    p = np.maximum(p, PROB_FLOOR)
    s = p.sum()
    return p / s if s > 0 else None


def _gpu_mem_used_mib():
    """Total GPU memory in use system-wide, or None if nvidia-smi is absent."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = [int(x) for x in out.stdout.split()]
        return max(vals) if vals else None
    except Exception:                                  # noqa: BLE001 - no GPU, no driver, etc.
        return None


class _PeakSampler:
    """Background poll of system GPU/RAM use, peak-tracked while a model runs.

    System-wide, not process-scoped: Ollama's server has no per-request memory
    attribution and nvidia-smi reports the whole card, so this is the peak
    observed on the machine during the sweep, not Ollama's exclusive
    footprint - it answers "does this fit", not a per-call profile.
    """
    def __init__(self, interval=1.0):
        import threading
        self.interval = interval
        self.peak_vram_mib = 0
        self.peak_ram_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil
        while not self._stop.is_set():
            v = _gpu_mem_used_mib()
            if v is not None:
                self.peak_vram_mib = max(self.peak_vram_mib, v)
            self.peak_ram_mib = max(self.peak_ram_mib,
                                    psutil.virtual_memory().used // (1024 * 1024))
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_vram_mib, self.peak_ram_mib


def _ask(narrative, model, timeout=600.0):
    prompt = _PROMPT.format(n=len(TYPE_NAMES),
                            types="\n".join("- {}".format(t) for t in TYPE_NAMES),
                            text=narrative)
    r = requests.post("{}/api/chat".format(HOST), timeout=timeout, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": _SCHEMA, "stream": False,
        "logprobs": True, "top_logprobs": 20,
        "options": {"temperature": 0.0},
    })
    r.raise_for_status()
    d = r.json()
    content = d.get("message", {}).get("content") or ""
    try:
        label = json.loads(content).get("incident_type")
    except ValueError:
        label = None
    if label not in TYPE_INDEX:
        label = None
    probs = _distribute(label, d.get("logprobs") or []) if label else None
    reconstructed = probs is not None
    if probs is None:                                # keep the row scorable
        if label:                                    # alignment failed: one-hot
            probs = np.full(len(TYPE_NAMES), PROB_FLOOR)
            probs[TYPE_INDEX[label]] = 1.0 - PROB_FLOOR * (len(TYPE_NAMES) - 1)
        else:                                        # no usable answer: uniform
            probs = np.full(len(TYPE_NAMES), 1.0 / len(TYPE_NAMES))
    # Ollama returns generation token count/duration on every non-streaming
    # response (nanoseconds) - captured here rather than re-measured, since a
    # wall-clock wrapper would also count HTTP/queueing overhead as "decode".
    tok_meta = {
        "eval_count": d.get("eval_count"),
        "eval_duration_s": (d["eval_duration"] / 1e9) if d.get("eval_duration") else None,
        "prompt_eval_count": d.get("prompt_eval_count"),
        "total_duration_s": (d["total_duration"] / 1e9) if d.get("total_duration") else None,
    }
    return label, probs, reconstructed, tok_meta


def run_model(tag, rows, out_dir):
    recs, P = [], []
    sampler = _PeakSampler().start()
    t0 = time.perf_counter()
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        narrative = str(r.get("Narrative/Summary") or "")
        gold = LABEL_BRIDGE.get(r.get("Incident Type"), r.get("Incident Type"))
        call_t0 = time.perf_counter()
        try:
            label, probs, recon, tok_meta = _ask(narrative, tag)
            err = ""
        except Exception as exc:                     # noqa: BLE001 - any HTTP failure
            label, recon, err = None, False, "{}: {}".format(type(exc).__name__, exc)
            probs = np.full(len(TYPE_NAMES), 1.0 / len(TYPE_NAMES))
            tok_meta = {}
        latency_s = time.perf_counter() - call_t0
        recs.append({"source_file": r["source_file"], "gold_type": gold,
                     "gold_group": get_category_group(gold),
                     "pred_type": label,
                     "pred_group": get_category_group(label) if label else None,
                     "p_pred": float(probs[TYPE_INDEX[label]]) if label else None,
                     "reconstructed": recon, "error": err,
                     "latency_s": round(latency_s, 4),
                     "eval_count": tok_meta.get("eval_count"),
                     "eval_duration_s": tok_meta.get("eval_duration_s")})
        P.append(probs)
        print("  [{:>2}/{}] {:<24} gold={:<30} pred={:<30} p={:.3f} {:.2f}s {}".format(
            i, len(rows), r["source_file"], str(gold), str(label),
            recs[-1]["p_pred"] or 0.0, latency_s, err[:40]), flush=True)
    elapsed = time.perf_counter() - t0
    peak_vram_mib, peak_ram_mib = sampler.stop()
    df = pd.DataFrame(recs)
    safe = tag.replace("/", "_").replace(":", "_").replace(".", "_").lower()
    df.to_csv(out_dir / "metrics_preds_{}.csv".format(safe), index=False, encoding="utf-8")
    np.save(out_dir / "metrics_probs_{}.npy".format(safe), np.array(P))
    return df, np.array(P), elapsed, safe, peak_vram_mib, peak_ram_mib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--models", nargs="+",
                    default=[m["tag"] for m in EVALUATED_MODELS])
    args = ap.parse_args()

    out_dir = ROOT / "models"
    rows = _load_rows(args.split, args.limit)
    print("{} records, split={}, {} candidates\n".format(
        len(rows), args.split, len(args.models)), flush=True)

    manifest = {"split": args.split, "n": len(rows), "types": TYPE_NAMES,
                "groups": GROUP_NAMES, "runs": []}
    for tag in args.models:
        print("=== {} ===".format(tag), flush=True)
        df, P, elapsed, safe, peak_vram_mib, peak_ram_mib = run_model(tag, rows, out_dir)
        acc = float((df.pred_type == df.gold_type).mean())

        lat = df["latency_s"].dropna().values
        toks_per_s = (df["eval_count"] / df["eval_duration_s"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        manifest["runs"].append({
            "tag": tag, "safe": safe, "seconds": round(elapsed, 1),
            "acc": acc,
            "reconstructed": int(df.reconstructed.sum()),
            "n_null": int(df.pred_type.isna().sum()),
            "latency_median_s": round(float(np.median(lat)), 3) if len(lat) else None,
            "latency_p95_s": round(float(np.percentile(lat, 95)), 3) if len(lat) else None,
            "throughput_median_toks": round(float(toks_per_s.median()), 1) if len(toks_per_s) else None,
            "peak_vram_mib": peak_vram_mib,
            "peak_ram_mib": peak_ram_mib,
        })
        print("  -> accuracy {:.1%} in {:.0f}s  (latency p50={:.2f}s p95={:.2f}s, "
              "peak VRAM {}MiB, peak RAM {}MiB)\n".format(
            acc, elapsed, np.median(lat) if len(lat) else 0.0,
            np.percentile(lat, 95) if len(lat) else 0.0, peak_vram_mib, peak_ram_mib),
            flush=True)
        (out_dir / "metrics_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote models/metrics_manifest.json")


if __name__ == "__main__":
    main()
