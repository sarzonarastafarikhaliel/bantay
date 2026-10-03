"""Print the measured half of the ISO 25010 scorecard.

Usage:  python tools/iso25010_metrics.py

Reads whatever the training notebook has produced so far and renders it in the
order of docs/ISO25010_EVALUATION.md section 2, ready to paste into Chapter 4.
Missing artifacts print as "not measured" rather than failing - a partial
scorecard mid-project is the normal state, and crashing here would only hide it.

It produces no questionnaire scores and deliberately prints no overall average.
See section 1 of the evaluation plan for why measured and perceived numbers stay
in separate columns.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def load(name):
    path = MODELS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def row(label, value, target=""):
    print(f"  {label:<42} {str(value):<28} {target}")


def section(title):
    print(f"\n{title}\n" + "-" * 78)


def pytest_summary():
    """Reliability and testability evidence. Returns the last pytest line."""
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not run pytest ({exc})"
    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    return tail[-1] if tail else "no output"


def db_stats():
    """Capacity evidence: how many records the measured system actually holds."""
    db = ROOT / "instance" / "bantay.db"
    if not db.exists():
        return "no database"
    import sqlite3
    try:
        con = sqlite3.connect(db)
        n = con.execute("SELECT COUNT(*) FROM incident_record").fetchone()[0]
        flagged = con.execute(
            "SELECT COUNT(*) FROM incident_record WHERE review_status = 'needs_review'"
        ).fetchone()[0]
        con.close()
        return f"{n} records, {flagged} flagged for review ({db.stat().st_size/2**20:.1f} MB)"
    except sqlite3.Error as exc:
        return f"unreadable ({exc})"


def main():
    metrics = load("iso25010_metrics.json")
    evalrep = load("eval_report.json")
    ocrrep = load("ocr_correction_report.json")

    print("=" * 78)
    print("BANTAY - ISO/IEC 25010:2023 measured scorecard")
    print(f"generated {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)
    print("  Perceived (questionnaire) scores are NOT here by design - keep the two")
    print("  columns separate. See docs/ISO25010_EVALUATION.md section 1.")

    section("2.1 Functional Suitability")
    if evalrep:
        real = evalrep.get("real_test") or {}
        row("type accuracy (real test)", real.get("type_accuracy", "-"))
        row("type macro F1 (real test)", real.get("type_macro_f1", "-"), "target >= 0.60")
        row("group macro F1 (real test)", real.get("group_macro_f1", "-"), "target >= 0.75")
        row("n real test records", real.get("n", "-"), "state this in the caption")
        row("axes agree rate", real.get("axes_agree_rate", "-"))
        row("type accuracy when axes agree", real.get("type_accuracy_when_agree", "-"),
            "the auto-accepted subset")
        e2e_raw = (evalrep.get("end_to_end_uncorrected") or {}).get("type_accuracy")
        e2e_cor = (evalrep.get("end_to_end_corrected") or {}).get("type_accuracy")
        if e2e_raw is not None:
            row("end-to-end accuracy, uncorrected OCR", e2e_raw)
            row("end-to-end accuracy, corrected OCR", e2e_cor, "the corrector's value")
    else:
        row("classification metrics", "not measured", "run notebook stage 5")

    if ocrrep:
        best = ocrrep.get("best", {})
        before, after = best.get("cer_before"), best.get("cer_after")
        rel = f"{(before - after) / before * 100:.1f}% relative" if before else "-"
        row("CER before correction", before)
        row("CER after correction", after, f"target >= 20% relative, got {rel}")
        row("edit precision", best.get("edit_precision", "-"), "target >= 0.80")
        row("lm_weight selected", best.get("lm_weight", "-"))
        gem = ocrrep.get("gemini")
        if gem:
            gdelta = gem["cer_corrector"] - gem["cer_gemini"]
            row("CER with LLM verification", gem["cer_gemini"],
                f"{gdelta:+.4f} vs corrector alone, {gem['n_pages']} real pages")
            row("LLM edit precision", gem.get("edit_precision", "-"), "same >= 0.80 bar")
            # The number that decides whether an LLM belongs anywhere near a legal
            # record. Anything above zero gets read by eye, not rounded away.
            row("LLM edits that missed the gold token", gem.get("fabrications", "-"), "target 0")
        else:
            row("LLM verification arm", "not measured",
                "optional; needs GEMINI_API_KEY, run notebook stage 3b")
    else:
        row("OCR correction metrics", "not measured", "run notebook stage 3")

    section("2.2 Performance Efficiency")
    if metrics:
        perf = metrics.get("performance_efficiency", {})
        for name in ("correction_latency", "classification_latency"):
            block = perf.get(name)
            row(f"{name} mean / p95 (ms)",
                f"{block['mean_ms']} / {block['p95_ms']}" if block else "not measured")
        for name, size in (metrics.get("artifact_sizes_mb") or {}).items():
            row(f"artifact size: {name}", f"{size} MB")
        row("platform", metrics.get("platform", {}).get("gpu", "-"))
    else:
        row("latency and artifact sizes", "not measured", "run notebook stage 6")

    section("2.2 Capacity / 2.8 Scalability")
    row("database", db_stats(), "target >= 5,000 records")

    section("2.5 Reliability / 2.7 Testability")
    row("pytest", pytest_summary(), "target 100% pass")

    section("2.6 Security (inspection - verify by hand, then tick)")
    for check in ("every record route requires login",
                  "delete restricted to admin role",
                  "bulk delete requires typed confirmation",
                  "SECRET_KEY set from the environment in deployment",
                  "person names masked before training (rule-based masking)"):
        row(check, "[ ] verify")

    section("2.9 Safety (inspection)")
    for check in ("scan never auto-commits a record",
                  "three review triggers active (OCR conf, model conf, axis disagreement)",
                  "every correction shown before save",
                  "raw OCR text still offered when the pipeline fails"):
        row(check, "[ ] verify")

    print("\nRemaining characteristics (Compatibility, Interaction Capability,")
    print("Maintainability, Flexibility) are inspection or questionnaire items -")
    print("see docs/ISO25010_EVALUATION.md sections 2.3, 2.4, 2.7, 2.8.\n")


if __name__ == "__main__":
    main()
