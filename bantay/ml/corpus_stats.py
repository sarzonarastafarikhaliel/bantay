"""Live real/synthetic corpus ratio.

UNUSED as of the banner's removal: the data-provenance banner this fed was
taken out of dashboard.html and records_new.html, along with the context
processor that injected it. Nothing imports this module now. Kept because the
ratio it computes is still a real property of the corpus and cheap to re-wire;
delete it if that is not wanted.


Reads the same corpus file every training path reads (PIPELINE.md pipeline
output, notebook stage 1's CFG["corpus"]), so the number shown to encoders is
never hand-maintained text that drifts from what the model actually trained
on. When more real records get encoded and the corpus gets rebuilt with a
higher real count, this updates on the next app restart - no code change.

Cached at module level: the corpus file does not change while the process is
running, so re-parsing 2,100 rows on every page load would be pure waste.
"""
import csv
import os

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
    "data", "raw", "bantay_corpus_2100.csv")

_cache = None


def get_corpus_stats(path=None):
    """{"real": int, "total": int, "ratio": float, "available": bool}.

    ratio is synthetic-per-real (PIPELINE.md's "27.8:1" framing). available is
    False when the corpus file is missing, so callers can skip the banner
    instead of showing a stat for data that is not there.
    """
    global _cache
    if _cache is not None and path is None:
        return _cache

    corpus_path = path or CORPUS_PATH
    real = total = 0
    try:
        with open(corpus_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("Include in ML", "")).strip().lower() not in ("yes", "true", "1"):
                    continue
                total += 1
                if str(row.get("_provenance", "")).strip().lower() == "real":
                    real += 1
    except (OSError, csv.Error):
        result = {"real": 0, "total": 0, "ratio": 0.0, "available": False}
        if path is None:
            _cache = result
        return result

    synthetic = total - real
    result = {
        "real": real,
        "total": total,
        "synthetic": synthetic,
        "ratio": round(synthetic / real, 1) if real else 0.0,
        "available": total > 0,
    }
    if path is None:
        _cache = result
    return result
