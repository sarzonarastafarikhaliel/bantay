"""Appendix K, Tables K1 and K2: how much of the gold narrative survives into the
narrative field (E. Salaysay) each Gemini arm extracts, and where the extracted
words can be traced. Uses the token accounting of tools/audit_ocr_gold.py.

    python tools/audit_arms_narrative.py                       # tag all188 (188 pages)
    python tools/audit_arms_narrative.py --tag ""              # the original 151-page files
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.audit_ocr_gold import FUZZY_FLOOR, MIN_TOKEN_LEN, _MASK_RE, account_tokens, normalise


def tokens(text):
    return [w for w in normalise(text).split() if len(w) >= MIN_TOKEN_LEN]


def k1(texts, calib):
    """Word-weighted exact / garbled / absent shares of gold content words."""
    n = ex = ga = ab = 0
    for src, text in texts.items():
        g, e, gb, a = account_tokens(str(text if isinstance(text, str) else ""), str(calib.loc[src, "gold_text"]))
        n, ex, ga, ab = n + g, ex + e, ga + gb, ab + a
    return {"gold_words": n, "exact": ex / n, "garbled": ga / n, "absent": ab / n}


def k2(texts, calib):
    """Each content word of the extracted narrative: in gold, else in Vision's page, else neither."""
    in_gold = in_vision = neither = 0
    for src, text in texts.items():
        gold = list(set(tokens(_MASK_RE.sub(" ", str(calib.loc[src, "gold_text"])))))
        vision = list(set(tokens(str(calib.loc[src, "ocr_text"]))))
        for w in tokens(str(text if isinstance(text, str) else "")):
            if w in gold or difflib.get_close_matches(w, gold, n=1, cutoff=FUZZY_FLOOR):
                in_gold += 1
            elif w in vision or difflib.get_close_matches(w, vision, n=1, cutoff=FUZZY_FLOOR):
                in_vision += 1
            else:
                neither += 1
    total = in_gold + in_vision + neither
    return {"total": total, "in_gold": in_gold, "in_vision": in_vision, "neither": neither}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all188", help="suffix of models/preds_gemini_<arm>_<tag>.csv; '' for untagged")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    calib = pd.read_csv(ROOT / "data/ocr_calibration.csv").set_index("source_file")
    arms = {arm: pd.read_csv(ROOT / f"models/preds_gemini_{arm}{suffix}.csv").set_index("source_file")["incident_summary"]
            for arm in ("text", "image")}
    pages = arms["image"].index
    out = {"pages": len(pages),
           "vision": k1(calib.loc[pages, "ocr_text"], calib),
           "text": k1(arms["text"], calib),
           "image": k1(arms["image"], calib),
           "image_trace": k2(arms["image"], calib),
           "vision_words_per_page": round(sum(len(str(t).split()) for t in calib.loc[pages, "ocr_text"]) / len(pages), 1)}
    for k in ("vision", "text", "image"):
        r = out[k]
        print(f"{k:<7} n={r['gold_words']:>5}  exact {r['exact']:.1%}  garbled {r['garbled']:.1%}  absent {r['absent']:.1%}")
    t = out["image_trace"]
    print(f"trace   total {t['total']:,}  gold {t['in_gold']:,} ({t['in_gold']/t['total']:.1%})  "
          f"vision {t['in_vision']:,} ({t['in_vision']/t['total']:.1%})  neither {t['neither']:,} ({t['neither']/t['total']:.1%})")
    print(f"vision words/page {out['vision_words_per_page']}")
    if args.tag:
        path = ROOT / f"results/pooled_188/arms_narrative_{args.tag}.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
