"""
Second-step verification for the PNP crime-classification lookup table.

bantay.normalize.PNP_CLASSIFICATION is a hand-maintained dict; this script
cross-checks it against data/pnp_legal_reference.json — a reference table
built from web-sourced legal text (RPC articles, RA statutes, official
titles) rather than from memory — and reports any category whose tier or
citation has drifted from the verified reference.

Run after editing PNP_CLASSIFICATION or the reference file:
    python -m bantay.ml.verify_pnp_classification
"""
import json
import os
import sys

from ..normalize import PNP_CLASSIFICATION

REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pnp_legal_reference.json"
)


def verify(reference_path=REFERENCE_PATH):
    """Compare PNP_CLASSIFICATION against the reference file.

    Returns a list of mismatch dicts: {category, field, code_value, reference_value}.
    Empty list means everything matches.
    """
    with open(reference_path, encoding="utf-8") as f:
        reference = json.load(f)["categories"]

    mismatches = []
    for category, ref in reference.items():
        if category not in PNP_CLASSIFICATION:
            mismatches.append({
                "category": category, "field": "presence",
                "code_value": "(missing)", "reference_value": "present",
            })
            continue
        code_tier, code_citation = PNP_CLASSIFICATION[category]
        if code_tier != ref["tier"]:
            mismatches.append({
                "category": category, "field": "tier",
                "code_value": code_tier, "reference_value": ref["tier"],
            })
        if code_citation != ref["citation"]:
            mismatches.append({
                "category": category, "field": "citation",
                "code_value": code_citation, "reference_value": ref["citation"],
            })

    for category in PNP_CLASSIFICATION:
        if category not in reference:
            mismatches.append({
                "category": category, "field": "presence",
                "code_value": "present", "reference_value": "(missing — unverified category)",
            })

    return mismatches


def main():
    mismatches = verify()
    if not mismatches:
        print(f"OK — {len(PNP_CLASSIFICATION)} categories match {REFERENCE_PATH}")
        return
    print(f"MISMATCH — {len(mismatches)} discrepancies found:")
    for m in mismatches:
        print(f"  [{m['category']}] {m['field']}: code={m['code_value']!r} vs reference={m['reference_value']!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
