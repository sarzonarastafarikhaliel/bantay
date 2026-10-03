"""Per-field reconciliation between the two independent field readers.

Why this exists: routes/scan.py already runs two readers over every page - the
deterministic regex extractor in scan.guess_fields() (grounded: it can never
invent a value, but it is limited to what a fixed pattern can find) and
Gemini's restore()/restore_from_image() fields (can read cursive and context,
but answers in its own words). Until now narrative.backfill() silently picked
a winner per slot and threw away whether the two readers agreed - agreement is
a free, calibrated confidence signal, exactly the trick the dual-axis
classifier already uses (docs/REIMPLEMENTATION.md), and OCR_ACCURACY_PLAN.md
Lever 3 / Phase 3 names it as the architecture change that beats the 54.3%
text-only ceiling. This module makes the signal explicit instead of implicit.

The four outcomes, matching Lever 3's table:

  both present, same (normalised)   auto-accept   source=both,  agree=True
  both present, differ              review        source=model or regex,
                                                   agree=False (see
                                                   PREFER_REGEX for which wins)
  only one present                  review        source=whichever answered,
                                                   agree=False - one reader
                                                   finding something the other
                                                   missed is still a
                                                   disagreement worth a look,
                                                   not a free pass
  neither present                   blank         source=none,  agree=True -
                                                   nothing to disagree about
"""
from .. import narrative

# The regex reader (scan.guess_fields) only ever answers these five slots -
# reporting_party, respondent and incident_summary have no regex counterpart
# by design (free text/names, not something a fixed pattern can extract), so
# "the model answered and the regex didn't" on those three is architecture,
# not a disagreement worth flagging to the encoder.
# The regex reads the page's first date as date_reported - see guess_fields.
RECONCILE_SLOTS = ("date_reported", "time", "location_purok", "status", "action_taken")

# status/action_taken: measured via tools/run_gemini_arms.py, the regex reader
# beats both Gemini arms under exact-match gold scoring (see
# narrative.PREFER_FALLBACK for the numbers and the reason - Gemini answers in
# its own words instead of the barangay's closed vocabulary/fixed phrasing).
# On a disagreement for these two slots specifically, take the regex value.
PREFER_REGEX = narrative.PREFER_FALLBACK


def _same(a, b):
    return a.strip().upper() == b.strip().upper()


def reconcile(model_fields, regex_fields, prefer_regex=PREFER_REGEX):
    """One row per slot: {"value", "source", "agree"}.

    Both inputs are expected already normalised to the same format (dates as
    YYYY-MM-DD, etc.) - callers should run normalize_fields() on the model's
    raw answer first, since guess_fields()'s regex output already is.
    """
    slots = set(model_fields or {}) | set(regex_fields or {})
    out = {}
    for slot in slots:
        m = str((model_fields or {}).get(slot) or "").strip()
        r = str((regex_fields or {}).get(slot) or "").strip()
        if m and r:
            if _same(m, r):
                out[slot] = {"value": m, "source": "both", "agree": True}
            elif slot in prefer_regex:
                out[slot] = {"value": r, "source": "regex", "agree": False}
            else:
                out[slot] = {"value": m, "source": "model", "agree": False}
        elif m or r:
            out[slot] = {"value": m or r, "source": "model" if m else "regex", "agree": False}
        else:
            out[slot] = {"value": "", "source": "none", "agree": True}
    return out


def fields_only(recon):
    """Drop the source/agree metadata - the shape narrative.render() wants."""
    return {slot: row["value"] for slot, row in recon.items()}


def disagreements(recon):
    """RECONCILE_SLOTS where the two readers gave a real, differing answer.

    Excludes the "only one reader answered" case's blank half and the
    both-blank case - a page mentioning a field zero times is not a
    disagreement, it is the field genuinely not being on the page. Also
    excludes slots outside RECONCILE_SLOTS - see that constant's docstring.
    """
    return [slot for slot in RECONCILE_SLOTS
            if slot in recon and not recon[slot]["agree"] and recon[slot]["value"]]
