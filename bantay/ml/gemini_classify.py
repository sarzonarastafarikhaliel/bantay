"""Few-shot incident-type classification via Gemini - a second opinion sourced
from general knowledge instead of more labeled blotter data.

Why this exists: at 73 real records, per-class support for the 35-way type
task is in the low single digits (docs/REIMPLEMENTATION.md sec 4), and the
previously trained encoder classifier measured that scarcity directly - 6.7%
real-test accuracy, 10.2% on its own synthetic (in-distribution) test set
(models/eval_report.json, models/clf_ablation.json). It has since been removed;
this module is now the only incident-type classifier in the scan path. No amount of prompting
fixes a training bug if there is one, and this module does not attempt to -
see tools/eval_gemini_classify.py's own docstring for that half. What it does
answer is a different, cheaper question: does a model that has NOT been
fine-tuned on this corpus at all, but has broad knowledge of Philippine legal
categories, do reasonably out of the box? If so, that is worth reporting on
its own regardless of what is wrong with the trained classifier.

Reuses bantay.ocr.gemini's client/backend/retry machinery rather than
duplicating it - same credential, same off-by-default contract, same
"never raises" shape as verify()/restore().
"""
import json
import os

from ..normalize import CANONICAL_CATEGORIES, FALLBACK_CATEGORY, get_category_group
from ..ocr.gemini import THINKING_BUDGET, _client, _http_options, _mode

# CANONICAL_CATEGORIES mixes two things: the 35-type ML taxonomy (what
# the removed encoder classifier was trained on) and 3
# legacy pre-CSV categories kept in the keyword-matcher for raw CSV imports
# only (Work/Service Dispute, Curfew Violation, Amicable Settlement - see that
# module's comment). Offering those 3 here would let Gemini answer with a
# label the trained classifier cannot produce, making the two readers
# incomparable - caught via tests/test_gemini_classify.py checking this set
# against labels.json. FALLBACK_CATEGORY ("Others / Miscellaneous") is the
# opposite gap: it IS one of the 35, but it's a constant, not a
# CANONICAL_CATEGORIES row, so it has to be added back in explicitly.
_LEGACY_ONLY = {"Work/Service Dispute", "Curfew Violation", "Amicable Settlement"}
TYPE_NAMES = [name for name, _ in CANONICAL_CATEGORIES if name not in _LEGACY_ONLY] + [FALLBACK_CATEGORY]

# A different default from ocr.gemini's _DEFAULT_MODEL only in that this
# module keeps its own constant - not a different backend or credential, just
# so a model-ID change for OCR repair doesn't silently retarget classification
# too, since the two tasks may want different tiers.
_DEFAULT_MODEL = "gemini-3.7-flash"

_PROMPT = """You are classifying an incident narrative from a barangay (village) \
blotter logbook in the Philippines. The text is Tagalog/English code-switched \
and often ALL CAPS. Names have already been replaced with tokens like \
[PERSON_7] - ignore those, they carry no information.

Choose exactly ONE incident type from this list of {n}. Use the exact spelling \
given. If genuinely nothing fits, use "Others / Miscellaneous" - never invent \
a type outside this list.

TYPES:
{types}

Return JSON only, in this exact shape:
{{"incident_type": "<one type from the list, exact spelling>", "confidence": "<high|medium|low>", "reason": "<a few words>"}}

NARRATIVE:
{text}
"""


def available():
    """True if a classification call would actually reach a backend."""
    return bool(_mode())


def _ask(narrative, model, timeout):
    from google.genai import types

    prompt = _PROMPT.format(n=len(TYPE_NAMES),
                            types="\n".join(f"- {t}" for t in TYPE_NAMES),
                            text=narrative)
    resp = _client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=_http_options(timeout),
        ),
    )
    return _parse(resp.text)


def _parse(raw):
    """Defensive parse: anything malformed or off-list is no answer, never a
    guess - an incident_type outside TYPE_NAMES would corrupt every downstream
    lookup (get_category_group, PNP tier) that assumes a canonical value."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    label = data.get("incident_type")
    if not isinstance(label, str) or label not in TYPE_NAMES:
        return None
    return {"incident_type": label,
            "confidence_label": str(data.get("confidence", ""))[:10],
            "reason": str(data.get("reason", ""))[:120]}


def classify(narrative, model=None, timeout=60.0):
    """One call. Returns {"incident_type", "category_group", "confidence_label",
    "reason", "status"} and never raises - same contract as ocr.gemini.verify/
    restore: a dead key, dead network or malformed reply degrades to no
    prediction (incident_type=None), and the caller surfaces that rather than
    passing off another model's answer as this one's."""
    if not available():
        return {"incident_type": None, "category_group": None, "status": "off"}
    if not (narrative or "").strip():
        return {"incident_type": None, "category_group": None, "status": "skipped: empty narrative"}

    model = model or os.environ.get("BANTAY_GEMINI_MODEL", _DEFAULT_MODEL)
    try:
        parsed = _ask(narrative, model, timeout)
    except Exception as exc:                      # noqa: BLE001 - any SDK/network failure
        return {"incident_type": None, "category_group": None,
                "status": f"error: {type(exc).__name__}: {exc}"[:200]}

    if not parsed:
        return {"incident_type": None, "category_group": None,
                "status": "ok: no usable answer"}

    return {"incident_type": parsed["incident_type"],
            "category_group": get_category_group(parsed["incident_type"]),
            "confidence_label": parsed["confidence_label"],
            "reason": parsed["reason"],
            "status": f"ok ({model} via {_mode()})"}


if __name__ == "__main__":       # python -m bantay.ml.gemini_classify
    mode = _mode()
    print(f"backend : {mode or 'off'}")
    if mode:
        sample = ("NAGTUNGO DITO SA BARANGAY ITONG SI [PERSON_1] UPANG IREKLAMO ANG "
                  "PAGKUHA NG MANOK NA PANABONG SA GABI. NAKAWALA RIN ANG MANOK NG "
                  "KATABING BAHAY KAY [PERSON_2].")
        out = classify(sample)
        print("in     :", sample)
        print("type   :", out["incident_type"])
        print("group  :", out["category_group"])
        print("status :", out["status"])
