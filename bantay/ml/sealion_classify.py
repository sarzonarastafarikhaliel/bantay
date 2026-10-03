"""Few-shot incident-type classification via a local Ollama model - a second
opinion sourced from general knowledge instead of more labeled blotter data.

Why this exists: at 188 real records, per-class support for the 35-way type
task is in the low single digits for most categories (docs/REIMPLEMENTATION.md
sec 4), and the previously trained encoder classifier measured that scarcity
directly - 6.7% real-test accuracy, 10.2% on its own synthetic
(in-distribution) test set (models/eval_report.json, models/clf_ablation.json).
It has since been removed; this module is now the only incident-type
classifier in the scan path. No amount of prompting fixes a training bug if
there is one, and this module does not attempt to - see
tools/eval_gemini_classify.py's own docstring for that half.

Model: gemma4:e4b (Gemma Team, 2026 - arXiv:2607.02770), a general-purpose
model with no Southeast-Asian-language tuning, not a SEA-LION variant despite
the module name. The original hypothesis here was the opposite of that choice:
a model tuned for Tagalog/Cebuano (aisingapore/Gemma-SEA-LION-v3-9B-IT,
q4_k_m) should read Taglish barangay narratives better than a general-purpose
model. Measuring both against the same 151-record dev split falsified that
hypothesis for this task - gemma4:e4b reached 67.5% type accuracy versus
SEA-LION v3-9B-IT's 55.6% (also the best-performing SEA-LION checkpoint on
this dataset), and versus every other SEA-LION checkpoint tried
(llama-sea-lion-v3.5-8b-r: 51.7%; Gemma-SEA-LION-v4-4B-VL: 47.0%;
Gemma-SEA-LION-v4.5-E2B-IT: 41.7%) (models/metrics_preds_*.csv, scored via
tools/eval_classifier_metrics.py + tools/score_classifier_metrics.py). Kept
the module name on the smallest-diff principle - it is still the module that
owns few-shot incident classification, just no longer via a SEA-LION
checkpoint.

Was bantay/ml/gemini_classify.py. Swapped to a local Ollama model instead of
the Gemini API: no per-call billing, no data leaving the machine. The
prompt, label set and defensive parser are unchanged from the Gemini version
- only the transport (a plain HTTP call to Ollama's own REST API instead of
the google-genai SDK) is different.

Off by default: BANTAY_OLLAMA must be set to "1" before this dials out, same
"the Flask app runs with no key, no SDK and no network by default" contract
the OCR Gemini pass uses. Ollama has no API key to auto-detect the way a
Gemini credential does, so there is nothing to detect - hence an explicit
opt-in flag instead of a "did the user configure this" probe.
"""
import json
import os

from ..normalize import CANONICAL_CATEGORIES, FALLBACK_CATEGORY, get_category_group

# See gemini_classify.py's comment (this module's predecessor) for why the
# legacy-only 3 categories are excluded and FALLBACK_CATEGORY is added back.
_LEGACY_ONLY = {"Work/Service Dispute", "Curfew Violation", "Amicable Settlement"}
TYPE_NAMES = [name for name, _ in CANONICAL_CATEGORIES if name not in _LEGACY_ONLY] + [FALLBACK_CATEGORY]

# See the module docstring for the accuracy comparison behind this choice.
# Verify with `ollama list` after `ollama pull` - a wrong tag fails per-scan,
# not at startup, same caveat as the Gemini module's model ID.
_DEFAULT_MODEL = "gemma4:e4b"
_DEFAULT_HOST = "http://localhost:11434"

# The five checkpoints the study actually measured, with their dev-split scores
# (n=151; see tools/eval_classifier_metrics.py and models/metrics_preds_*.csv).
# The UI offers exactly these and classify() accepts nothing else: a free-text
# model field would post an arbitrary string to Ollama and, worse, would let a
# record be classified by something with no accuracy figure behind it. Tags are
# spelled as `ollama list` reports them - Ollama preserves the pulled case.
# Listed in descending type_acc order, which the UI dropdown preserves.
EVALUATED_MODELS = [
    {"tag": "gemma4:e4b", "label": "Gemma 4 (deployed)",
     "type_acc": 0.675, "group_acc": 0.841},
    {"tag": "aisingapore/Gemma-SEA-LION-v3-9B-IT:q4_k_m", "label": "Gemma SEA-LION v3 9B-IT (q4_k_m)",
     "type_acc": 0.556, "group_acc": 0.762},
    {"tag": "aisingapore/llama-sea-lion-v3.5-8b-r", "label": "Llama SEA-LION v3.5 8B-R",
     "type_acc": 0.517, "group_acc": 0.735},
    {"tag": "aisingapore/Gemma-SEA-LION-v4-4B-VL", "label": "Gemma SEA-LION v4 4B-VL",
     "type_acc": 0.470, "group_acc": 0.656},
    {"tag": "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT", "label": "Gemma SEA-LION v4.5 E2B-IT",
     "type_acc": 0.417, "group_acc": 0.656},
]
_ALLOWED_TAGS = {m["tag"] for m in EVALUATED_MODELS}


def load_model_comparison(model_dir):
    """Recent classifier comparison for the dashboard/analytics "Model
    Comparison Results" table (Chapter 4, §4.3.2). This project trains
    nothing - EVALUATED_MODELS above already ships the two headline numbers
    used at classification time - so this reads the full metric sweep instead
    (models/classifier_metrics.json, written by tools/score_classifier_metrics.py),
    since the analytics table also shows precision/recall/AUC/eval loss.
    Returns None if that file is missing or unreadable (e.g. a fresh checkout
    before the sweep has been run), same fallback shape the old
    comparison_results.json loader used.
    """
    path = os.path.join(model_dir, "classifier_metrics.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    results = []
    for m in data.get("models", []):
        t, g = m["type"], m["group"]
        results.append({
            "model": m["label"], "accuracy": t["accuracy"],
            "precision": t["precision_weighted"], "recall": t["recall_weighted"],
            "f1": t["f1_weighted"], "f1_macro": t["f1_macro"],
            "auc": t["auc_macro"], "eval_loss": t["eval_loss"],
            "group_accuracy": g["accuracy"],
            "latency_median_s": m.get("latency_median_s"),
        })
    if not results:
        return None
    best = max(results, key=lambda r: r["accuracy"])
    return {"results": results, "best_model": best["model"],
            "n": data.get("n"), "split": data.get("split")}


def resolve_model(tag):
    """The model a request should use. An unknown or empty tag falls back to the
    configured default rather than raising: a stale dropdown value should still
    classify the record, just not with whatever string arrived."""
    if tag and tag in _ALLOWED_TAGS:
        return tag
    return os.environ.get("BANTAY_OLLAMA_MODEL", _DEFAULT_MODEL)

# Measured at ~18.5s/record average over a 151-record dev-split run on this
# project's hardware (gemma4:e4b is larger than the SEA-LION checkpoint this
# module used previously). Ollama unloads an idle model by default (5 min with
# no requests), and a cold reload from disk to VRAM costs more than that -
# 120s covers a cold load plus generation with room to spare. Raise via
# BANTAY_OLLAMA_TIMEOUT if your disk or GPU is slower.
DEFAULT_TIMEOUT = 120.0

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

# Ollama's structured-output contract: passing a JSON schema (rather than the
# bare string "json") constrains generation to this shape, the same guarantee
# response_mime_type="application/json" gave on the Gemini side.
_SCHEMA = {
    "type": "object",
    "properties": {
        "incident_type": {"type": "string"},
        "confidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["incident_type"],
}


def available():
    """True if a classification call would actually reach a backend."""
    return os.environ.get("BANTAY_OLLAMA", "0") == "1"


def _ask(narrative, model, host, timeout):
    import requests

    prompt = _PROMPT.format(n=len(TYPE_NAMES),
                            types="\n".join(f"- {t}" for t in TYPE_NAMES),
                            text=narrative)
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": _SCHEMA,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _parse(resp.json().get("message", {}).get("content"))


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


def classify(narrative, model=None, host=None, timeout=DEFAULT_TIMEOUT):
    """One call. Returns {"incident_type", "category_group", "confidence_label",
    "reason", "status"} and never raises - same contract as gemini_classify's:
    a dead server, dead network or malformed reply degrades to no prediction
    (incident_type=None), and routes/scan.py surfaces the reason rather than
    passing off another model's answer as this one's."""
    if not available():
        return {"incident_type": None, "category_group": None, "status": "off"}
    if not (narrative or "").strip():
        return {"incident_type": None, "category_group": None, "status": "skipped: empty narrative"}

    model = model or os.environ.get("BANTAY_OLLAMA_MODEL", _DEFAULT_MODEL)
    host = host or os.environ.get("BANTAY_OLLAMA_HOST", _DEFAULT_HOST)
    timeout = float(os.environ.get("BANTAY_OLLAMA_TIMEOUT", timeout))
    try:
        parsed = _ask(narrative, model, host, timeout)
    except Exception as exc:                      # noqa: BLE001 - any HTTP/network failure
        return {"incident_type": None, "category_group": None,
                "status": f"error: {type(exc).__name__}: {exc}"[:200]}

    if not parsed:
        return {"incident_type": None, "category_group": None,
                "status": "ok: no usable answer"}

    return {"incident_type": parsed["incident_type"],
            "category_group": get_category_group(parsed["incident_type"]),
            "confidence_label": parsed["confidence_label"],
            "reason": parsed["reason"],
            "status": f"ok ({model} via ollama@{host})"}


# Second model consulted purely to check the primary's answer - agreement
# between two independently-reasoning models is a validated confidence
# signal, the same trick ocr/reconcile.py already uses between the regex and
# Gemini field readers (see that module's docstring). The primary model's own
# self-reported "confidence" field is not validated: Chapter 4, §4.3.2
# measured Gemma 4's stated certainty saturated at ~100% on every one of 151
# dev-split records, including the 49 it got wrong - a signal with no spread
# cannot separate anything. Picked v3-9B-IT as the check partner because it
# gave the widest measured split against Gemma 4 specifically on that same
# dev split: 85.2% accurate when the two agree (n=81) vs 47.1% when they
# disagree (n=70) - a 38-point gap, against ~40-point-or-less gaps for the
# other three candidates. A group-level "same Category Group but different
# type" middle tier was also measured and rejected: its accuracy (48.5%,
# n=33) does not separate from outright disagreement (45.9%, n=37), so a
# 3-way high/medium/low split would be manufactured granularity the data
# does not support - this is a 2-way signal (agree/disagree) by design.
CHECK_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B-IT:q4_k_m"


def classify_with_agreement(narrative, model=None, check_model=None,
                            host=None, timeout=DEFAULT_TIMEOUT):
    """classify(), plus a second, independent model call whose only job is to
    agree or disagree with the first. confidence_label comes from that
    agreement ("high" if they match, "low" if they don't) and never from
    either model's own stated confidence, which is discarded - see
    CHECK_MODEL's comment for why the self-reported figure is not trusted.

    check_model defaults to CHECK_MODEL, unless the primary IS CHECK_MODEL
    (the record form lets an encoder pick any of the five candidates), in
    which case Gemma 4 stands in as the check partner instead - checking a
    model against itself would trivially always "agree" and manufacture a
    false "high".

    ponytail: sequential, not concurrent. This project's GPU has 6GB VRAM;
    each candidate alone peaks at ~5.4-5.8GB (Chapter 4, §4.3.2), so two
    cannot stay resident together - a "concurrent" pair of calls would just
    make Ollama swap models mid-flight rather than save wall time. Upgrade
    path: run the two calls concurrently once deployed on hardware with room
    for both models loaded at once.

    Falls back to "medium" (unverified, not a free pass to "high") if the
    check call itself returns no answer - Ollama off, unreachable, or a
    malformed reply. The primary's own answer is still returned and its
    self-reported confidence is still discarded either way; there is just no
    second opinion to grade it against this time.
    """
    # host/timeout are only forwarded when the caller actually set them -
    # classify()'s own defaults (env-resolved host, DEFAULT_TIMEOUT) already
    # match classify_with_agreement's, and every existing caller (and test
    # double) invokes classify() bare as classify(text, model=...), so
    # forcing these kwargs through unconditionally would require every one
    # of them to also accept host/timeout for no behavioural difference.
    kwargs = {}
    if host is not None:
        kwargs["host"] = host
    if timeout != DEFAULT_TIMEOUT:
        kwargs["timeout"] = timeout

    primary = classify(narrative, model=model, **kwargs)
    if not primary.get("incident_type"):
        return primary

    resolved_model = model or os.environ.get("BANTAY_OLLAMA_MODEL", _DEFAULT_MODEL)
    check_model = check_model or (CHECK_MODEL if resolved_model != CHECK_MODEL else _DEFAULT_MODEL)

    check = classify(narrative, model=check_model, **kwargs)
    check_type = check.get("incident_type")

    primary["check_model"] = check_model
    primary["check_prediction"] = check_type
    if not check_type:
        primary["confidence_label"] = "medium"
        primary["agree"] = None
        return primary

    agree = primary["incident_type"] == check_type
    primary["confidence_label"] = "high" if agree else "low"
    primary["agree"] = agree
    if not agree:
        label = check_model.rsplit("/", 1)[-1]
        base = primary.get("reason", "")
        primary["reason"] = f'{base} — {label} predicted "{check_type}" instead'.strip(" —")
    return primary


if __name__ == "__main__":       # python -m bantay.ml.sealion_classify
    on = available()
    print(f"backend : {'on' if on else 'off (set BANTAY_OLLAMA=1)'}")
    if on:
        sample = ("NAGTUNGO DITO SA BARANGAY ITONG SI [PERSON_1] UPANG IREKLAMO ANG "
                  "PAGKUHA NG MANOK NA PANABONG SA GABI. NAKAWALA RIN ANG MANOK NG "
                  "KATABING BAHAY KAY [PERSON_2].")
        out = classify(sample)
        print("in     :", sample)
        print("type   :", out["incident_type"])
        print("group  :", out["category_group"])
        print("status :", out["status"])
