"""Text-only OCR-correction pass via a local Ollama SEA-LION model - the
Ollama counterpart to gemini.py's restore() text arm, for side-by-side
comparison against Gemini on the exact same Vision transcriptions.

Same opt-in convention as ml/sealion_classify.py: off unless BANTAY_OLLAMA=1,
BANTAY_OLLAMA_MODEL/_HOST override the model/host. Reuses gemini.py's prompt,
guards and diff/splice logic unchanged - only the call itself (a plain HTTP
POST to Ollama instead of the google-genai SDK) differs, so a model swap does
not also quietly change what counts as a safe edit.

Text-only by design, matching the user's ask: there is no restore_from_image()
counterpart here. Same ceiling as gemini.restore()'s text arm - it can only
repair a token Vision already emitted (see gemini.py's module docstring).
"""
import json
import os

from .gemini import (
    _FIELD_KEYS, _LEXICON_BLOCK, _RESTORE_PROMPT,
    _changed_chars, _diff_edits, _keep_restore, _result,
)


# Three candidates tried on this hardware before landing here:
#   llama-sea-lion-v3.5-8b-r (the "-R" reasoning model ml/sealion_classify.py
#     uses)         even with `think` disabled, spiralled in its own reasoning
#                   channel on this task - 1 of 3 identical calls converged
#                   (~105s for one short line), the other 2 ran 25+ minutes or
#                   hit a 2048-token cap still "thinking", no answer.
#   Gemma-SEA-LION-v4.5-E2B-IT (2B)   fast (5.8s) and never got stuck, but on
#                   3 real logbook pages proposed 0 edits on all 3 - too
#                   conservative to be a useful corrector.
#   Gemma-SEA-LION-v3-9B-IT, full F16 (18GB)   correction quality was good on
#                   the smoke sample (fixed NAG5ADYA -> NAGSADYA exactly), but
#                   18GB does not fit this machine's VRAM, so it runs mostly
#                   on CPU: 209s for one short line, and both real pages timed
#                   out at 300s.
# q4_k_m (5.8GB, same quant tier ml/sealion_classify.py's default already
# uses) is the practical middle: fits the GPU, actually proposes real edits
# (18/12/10 on 3 real pages, where the 2B model proposed none), and finishes
# a real page in 65-107s - slow next to Gemini's 3-11s, but bounded, not
# stuck. If a smaller/faster quant proposes real edits too, that is worth
# trying; q2_k through q5_k_m are the other tags this model ships.
_DEFAULT_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B-IT:q4_k_m"
_DEFAULT_HOST = "http://localhost:11434"

# Measured 65-107s per real logbook page (text + 8-field extraction) on this
# hardware - slower than the 5.8s smoke-sample figure because a real page is
# far longer input and output. Headroom above the observed worst case; raise
# via BANTAY_OLLAMA_RESTORE_TIMEOUT if a page still times out.
DEFAULT_TIMEOUT = 180.0

# think=False plus a mild repeat penalty: belt-and-suspenders left in from
# debugging the -R model's reasoning loops (see _DEFAULT_MODEL above). Gemma-
# SEA-LION did not need them to converge, but they cost nothing on a model
# that behaves, and cap the worst case if a future default model does not.
_OPTIONS = {"temperature": 0.2, "repeat_penalty": 1.3, "repeat_last_n": 64, "num_predict": 2048}

MAX_RESTORE_CHANGE = float(os.environ.get("BANTAY_RESTORE_MAX_CHANGE", "0.35"))

_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_text": {"type": "string"},
        "fields": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in _FIELD_KEYS},
        },
    },
    "required": ["corrected_text"],
}


def available():
    """True if a restore() call would actually reach a backend."""
    return os.environ.get("BANTAY_OLLAMA", "0") == "1"


def _ask(text, lexicon, model, host, timeout):
    import requests

    words = ""
    if lexicon:
        top = sorted(lexicon, key=lexicon.get, reverse=True)[:300]
        words = _LEXICON_BLOCK.format(words=", ".join(top))

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _RESTORE_PROMPT.format(lexicon=words, text=text)}],
            "format": _SCHEMA,
            "stream": False,
            "think": False,
            "options": _OPTIONS,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _parse(resp.json().get("message", {}).get("content"))


def _parse(raw):
    """Defensive parse: a malformed reply is (None, {}), never an exception."""
    if not raw:
        return None, {}
    try:
        data = json.loads(raw)
    except ValueError:
        return None, {}
    if not isinstance(data, dict):
        return None, {}
    corrected = data.get("corrected_text")
    corrected = corrected if isinstance(corrected, str) and corrected.strip() else None
    fields = {}
    raw_fields = data.get("fields")
    if isinstance(raw_fields, dict):
        for key in _FIELD_KEYS:
            value = raw_fields.get(key)
            if isinstance(value, str):
                fields[key] = value.strip()
    return corrected, fields


def restore(text, lexicon=None, model=None, host=None, timeout=None, max_change=None):
    """Repair a Vision transcription via SEA-LION. Same contract, guards and
    splice-back-into-the-original behaviour as gemini.restore() - see that
    docstring.

    Returns {"text", "edits", "fields", "fidelity", "status"}, never raises.
    The field grounding and the loss/gain report come from gemini._result, so
    both arms are gated identically - a comparison where only one arm's fields
    are checked measures the guard, not the model.
    """
    if not available():
        return _result(text, text, [], {}, "off")
    if not text or not text.strip():
        return _result(text, text, [], {}, "skipped: empty page")

    model = model or os.environ.get("BANTAY_OLLAMA_MODEL", _DEFAULT_MODEL)
    host = host or os.environ.get("BANTAY_OLLAMA_HOST", _DEFAULT_HOST)
    timeout = float(os.environ.get("BANTAY_OLLAMA_RESTORE_TIMEOUT", timeout or DEFAULT_TIMEOUT))
    max_change = MAX_RESTORE_CHANGE if max_change is None else max_change
    try:
        corrected, fields = _ask(text, lexicon, model, host, timeout)
    except Exception as exc:                      # noqa: BLE001 - any HTTP/network failure
        return _result(text, text, [], {},
                       f"error: {type(exc).__name__}: {exc}"[:200])

    if corrected is None:
        return _result(text, text, [], fields,
                       "ok: no corrected text returned, fields only")

    proposed = _diff_edits(text, corrected)
    for e in proposed:
        e["source"] = "sealion"                   # gemini._diff_edits hardcodes "gemini"
    kept = [e for e in proposed if _keep_restore(e)]
    dropped = len(proposed) - len(kept)

    out, cursor = [], 0
    for e in sorted(kept, key=lambda e: e["start"]):
        if e["start"] < cursor:                   # overlapping spans - keep the first
            continue
        out.append(text[cursor:e["start"]])
        out.append(e["after"])
        cursor = e["end"]
    out.append(text[cursor:])
    new_text = "".join(out)

    touched = sum(_changed_chars(e["before"], e["after"]) for e in kept) / max(len(text), 1)
    if touched > max_change:
        return _result(text, text, [], fields,
                       f"rejected: model rewrote {touched:.0%} of the page "
                       f"(limit {max_change:.0%}) - raw OCR kept")

    note = f" ({dropped} rejected by guards)" if dropped else ""
    return _result(text, new_text, kept, fields,
                   f"ok: {len(kept)} edits{note} ({model} via ollama@{host})")


if __name__ == "__main__":       # python -m bantay.ocr.sealion [--smoke]
    import sys

    on = available()
    print(f"backend : {'on' if on else 'off (set BANTAY_OLLAMA=1)'}")
    print(f"model   : {os.environ.get('BANTAY_OLLAMA_MODEL', _DEFAULT_MODEL)}")
    print(f"host    : {os.environ.get('BANTAY_OLLAMA_HOST', _DEFAULT_HOST)}")

    if on and "--smoke" in sys.argv:
        sample = ("NAG5ADYA DIT0 SA BRGY. HALL NG ANUNAS UPANG IPA-BL0TTER ANG "
                  "NANGYARING PANL0LOOB SA KANILANG BAHAY N00NG JAN. 3, 2026")
        print()
        print("smoke test - one Ollama call")
        print("  in     :", sample)
        out = restore(sample)
        print("  out    :", out["text"])
        print("  status :", out["status"])
        for e in out["edits"]:
            print(f"    {e['before']!r} -> {e['after']!r}")
        if "JAN. 3, 2026" not in out["text"]:
            print("  WARNING: the date changed. That must not happen - stop and check _keep_restore().")
