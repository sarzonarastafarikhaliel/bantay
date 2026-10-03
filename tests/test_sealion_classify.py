import json

from bantay.ml.sealion_classify import TYPE_NAMES, _parse


def test_parses_a_valid_reply():
    raw = json.dumps({"incident_type": "Theft", "confidence": "high", "reason": "manok stolen"})
    out = _parse(raw)
    assert out["incident_type"] == "Theft"
    assert out["confidence_label"] == "high"


def test_rejects_an_off_list_label():
    """An incident_type outside TYPE_NAMES would corrupt every downstream
    lookup (get_category_group, PNP tier) that assumes a canonical value -
    this must come back as no answer, never a fabricated 36th type."""
    raw = json.dumps({"incident_type": "Something SEA-LION Invented"})
    assert _parse(raw) is None


def test_malformed_reply_is_no_answer_never_an_exception():
    assert _parse("not json") is None
    assert _parse("") is None
    assert _parse(json.dumps(["a", "list", "not", "a", "dict"])) is None
    assert _parse(json.dumps({"wrong_key": "Theft"})) is None


def test_type_names_is_the_canonical_taxonomy_minus_legacy_only_labels():
    """The prompt's label set must stay the one every downstream lookup
    (get_category_group, PNP tier) is defined over, or an accepted answer
    corrupts the record it lands on. Pinned against normalize.py directly now
    that the trained head whose labels.json used to be the reference is gone."""
    from bantay.normalize import CANONICAL_CATEGORIES, FALLBACK_CATEGORY

    canonical = {name for name, _ in CANONICAL_CATEGORIES} | {FALLBACK_CATEGORY}
    assert set(TYPE_NAMES) <= canonical
    assert FALLBACK_CATEGORY in TYPE_NAMES


def test_resolve_model_accepts_only_evaluated_checkpoints():
    """The model arrives from a UI dropdown, so it is untrusted input that ends
    up in an Ollama request body. Anything off the evaluated list falls back to
    the default: a record must never be classified by a checkpoint with no
    measured accuracy behind it, and an arbitrary string must never be
    forwarded to the backend.
    """
    from bantay.ml.sealion_classify import (EVALUATED_MODELS, _DEFAULT_MODEL,
                                            resolve_model)

    for entry in EVALUATED_MODELS:
        assert resolve_model(entry["tag"]) == entry["tag"]

    for junk in ("", None, "gpt-4", "gemma4:e4b; rm -rf /", "../../etc/passwd"):
        assert resolve_model(junk) == _DEFAULT_MODEL


def test_evaluated_models_carry_the_accuracy_the_ui_advertises():
    """The dropdown renders type_acc/group_acc next to each name, so a missing
    or out-of-range figure would put a wrong number in front of an encoder."""
    from bantay.ml.sealion_classify import EVALUATED_MODELS, _DEFAULT_MODEL

    assert _DEFAULT_MODEL in {m["tag"] for m in EVALUATED_MODELS}
    for m in EVALUATED_MODELS:
        assert m["label"]
        assert 0.0 < m["type_acc"] <= 1.0
        assert 0.0 < m["group_acc"] <= 1.0
