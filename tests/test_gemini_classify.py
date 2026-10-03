import json

from bantay.ml.gemini_classify import TYPE_NAMES, _parse


def test_parses_a_valid_reply():
    raw = json.dumps({"incident_type": "Theft", "confidence": "high", "reason": "manok stolen"})
    out = _parse(raw)
    assert out["incident_type"] == "Theft"
    assert out["confidence_label"] == "high"


def test_rejects_an_off_list_label():
    """An incident_type outside TYPE_NAMES would corrupt every downstream
    lookup (get_category_group, PNP tier) that assumes a canonical value -
    this must come back as no answer, never a fabricated 36th type."""
    raw = json.dumps({"incident_type": "Something Gemini Invented"})
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
