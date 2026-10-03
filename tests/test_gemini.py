"""Guards on the LLM verification pass.

Every test here stubs `_ask`, so nothing needs a key, a network or the SDK. What
is being tested is not Gemini - it is the wall between Gemini and the record:
the module must apply a genuine typo fix and refuse everything else, including
its own failures. If these pass, the worst a bad model reply can do is nothing.
"""
import pytest

from bantay.ocr import gemini

PAGE = ("NAG5ADYA DITO SA BRGY. HALL NG ANUNAS UPANG IPA-BLOTTER ANG NANGYARING "
        "PANLOLOOB SA KANILANG BAHAY NOONG 2023")


_ENV = ("GEMINI_API_KEY", "BANTAY_GEMINI", "BANTAY_GEMINI_MODEL",
        "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from 'nothing configured'.

    Autouse because these tests run on the same machine that has the real Vertex
    variables exported - without this, 'no backend configured' quietly becomes
    'my project's backend' and the off-switch tests pass for the wrong reason.
    """
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")


@pytest.fixture
def vertexed(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "bantay-test-project")


def _reply(monkeypatch, edits):
    """Stub the one function that talks to Google. (before, after, reason)."""
    monkeypatch.setattr(gemini, "_ask", lambda *a, **kw: list(edits))


def test_nothing_configured_is_a_no_op():
    """The app must run on a machine with no key, no project and no network."""
    out = gemini.verify(PAGE)
    assert out["text"] == PAGE and out["edits"] == [] and out["status"] == "off"
    assert gemini.available() is False


def test_kill_switch_beats_both_backends(keyed, vertexed, monkeypatch):
    monkeypatch.setenv("BANTAY_GEMINI", "0")
    assert gemini._mode() == ""
    assert gemini.verify(PAGE)["status"] == "off"


def test_vertex_is_recognised_and_preferred_over_a_stray_key(keyed, vertexed):
    """Vertex bills the GCP project, which is the point of choosing it - so a
    leftover AI Studio key must not quietly redirect the spend."""
    assert gemini._mode() == "vertex"


def test_api_key_alone_selects_the_api_key_backend(keyed):
    assert gemini._mode() == "api-key"


def test_vertex_without_a_project_is_not_configured(monkeypatch):
    """Half-set env is off, not a crash at scan time."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert gemini._mode() == "" and gemini.verify(PAGE)["status"] == "off"


def test_status_names_the_backend_and_model(vertexed, monkeypatch):
    """The scan page shows this string; 'which endpoint did this text touch' is
    a provenance question, not a debugging nicety."""
    monkeypatch.setenv("BANTAY_GEMINI_MODEL", "gemini-x-flash")
    _reply(monkeypatch, [("NAG5ADYA", "NAGSADYA", "5 read for S")])
    assert gemini.verify(PAGE)["status"] == "ok: 1 edits (gemini-x-flash via vertex)"


def test_applies_a_real_typo_fix(keyed, monkeypatch):
    _reply(monkeypatch, [("NAG5ADYA", "NAGSADYA", "5 read for S")])
    out = gemini.verify(PAGE)
    assert out["text"].startswith("NAGSADYA DITO")
    assert len(out["edits"]) == 1
    edit = out["edits"][0]
    assert edit["before"] == "NAG5ADYA" and edit["after"] == "NAGSADYA"
    assert edit["source"] == "gemini", "the encoder must be able to tell where an edit came from"
    assert PAGE[edit["start"]:edit["end"]] == "NAG5ADYA", "offsets must point at the token"


def test_rejects_a_rewrite_dressed_as_a_correction(keyed, monkeypatch):
    """The failure that would end the thesis: the model paraphrasing the record."""
    _reply(monkeypatch, [
        ("PANLOLOOB", "ROBBERY WITH VIOLENCE", "translated"),   # multi-token
        ("NANGYARING", "ARSON", "unrelated word"),              # similarity too low
    ])
    out = gemini.verify(PAGE)
    assert out["text"] == PAGE
    assert out["edits"] == []
    assert "none passed the guards" in out["status"]


def test_never_touches_numbers(keyed, monkeypatch):
    """Dates, times and amounts are legal facts, not spelling."""
    _reply(monkeypatch, [("2023", "2028", "looks like an 8")])
    out = gemini.verify(PAGE)
    assert "2023" in out["text"] and "2028" not in out["text"]


def test_rejects_an_edit_for_a_word_not_on_the_page(keyed, monkeypatch):
    _reply(monkeypatch, [("SAKSAK", "SAKSAKAN", "hallucinated token")])
    assert gemini.verify(PAGE)["text"] == PAGE


def test_drops_everything_when_the_page_is_rewritten_wholesale(keyed, monkeypatch):
    """Individually plausible edits, together a rewrite. Half a rewrite is not
    safer than a whole one, so the accept is all-or-nothing.
    """
    text = "NAGSADYA PANLOLOOB NANGYARING"
    _reply(monkeypatch, [("NAGSADYA", "NAGSAKSAK", ""),
                         ("PANLOLOOB", "PANGUNAHIN", ""),
                         ("NANGYARING", "NANGGALING", "")])
    out = gemini.verify(text)
    assert out["text"] == text and out["edits"] == []
    assert out["status"].startswith("rejected:")


def test_many_small_repairs_are_not_a_rewrite(keyed, monkeypatch):
    """The regression the live smoke test caught: the cap used to count the whole
    length of every edited token, so a badly garbled page - the case this layer
    exists for - scored as a paraphrase and lost every correct fix.
    """
    text = ("NAG5ADYA DIT0 SA BRGY. HALL NG ANUNAS UPANG IPA-BL0TTER ANG "
            "NANGYARING PANL0LOOB SA KANILANG BAHAY N00NG JAN. 3, 2026")
    _reply(monkeypatch, [("NAG5ADYA", "NAGSADYA", "5/S"), ("DIT0", "DITO", "0/O"),
                         ("IPA-BL0TTER", "IPA-BLOTTER", "0/O"),
                         ("PANL0LOOB", "PANLOLOOB", "0/O"), ("N00NG", "NOONG", "0/O")])
    out = gemini.verify(text)
    assert len(out["edits"]) == 5, out["status"]
    assert "NAGSADYA DITO" in out["text"] and "IPA-BLOTTER" in out["text"]
    assert "JAN. 3, 2026" in out["text"], "the date must survive a busy page"


def test_api_failure_degrades_the_scan_but_does_not_break_it(keyed, monkeypatch):
    def boom(*a, **kw):
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(gemini, "_ask", boom)
    out = gemini.verify(PAGE)
    assert out["text"] == PAGE and out["edits"] == []
    assert out["status"].startswith("error:"), "a silent no-op would hide a dead key for weeks"


@pytest.mark.parametrize("raw", ["", "not json at all", "{}", '{"edits": "nope"}',
                                 '{"edits": [{"before": 1, "after": 2}]}'])
def test_malformed_replies_parse_to_nothing(raw):
    assert gemini._parse(raw) == []


def test_parse_reads_a_well_formed_reply():
    got = gemini._parse('{"edits": [{"before": "NAG5ADYA", "after": "NAGSADYA", "reason": "5/S"}]}')
    assert got == [("NAG5ADYA", "NAGSADYA", "5/S")]
