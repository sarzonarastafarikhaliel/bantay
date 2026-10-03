"""The record template and the Gemini restore pass.

Every test here stubs the network, so nothing needs a key, a project or the SDK.
What is under test is not Gemini - it is the wall between Gemini and the record.
The restore pass is allowed to return whole text, which is exactly why the diff,
the numeric guard and the rewrite ceiling have to hold: if these pass, the worst
a bad model reply can do is leave the Vision output alone.
"""
import json

import pytest

from bantay import narrative
from bantay.ocr import gemini

PAGE = "NAG5ADY4 SI JU4N SA BRGY H4LL NOONG 2023 ALAS 3:00"

_ENV = ("GEMINI_API_KEY", "BANTAY_GEMINI", "BANTAY_GEMINI_MODEL",
        "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from 'nothing configured'. Autouse for the same reason
    test_gemini.py does it: this machine has the real Vertex variables exported,
    and without this the off-switch tests would pass for the wrong reason."""
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")


def _reply(monkeypatch, corrected, fields=None):
    """Stub the one function that talks to Google."""
    monkeypatch.setattr(gemini, "_ask_restore", lambda *a, **kw: (corrected, fields or {}))


# --- the template -----------------------------------------------------------

def test_render_marks_slots_the_page_never_stated():
    out = narrative.render({"incident_summary": "NAWALA ANG ALAGANG ASO"})
    assert "NAWALA ANG ALAGANG ASO" in out
    # A missing fact must LOOK missing - an encoder can fill a blank they see.
    assert out.count(narrative.BLANK) == len(narrative.SLOTS) - 1


def test_render_is_stable_for_the_same_fields():
    fields = {"date": "2026-01-03", "incident_summary": "X"}
    assert narrative.render(fields) == narrative.render(dict(fields))


def test_normalize_fields_matches_the_csv_import_formats():
    out = narrative.normalize_fields(
        {"date": "JAN. 3, 2026", "time": "2:30 PM", "location_purok": "Purok 5 Anunas"})
    assert out["date"] == "2026-01-03"
    assert out["time"] == "14:30"
    assert out["location_purok"] == "Purok 5"


def test_unparseable_date_becomes_blank_not_a_guess():
    assert narrative.normalize_fields({"date": "wala"})["date"] == ""


def test_backfill_never_overwrites_what_the_model_read():
    merged = narrative.backfill({"date": "2026-01-03", "time": ""},
                                {"date": "1999-01-01", "time": "09:00"})
    assert merged["date"] == "2026-01-03"   # model value wins
    assert merged["time"] == "09:00"        # regex fills only the gap


def test_backfill_prefers_the_closed_set_reader_for_status_and_action():
    """Regression: measured via tools/run_gemini_arms.py, Gemini's own status/
    action_taken text ("SETTLEMENT", a copied narrative) loses to the regex
    classifier's canonical phrasing under exact-match gold scoring - so unlike
    every other slot, these two should NOT defer to a non-empty model value."""
    merged = narrative.backfill(
        {"status": "SETTLEMENT", "action_taken": "long copied narrative..."},
        {"status": "Settled", "action_taken": "Settlement / confrontation held at barangay"})
    assert merged["status"] == "Settled"
    assert merged["action_taken"] == "Settlement / confrontation held at barangay"

    # Still only fills a real value - a blank regex reading must not erase the
    # model's answer, since a model answer beats no answer at all.
    merged = narrative.backfill({"status": "Pending"}, {"status": ""})
    assert merged["status"] == "Pending"


def test_missing_reports_required_slots():
    assert set(narrative.missing({"date": "2026-01-03"})) == {"location_purok",
                                                              "incident_summary"}
    assert narrative.missing({"date": "d", "location_purok": "p",
                              "incident_summary": "s"}) == []


# --- the restore pass -------------------------------------------------------

def test_repairs_a_genuine_misread(monkeypatch, keyed):
    _reply(monkeypatch, "NAGSADYA SI JUAN SA BRGY HALL NOONG 2023 ALAS 3:00")
    out = gemini.restore(PAGE)
    assert "NAGSADYA" in out["text"] and "JUAN" in out["text"]
    assert all(e["source"] == "gemini" for e in out["edits"])


def test_never_changes_a_year(monkeypatch, keyed):
    """The one failure that would actually surface in a hearing."""
    _reply(monkeypatch, "NAGSADYA SI JUAN SA BRGY HALL NOONG 2028 ALAS 3:00")
    out = gemini.restore(PAGE)
    assert "2023" in out["text"] and "2028" not in out["text"]


def test_never_changes_a_time(monkeypatch, keyed):
    _reply(monkeypatch, "NAG5ADY4 SI JU4N SA BRGY H4LL NOONG 2023 ALAS 9:00")
    assert "3:00" in gemini.restore(PAGE)["text"]


def test_rejects_a_rewrite_whole(monkeypatch, keyed):
    """A partial accept would keep whichever half of a rewrite sorted first."""
    _reply(monkeypatch, "THE COMPLAINANT ALLEGED THEFT OF PROPERTY AT THE BARANGAY HALL",
           {"status": "Pending"})
    out = gemini.restore(PAGE)
    assert out["text"] == PAGE
    assert out["edits"] == []
    assert out["status"].startswith("rejected")
    # Fields still come back: they land in form inputs the encoder reviews.
    assert out["fields"]["status"] == "Pending"


def test_will_not_delete_a_word_it_could_not_read(monkeypatch, keyed):
    _reply(monkeypatch, "SI JU4N SA BRGY H4LL NOONG 2023 ALAS 3:00")
    assert "NAG5ADY4" in gemini.restore(PAGE)["text"]


# --- the field guard, and the minimally-lossy budgets ------------------------
# These three failures all reached the record through `fields`, which until now
# no guard touched. See gemini._ground_fields.

def test_a_paraphrased_summary_never_reaches_the_record(monkeypatch, keyed):
    """The failure from models/preds_gemini_text.csv, Blotter_Pic (10).

    The model returned a fluent Tagalog paraphrase, the text guard rejected it
    at 69% and kept the raw OCR - and the paraphrase was filed as the record's
    narrative anyway, because fields bypassed every guard in the module.
    """
    _reply(monkeypatch, PAGE, {"incident_summary":
                               "THE COMPLAINANT REPORTED A THEFT OF POULTRY AT NIGHT"})
    out = gemini.restore(PAGE)
    assert out["fields"]["incident_summary"] == ""
    assert "dropped ungrounded incident_summary" in out["status"]


def test_a_summary_copied_off_the_page_survives_the_guard(monkeypatch, keyed):
    """The guard must not cost the pipeline its normal, correct answer.

    Grounding runs against the GATED text, not Vision's raw output, which is
    what lets a summary quote the repaired spelling: the page reads JU4N, the
    accepted repair makes it JUAN, and the summary saying JUAN is then supported.
    """
    _reply(monkeypatch, "NAGSADYA SI JUAN SA BRGY HALL NOONG 2023 ALAS 3:00",
           {"incident_summary": "NAGSADYA SI JUAN SA BRGY HALL"})
    out = gemini.restore(PAGE)
    assert out["fields"]["incident_summary"] == "NAGSADYA SI JUAN SA BRGY HALL"
    assert "dropped ungrounded" not in out["status"]


def test_a_hallucinated_respondent_is_dropped(monkeypatch, keyed):
    """A name nobody on the page is accused by is the worst slot to invent."""
    _reply(monkeypatch, PAGE, {"respondent": "PEDRO SANTOS", "date": "NOONG 2023"})
    out = gemini.restore(PAGE)
    assert out["fields"]["respondent"] == ""
    # date is reformatted downstream, not copied, so grounding must leave it be.
    assert out["fields"]["date"] == "NOONG 2023"


def test_a_number_the_page_does_not_state_drops_the_field(monkeypatch, keyed):
    """The failure from models/preds_gemini_text.csv, Blotter_Pic (100).

    Vision read "P10,000". The text guard reverted the model's "110,000"
    correctly - and "110,000 PESOS" was filed in the summary regardless, because
    _WORD_RE cannot see numbers and nothing else looked at the fields. A tenfold
    error in an amount is the failure this pipeline can least afford.
    """
    page = "ANG SANLA NA P10,000 PESOS AY HINDI NABAYARAN"
    _reply(monkeypatch, page, {"incident_summary": "ANG SANLA NA 110,000 PESOS"})
    out = gemini.restore(page)
    assert out["fields"]["incident_summary"] == ""
    assert "110000 is not on the page" in out["status"]


def test_the_same_amount_written_differently_is_still_grounded(monkeypatch, keyed):
    """Grounding compares digits, not formatting - it is the VALUE that has to
    be on the page. Otherwise every currency symbol becomes a false positive."""
    page = "ANG SANLA NA P10,000 PESOS AY HINDI NABAYARAN"
    _reply(monkeypatch, page, {"incident_summary": "ANG SANLA NA 10000 PESOS"})
    assert gemini.restore(page)["fields"]["incident_summary"] == "ANG SANLA NA 10000 PESOS"


def test_a_paraphrase_is_not_a_misread(monkeypatch, keyed):
    """MIN_RESTORE_SIMILARITY. A swap to an unrelated word is not a repair."""
    _reply(monkeypatch, "NAG5ADY4 SI JU4N SA TANGGAPAN H4LL NOONG 2023 ALAS 3:00")
    out = gemini.restore(PAGE)
    assert "BRGY" in out["text"]
    assert out["status"].startswith("rejected")


def test_a_real_misread_still_passes(monkeypatch, keyed):
    """The floor is not so high that it blocks the repairs this pass exists for."""
    _reply(monkeypatch, "NAGSADYA SI JU4N SA BRGY H4LL NOONG 2023 ALAS 3:00")
    out = gemini.restore(PAGE)
    assert "NAGSADYA" in out["text"]
    assert out["fidelity"]["distortion"] == 1 and out["fidelity"]["loss"] == 0


def test_invented_words_are_rejected_under_the_character_ceiling(monkeypatch, keyed):
    """MAX_TOKEN_GAIN, the axis nothing else bounded.

    Sentences the scanner never read, added to a long page, move only a few
    percent of its CHARACTERS - so MAX_RESTORE_CHANGE waves them through. In a
    blotter those are invented evidence, which is why gain gets its own ceiling.
    """
    long_page = PAGE + " " + " ".join(f"SALITA{n}" for n in range(40))
    _reply(monkeypatch, long_page + " AT SIYA AY UMAMIN NA SIYA ANG MAY SALA")
    out = gemini.restore(long_page)
    assert out["text"] == long_page
    assert "UMAMIN" not in out["text"]
    assert "not minimally lossy" in out["status"]


def test_a_few_recovered_words_are_still_allowed(monkeypatch, keyed):
    """The ceiling is a ceiling, not a ban - Vision drops words, and the point
    of reading the page again is to put some of them back."""
    long_page = PAGE + " " + " ".join(f"SALITA{n}" for n in range(40))
    _reply(monkeypatch, long_page + " SA ANUNAS")
    out = gemini.restore(long_page)
    assert "ANUNAS" in out["text"]
    assert out["fidelity"]["gain"] == 2


def test_fidelity_separates_the_three_error_types():
    """No network. The metric itself - repaired, added and dropped are not the
    same event and a single change-ratio cannot tell them apart."""
    f = gemini.fidelity("NAGSADYA DITO SA BRGY HALL", "NAGSADYA DITO SA BARANGAY HALL NG ANUNAS")
    assert (f["distortion"], f["gain"], f["loss"]) == (1, 2, 0)
    f = gemini.fidelity("KUMUHA NG MANOK NA PANABONG", "PAGKUHA NG MANOK")
    assert (f["distortion"], f["gain"], f["loss"]) == (1, 0, 2)


def test_off_without_a_backend(monkeypatch):
    monkeypatch.setattr(gemini, "_ask_restore",
                        lambda *a, **kw: pytest.fail("called with no backend"))
    out = gemini.restore(PAGE)
    assert out["text"] == PAGE
    assert out["edits"] == [] and out["fields"] == {}
    assert out["status"] == "off"
    # Present on every exit, so a caller can log it without branching on status.
    assert out["fidelity"]["loss"] == out["fidelity"]["gain"] == 0


def test_a_dead_network_degrades_the_scan_it_does_not_break_it(monkeypatch, keyed):
    def boom(*a, **kw):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(gemini, "_ask_restore", boom)
    out = gemini.restore(PAGE)
    assert out["text"] == PAGE
    assert out["status"].startswith("error")


def test_a_deadline_names_the_knob_that_fixes_it(monkeypatch, keyed):
    """504 DEADLINE_EXCEEDED is the failure this pass actually hits, because it
    asks the model to write the whole page back out. The status line is the only
    place an encoder sees it, so it has to say what to do, not just what broke."""
    def deadline(*a, **kw):
        raise RuntimeError("504 DEADLINE_EXCEEDED. Deadline expired before "
                           "operation could complete.")
    monkeypatch.setattr(gemini, "_ask_restore", deadline)
    out = gemini.restore(PAGE)
    assert out["text"] == PAGE          # the scan degrades, it does not break
    assert "BANTAY_GEMINI_TIMEOUT" in out["status"]


def test_malformed_reply_is_zero_edits_never_an_exception():
    assert gemini._parse_restore("not json at all") == (None, {})
    assert gemini._parse_restore("") == (None, {})
    assert gemini._parse_restore("[1, 2, 3]") == (None, {})


def test_non_string_field_values_are_dropped():
    """A field answered with a list is not an answer. An empty slot the encoder
    fills beats a stringified dict landing in a legal record."""
    _, fields = gemini._parse_restore(json.dumps(
        {"corrected_text": "X", "fields": {"date": "2026-01-03", "status": ["a", "b"],
                                           "location_purok": {"n": 5}}}))
    assert fields == {"date": "2026-01-03"}


def test_fields_survive_when_no_corrected_text_comes_back(monkeypatch, keyed):
    _reply(monkeypatch, None, {"status": "Settled"})
    out = gemini.restore(PAGE)
    assert out["text"] == PAGE
    assert out["fields"]["status"] == "Settled"


def test_empty_page_spends_no_quota(monkeypatch, keyed):
    monkeypatch.setattr(gemini, "_ask_restore",
                        lambda *a, **kw: pytest.fail("called on an empty page"))
    assert gemini.restore("   ")["status"].startswith("skipped")


def test_new_record_preview_renders_from_the_python_template():
    """records_new.html's New Record mode re-renders the narrative client-side
    (typed by hand, or pre-filled after a scan). It must fill narrative._TEMPLATE
    itself, passed in from the route, not a hand-kept copy - a copy drifts, and
    then the box the encoder reads stops matching the text the server stores."""
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent
            / "bantay" / "templates" / "records_new.html").read_text(encoding="utf-8")
    assert "narrative_template | tojson" in html
    assert "BLOTTER FORM" not in html.split("{% block scripts %}")[1]
    # Every slot has an input of the same name, or render() would store a
    # blank the encoder was never offered.
    for slot in narrative.SLOTS:
        assert f'name="{slot}"' in html, f"records_new.html has no input for {slot!r}"


def test_template_follows_the_paper_blotter_form():
    """Template/Template Blotter.png, section by section and in order."""
    out = narrative.render({})
    order = ["BLOTTER FORM", "NO.", "PETSA NG PAGBLOTTER",
             "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO", "TIRAHAN", "CONTACT NO.", "EDAD",
             "B. PANGALAN NG INIREREKLAMO", "C. REKLAMO", "D. KAGANAPAN NG PANGYAYARI",
             "PETSA", "ORAS", "LUGAR", "E. SALAYSAY", "LAGDA NG SUMULAT NG SALAYSAY",
             "PANGALAN NG SUMULAT NG SALAYSAY", "(MGA) TESTIGO",
             "PANGALAN NG BARANGAY DESK OFFICER / IMBESTIGADOR", "KATAYUAN", "AKSYONG GINAWA"]
    pos = 0
    for label in order:
        pos = out.index(label, pos)


def test_parse_round_trips_every_slot():
    """parse() is render()'s inverse - including the repeated sub-labels
    (TIRAHAN under A, B and the witness; PETSA under D and the writer), which
    only the section they sit in tells apart."""
    fields = {k: f"value {i}" for i, k in enumerate(narrative.SLOTS)}
    fields["incident_summary"] = "LINE ONE\nPETSA: written inside the narration"
    fields["action_taken"] = "Summoned both\nparties"
    assert narrative.parse(narrative.render(fields)) == [fields]


def test_parse_survives_lost_indentation():
    """pypdf drops leading spaces; sections are tracked by label, not indent."""
    text = "\n".join(line.lstrip() for line in narrative.render(
        {"complainant_address": "Purok 1", "respondent_address": "Purok 9",
         "incident_summary": "X"}).splitlines())
    entry = narrative.parse(text)[0]
    assert (entry["complainant_address"], entry["respondent_address"]) == ("Purok 1", "Purok 9")


def test_parse_still_reads_the_legacy_layout():
    """Records stored before the form-shaped template, and the shipped PDF
    import template, are in the old BARANGAY BLOTTER ENTRY layout."""
    legacy = narrative._LEGACY_TEMPLATE.format(
        date="2023-01-02", time="09:00", location_purok="Purok 3",
        reporting_party="Juan", respondent="Pedro", status="Filed",
        incident_summary="Nawala ang manok.", action_taken="Summoned")
    entry = narrative.parse(legacy)[0]
    assert (entry["date"], entry["reporting_party"], entry["respondent"]) == ("2023-01-02", "Juan", "Pedro")
    assert (entry["incident_summary"], entry["action_taken"]) == ("Nawala ang manok.", "Summoned")


# --- redaction ---------------------------------------------------------------

def test_redact_names_masks_both_party_lines():
    text = narrative.render({"reporting_party": "Juan Dela Cruz", "respondent": "Pedro Santos",
                              "incident_summary": "Theft of poultry reported."})
    out = narrative.redact_names(text)
    assert "Juan Dela Cruz" not in out
    assert "Pedro Santos" not in out
    assert "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO : [REDACTED]" in out
    assert "B. PANGALAN NG INIREREKLAMO : [REDACTED]" in out
    # Only the two labelled name lines change - the rest of the entry is untouched.
    assert "Theft of poultry reported." in out


def test_redact_names_masks_even_an_unstated_name_line():
    """render() fills an empty name slot with the BLANK marker, not silence -
    redact_names must still mask that line, since (not stated on page) is
    itself a template-shaped string an over-narrow regex could skip."""
    text = narrative.render({"incident_summary": "X"})
    out = narrative.redact_names(text)
    name_lines = [l for l in out.splitlines() if l.startswith(("A. PANGALAN", "B. PANGALAN"))]
    assert len(name_lines) == 2
    assert all(narrative.BLANK not in l for l in name_lines)
    assert "X" in out  # unrelated slots untouched


def test_redact_names_handles_empty_input():
    assert narrative.redact_names("") == ""
    assert narrative.redact_names(None) == ""


def test_redact_names_also_catches_bare_mentions_in_the_free_prose():
    """A name isn't confined to its own labelled line - the encoder's prose
    in E. SALAYSAY routinely repeats it, first name or surname alone."""
    text = narrative.render({
        "reporting_party": "Juan Dela Cruz",
        "respondent": "Pedro Santos",
        "incident_summary": "NAGSADYA SI JUAN DELA CRUZ LABAN KAY PEDRO SANTOS. Si Santos ay tumangging umamin.",
        "action_taken": "Pinag-usapan sina Juan at Pedro sa opisina.",
    })
    out = narrative.redact_names(text)
    for leaked in ("Juan", "Cruz", "Pedro", "Santos"):
        assert leaked not in out, f"{leaked!r} leaked into: {out}"
    assert "NAGSADYA SI" in out and "LABAN KAY" in out and "opisina" in out


def test_redact_names_does_not_touch_unrelated_prose():
    text = narrative.render({
        "reporting_party": "Juan Dela Cruz",
        "incident_summary": "Nawala ang alagang aso sa may plaza noong gabi.",
    })
    out = narrative.redact_names(text)
    assert "Nawala ang alagang aso sa may plaza noong gabi." in out


def test_redact_masks_addresses_contacts_and_the_witness():
    text = narrative.render({
        "reporting_party": "Juan Dela Cruz", "complainant_address": "Blk 4 Lot 2 Anunas",
        "complainant_contact": "09171234567", "witness": "Maria Reyes",
        "witness_address": "Purok 1", "desk_officer": "Kag. Santos",
        "incident_summary": "Tumawag si Maria sa 09171234567.",
    })
    out = narrative.redact_names(text)
    for leaked in ("Blk 4 Lot 2", "09171234567", "Maria", "Reyes", "Purok 1"):
        assert leaked not in out, f"{leaked!r} leaked into: {out}"
    # The desk officer is barangay staff acting officially, not a party.
    assert "Kag. Santos" in out


def test_redact_still_masks_legacy_records():
    legacy = narrative._LEGACY_TEMPLATE.format(
        date="", time="", location_purok="", reporting_party="Juan Dela Cruz",
        respondent="Pedro Santos", status="", incident_summary="Si Pedro ay umalis.",
        action_taken="")
    out = narrative.redact_names(legacy)
    assert "RESPONDENT        : [REDACTED]" in out
    assert "Pedro" not in out


def test_record_columns_follow_the_corpus_convention():
    """The record's date is the entry date and its location the purok - see
    narrative.record_columns. The narrative itself is left as the page said."""
    cols = narrative.record_columns({"date_reported": "2023-11-20", "date": "2023-10-28",
                                     "location_purok": "SIDE NG BAHAY KO",
                                     "complainant_address": "P-4, Brgy. Anunas"})
    assert cols == {"date": "2023-11-20", "location_purok": "Purok 4"}
    # D. Petsa and the Lugar text are used only when nothing better exists.
    assert narrative.record_columns({"date": "2023-10-28", "location_purok": "Plaza"}) == \
        {"date": "2023-10-28", "location_purok": "Plaza"}
    # A page with only a filing date is not reported as missing its date.
    assert "date" not in narrative.missing({"date_reported": "2023-11-20"})


def test_an_address_keeps_a_purok_digit_vision_dropped(monkeypatch, keyed):
    """Vision read "ADD. PUROK BRGY ANUNAS"; the photo shows Purok 1. The number
    guard must not drop the address for that - other slots still get it."""
    page = "NAGSADYA SI JUAN ADD. PUROK BRGY ANUNAS"
    _reply(monkeypatch, page, {"complainant_address": "PUROK 1 BRGY ANUNAS",
                               "complainant_contact": "09170000000"})
    fields = gemini.restore(page)["fields"]
    assert fields["complainant_address"] == "PUROK 1 BRGY ANUNAS"
    assert fields["complainant_contact"] == ""
