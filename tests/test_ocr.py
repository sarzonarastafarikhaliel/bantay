"""Checks for the OCR intake path: noise model, corrector, field extraction.

Deliberately small. These cover the four things that would silently break the
pipeline without anyone noticing: the corrector inventing text, the corrector
touching real numbers, the noise model losing determinism, and the date parser
emitting an invalid month.
"""
import sys
from types import SimpleNamespace as NS

import pytest

from bantay.ocr.correct import OCRCorrector, build_lexicon
from bantay.ocr.noise import cer, corrupt, fit_confusions, wer

CLEAN = ("NAGSADYA DITO SA BRGY. HALL NG ANUNAS UPANG IPA-BLOTTER ANG NANGYARING "
         "PANLOLOOB SA KANILANG BAHAY NOONG 2023")
LEXICON = build_lexicon([CLEAN] * 2)


def test_corrupt_changes_text_and_is_deterministic():
    a = corrupt(CLEAN, rate=0.15, seed=7)
    b = corrupt(CLEAN, rate=0.15, seed=7)
    assert a == b, "same seed must reproduce the same page - the audit trail rests on it"
    assert a != CLEAN
    assert 0.02 < cer(a, CLEAN) < 0.5


def test_cer_and_wer_bounds():
    assert cer(CLEAN, CLEAN) == 0.0
    assert wer(CLEAN, CLEAN) == 0.0
    assert cer("", CLEAN) == pytest.approx(1.0)


def test_fit_confusions_learns_from_pairs():
    table = fit_confusions([("N4GSADYA", "NAGSADYA")], min_count=1)
    assert "A" in table and "4" in table["A"]


def test_corrector_only_emits_lexicon_tokens():
    """The safety property: it may replace a word, never invent one."""
    noisy = corrupt(CLEAN, rate=0.2, seed=3)
    result = OCRCorrector(lexicon=LEXICON).correct(noisy)
    for edit in result["edits"]:
        assert edit["after"].upper() in LEXICON
    assert len(result["text"].split()) == len(noisy.split())


def test_corrector_leaves_numbers_alone():
    text = "NAGSADYA DITO NOONG 2023 AT 12:30"
    result = OCRCorrector(lexicon=LEXICON).correct(text)
    assert "2023" in result["text"]
    assert "12:30" in result["text"]


def test_corrector_without_lexicon_is_a_no_op():
    text = "ANYTHING AT ALL"
    assert OCRCorrector(lexicon={}).correct(text)["text"] == text


def test_corrector_bar_rejects_a_weak_lexical_match():
    """Regression: the acceptance bar used to sink as the (now removed) LM
    weight rose - 0.86 lexical-only against 0.22 at w=0.8 - which is what put
    11,084 edits at 0.58 precision into models/ocr_correction_report.json.
    One channel now, one bar, and it is the strict one.
    """
    # NAG5ADYA vs NAGSADYA: similarity 0.875, over the 0.86 bar - this lands.
    assert OCRCorrector(lexicon=LEXICON).correct("NAG5ADYA")["text"] == "NAGSADYA"
    # NAGSADY4X vs NAGSADYA: similarity 0.824, under the bar - left alone.
    assert OCRCorrector(lexicon=LEXICON).correct("NAGSADY4X")["n_edits"] == 0


def test_guess_fields_parses_both_date_styles_and_rejects_nonsense():
    from bantay import create_app
    from bantay.routes.scan import guess_fields

    app = create_app(overrides={"SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.test_request_context():
        assert guess_fields("NOONG JAN. 3, 2026 GANAP NA 12:30 PM")["date_reported"] == "2026-01-03"
        assert guess_fields("PETSA 1-3-25 ORAS 9:15 AM")["date_reported"] == "2025-01-03"
        # 25 cannot be a month; the fallback must swap rather than emit month 25
        assert guess_fields("PETSA 25-12-24")["date_reported"] == "2024-12-25"
        assert guess_fields("WALANG PETSA")["date_reported"] == ""


def test_guess_fields_purok_suffix_status_and_action(monkeypatch):
    """Regression for the location_purok/status/action_taken fixes.

    Purok suffix: normalize_purok used to drop the trailing letter entirely
    (Purok 4B -> Purok 4). Status/action: guess_fields used to never fill
    either slot, so the offline path scored 0% on both no matter what the
    page said.
    """
    from bantay import create_app
    from bantay.routes import scan as scan_mod
    from bantay.routes.scan import guess_fields

    monkeypatch.setattr(scan_mod, "_puroks", lambda: [])
    app = create_app(overrides={"SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.test_request_context():
        assert guess_fields("NAGTUNGO SA PUROK 4B EXT BRGY ANUNAS")["location_purok"] == "Purok 4B"

        settled = guess_fields("KAMI AY NAGKASUNDO AT LALAGDA SA KASULATANG ITO")
        assert settled["status"] == "Settled"
        assert settled["action_taken"] == "Settlement / confrontation held at barangay"

        hearing = guess_fields("COMPLAINANT SCHEDULE / NOVEMBER 23-2023 TIME 10:00 AM")
        assert hearing["status"] == "For Hearing"
        assert "NOVEMBER 23-2023" in hearing["action_taken"]
        assert "10:00 AM" in hearing["action_taken"]

        # No resolution language on the page: stay blank, never guess "Filed".
        # "Filed" comes from the page's own FOR BLOTTER marker, not from the
        # absence of other signals - Vision's O/0 and TT/TI confusions on that
        # word are in the pattern. Resolution language still outranks it: a
        # page carrying both was settled after it was filed.
        assert guess_fields("* FON BLOTTER * DUMATING SA TANGGAPAN")["status"] == "Filed"
        assert guess_fields("NASSADYA SA BRGY UPANG IPA-BL0TTER ANG NANGYARI")["status"] == "Filed"
        assert guess_fields("FOR BLOTTER ... KAMI AY NAGKASUNDO")["status"] == "Settled"

        neutral = guess_fields("NAGSADYA SI JUAN UPANG IREKLAMO ANG UTANG")
        assert neutral["status"] == ""
        assert neutral["action_taken"] == ""


# --- gvision -----------------------------------------------------------------
# The backend is stubbed rather than mocked against the real client: the two
# things worth pinning are the line rebuild and the quota exit, and neither
# needs a network or a Google SDK install to exercise.

class _Exhausted(Exception):
    pass


def _stub_google(monkeypatch, resp=None, raises=None):
    """Put a fake google.cloud.vision + google.api_core into sys.modules."""
    def call(**kwargs):
        if raises is not None:
            raise raises
        return resp

    exceptions = NS(ResourceExhausted=_Exhausted, TooManyRequests=type("TooMany", (Exception,), {}),
                    ServiceUnavailable=type("Unavail", (Exception,), {}),
                    DeadlineExceeded=type("Deadline", (Exception,), {}),
                    InternalServerError=type("ISE", (Exception,), {}),
                    RetryError=type("RetryError", (Exception,), {}),
                    GoogleAPICallError=type("APICallError", (Exception,), {"message": ""}))
    vision = NS(ImageAnnotatorClient=lambda: NS(document_text_detection=call),
                Image=lambda **kw: None,
                ImageContext=lambda **kw: None,
                TextAnnotation=NS(DetectedBreak=NS(BreakType=BREAKS)))
    api_core = NS(exceptions=exceptions,
                  retry=NS(Retry=lambda **kw: None, if_exception_type=lambda *a: None))
    monkeypatch.setitem(sys.modules, "google", NS(cloud=NS(vision=vision), api_core=api_core))
    monkeypatch.setitem(sys.modules, "google.cloud", NS(vision=vision))
    monkeypatch.setitem(sys.modules, "google.cloud.vision", vision)
    monkeypatch.setitem(sys.modules, "google.api_core", api_core)

    from bantay.ocr import engine
    monkeypatch.setattr(engine, "_READERS", {})   # fresh client cache, restored after
    return engine


BREAKS = NS(SPACE=1, SURE_SPACE=2, EOL_SURE_SPACE=3, HYPHEN=4, LINE_BREAK=5)


def _word(text, conf, brk=0):
    syms = [NS(text=c, property=NS(detected_break=NS(type_=0))) for c in text]
    syms[-1].property.detected_break.type_ = brk
    return NS(symbols=syms, confidence=conf)


def test_gvision_rebuilds_lines_from_break_hints(monkeypatch):
    """Vision returns words, not lines. The pipeline downstream assumes lines,
    so the break-hint walk is the one piece of gvision that can silently lose
    text - a missed flush swallows the last line of every paragraph.
    """
    para = NS(words=[_word("NAGSADYA", 0.9, BREAKS.SPACE),
                     _word("DITO", 0.8, BREAKS.LINE_BREAK),
                     _word("SA", 0.7, BREAKS.SPACE),
                     _word("HALL", 0.5)])           # no hint: paragraph end must flush
    resp = NS(error=NS(message=""),
              full_text_annotation=NS(pages=[NS(blocks=[NS(paragraphs=[para])])]))

    engine = _stub_google(monkeypatch, resp=resp)
    lines = engine._gvision(b"", ("tl", "en"), gpu=False)
    assert [l["text"] for l in lines] == ["NAGSADYA DITO", "SA HALL"]
    assert lines[0]["conf"] == pytest.approx(0.85)
    assert lines[1]["conf"] == pytest.approx(0.60)


def test_gvision_quota_raises_quota_exceeded_not_bare_runtime_error(monkeypatch):
    """A bulk run must be able to tell 'this page failed' from 'the quota is
    gone', or it skips every remaining page and writes a truncated file.
    """
    from bantay.ocr.engine import QuotaExceeded

    engine = _stub_google(monkeypatch, raises=_Exhausted("429 quota"))
    with pytest.raises(QuotaExceeded) as caught:
        engine._gvision(b"", ("tl", "en"), gpu=False)
    assert "quota" in str(caught.value).lower()
    assert isinstance(caught.value, RuntimeError), "callers catching RuntimeError must still work"


def test_as_image_applies_exif_rotation():
    """A phone-captured page carries an EXIF orientation tag; 10 of the 104
    collected scans do. PIL does not apply it on open, so without the transpose
    those pages reach the recognizer rotated and score as garbage - a loader
    artifact that would be misread as the engine failing.
    """
    import io

    from PIL import Image

    from bantay.ocr.engine import _as_image

    portrait = Image.new("RGB", (40, 90), "white")
    buf = io.BytesIO()
    exif = portrait.getexif()
    exif[274] = 6  # rotate 90 CW on display
    portrait.save(buf, format="JPEG", exif=exif)

    raw = Image.open(io.BytesIO(buf.getvalue()))
    assert raw.size == (40, 90), "PIL leaves the tag unapplied - the bug this guards"
    assert _as_image(buf.getvalue()).size == (90, 40)
