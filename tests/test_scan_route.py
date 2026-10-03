"""End-to-end smoke test for routes/scan.py's reconcile() wiring, exercised
through the merged New Record page (records.new_record) that now embeds the
scan-intake step - see routes/scan.py's run_scan_pipeline and
routes/records.py's new_record().

Unit tests already cover guess_fields() (test_ocr.py), reconcile()
(test_reconcile.py) and backfill()'s PREFER_FALLBACK (test_narrative.py)
individually. This is the one check that they compose correctly inside the
actual route handler - stubbed OCR/Gemini, real Flask request, real
narrative/reconcile code path, checking the page renders and the disagreement
actually reaches the encoder instead of one reader silently winning.
"""
import io

import pytest

from bantay import create_app, db
from bantay.models import User


@pytest.fixture
def app(tmp_path):
    flask_app = create_app(overrides={
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "TESTING": True,
        "MODEL_DIR": str(tmp_path / "models"),
    })
    with flask_app.app_context():
        staff = User(username="staff", role="staff")
        staff.set_password("staffpass")
        db.session.add(staff)
        db.session.commit()
    yield flask_app


def test_new_record_scan_upload_surfaces_field_disagreement_and_prefers_regex_status(app, monkeypatch):
    from bantay import ocr
    from bantay.ml import sealion_classify
    from bantay.ocr import gemini

    monkeypatch.setattr(ocr, "available_engines", lambda: ["gvision"])
    monkeypatch.setattr(ocr, "extract", lambda stream, backend=None: {
        "text": "NAGSADY4 SI JUAN SA BRGY HALL NOV. 19, 2023 12:50 PM. "
                "COMPLAINANT SCHEDULE / NOV 25, 2023 TIME 10:00 AM",
        "engine": "gvision", "mean_conf": 0.9})
    monkeypatch.setattr(gemini, "available", lambda: True)
    monkeypatch.setattr(gemini, "restore", lambda text, lexicon=None: {
        "text": text.replace("NAGSADY4", "NAGSADYA"),
        "edits": [{"before": "NAGSADY4", "after": "NAGSADYA", "score": 0.9,
                   "source": "gemini", "reason": "digit for letter"}],
        # Disagrees with the regex reader's "For Hearing" (from SCHEDULE) -
        # this is the exact shape measured via tools/run_gemini_arms.py.
        "fields": {"status": "SETTLEMENT"},
        "status": "ok: 0 edits (stub)"})
    # Off by default (BANTAY_OLLAMA unset), but forced explicitly so this test
    # never depends on env state or spends time on a real local model call.
    monkeypatch.setattr(sealion_classify, "available", lambda: False)

    client = app.test_client()
    client.post("/login", data={"username": "staff", "password": "staffpass"})
    data = {"image": (io.BytesIO(b"fake image bytes"), "test.jpg")}
    resp = client.post("/records/new", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "The two readers disagree on" in body
    # PREFER_FALLBACK: the regex reading must win the status field, not Gemini's.
    assert "For Hearing" in body
    # With SEA-LION off, the page must SAY so rather than silently presenting
    # the TF-IDF fallback's answer as the pipeline's - a misconfigured Ollama used to be
    # indistinguishable from a working one. See routes/scan.py step 4.
    assert "off (set BANTAY_OLLAMA=1)" in body
    assert "SEA-LION did not answer" in body
    # The encoder is reading this draft right now, so a scan never queues itself
    # for a second review - the warnings above the Save button ARE the review.
    # See routes/scan.py step 5.
    assert 'name="review_status" value="accepted"' in body
    # One click must send one POST. Without the lock the browser can deliver a
    # single click twice and the encoder gets two identical records - see
    # tests/test_records_route.py for the server-side half of this.
    assert "saveBtn.disabled = true" in body
    assert "Check these before saving" in body
    assert "needs_review" not in body
    # The extracted narrative must still render through the BARANGAY BLOTTER
    # ENTRY template - a scanned record must stay shaped exactly like a
    # manually-encoded one (see bantay/narrative.py).
    assert "Blotter Record Template" in body
    # What Gemini replaced must never reach the page - not the misread token,
    # not the before/after table, not the raw/repaired text panels. Dropped in
    # run_scan_pipeline itself, so it is absent from the HTML source too.
    assert "NAGSADY4" not in body
    assert "digit for letter" not in body
    assert "What Gemini changed" not in body
    assert "Raw Vision output" not in body and "After repair" not in body
    # Readability (Vision's own mean confidence, stubbed at 0.9 above) must be
    # shown as a percentage, not a raw 0-1 decimal.
    assert "Readability" in body and "90%" in body
    assert "BLOTTER FORM" in body
    assert "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO :" in body and "B. PANGALAN NG INIREREKLAMO :" in body
    # The field the manual-entry form used to expose for this is gone - see
    # test_records_route.py's persons_involved_masked removal test.
    assert "Persons Involved" not in body


def test_new_record_scan_upload_uses_sealion_classification_as_primary(app, monkeypatch):
    """The local SEA-LION few-shot classify() is primary when available (its
    Gemini-backed predecessor measured 69.0% vs the removed encoder's 6.7% real-test
    accuracy - see tools/eval_gemini_classify.py). This exercises that branch
    specifically, since the other test forces sealion_classify off to avoid
    depending on a real local Ollama model on every test run."""
    from bantay import ocr
    from bantay.ml import sealion_classify
    from bantay.ocr import gemini, sealion

    monkeypatch.setattr(ocr, "available_engines", lambda: ["gvision"])
    monkeypatch.setattr(ocr, "extract", lambda stream, backend=None: {
        "text": "NAGSADYA SI JUAN UPANG IREKLAMO ANG PAGKAWALA NG MANOK.",
        "engine": "gvision", "mean_conf": 0.9})
    monkeypatch.setattr(gemini, "available", lambda: False)     # OCR-repair arm off
    # Same reasoning as the other test's sealion_classify stub below: off by
    # default (BANTAY_OLLAMA unset), forced explicitly so this test exercises
    # the lexicon-corrector fallback it is named for, not a real Ollama call,
    # regardless of the running shell's env state.
    monkeypatch.setattr(sealion, "available", lambda: False)
    monkeypatch.setattr(sealion_classify, "available", lambda: True)
    monkeypatch.setattr(sealion_classify, "classify", lambda text, **kw: {
        "incident_type": "Theft", "category_group": "Criminal/Penal Code",
        "confidence_label": "high", "reason": "manok stolen",
        "status": "ok (stub)"})

    client = app.test_client()
    client.post("/login", data={"username": "staff", "password": "staffpass"})
    data = {"image": (io.BytesIO(b"fake image bytes"), "test.jpg")}
    resp = client.post("/records/new", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "sealion few-shot" in body
    assert "Theft" in body
    # The classification reason must be shown, labelled, next to the
    # suggested type - not just carried in the response and dropped.
    assert "Reason: manok stolen" in body
    # Both percentages the encoder needs at a glance: how well the page was
    # read, and how confident the classifier is in the suggested type.
    # confidence_label "high" maps to 0.8 (see routes/scan.py step 4).
    assert "Readability" in body and "90%" in body
    assert "Classification Confidence Score" in body and "80%" in body
    # Repair edits are never shown, whichever repair engine ran.
    assert "What Gemini changed" not in body


def test_scan_url_redirects_to_new_record(app):
    """Old bookmarks/links to the standalone scan page must not 404 - scanning
    now lives on the New Record page (see routes/scan.py's scan_page())."""
    client = app.test_client()
    client.post("/login", data={"username": "staff", "password": "staffpass"})
    resp = client.get("/scan/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/records/new")
