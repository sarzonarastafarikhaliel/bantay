"""Save-path checks for routes/records.py.

The one thing here that is not obvious from reading the route: saving is not a
repeatable action, but the endpoint accepted it as one. Two byte-identical rows
6 ms apart turned up in the pilot database (records 6 and 7, same page, same
encoder, same narrative) - one click delivered as two POSTs. Neither form that
posts here carries a CSRF token or a submission id, so nothing distinguished the
replay from a second save.
"""
from datetime import datetime, timedelta

import pytest

from bantay import create_app, db
from bantay.models import IncidentRecord, User
from bantay.routes import records as records_route
from bantay.ml import sealion_classify


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
        admin = User(username="boss", role="admin")
        admin.set_password("bosspass")
        db.session.add(admin)
        db.session.commit()
    yield flask_app


@pytest.fixture
def client(app):
    c = app.test_client()
    c.post("/login", data={"username": "staff", "password": "staffpass"})
    return c


@pytest.fixture
def admin_client(app):
    c = app.test_client()
    c.post("/login", data={"username": "boss", "password": "bosspass"})
    return c


DRAFT = {
    "date": "2026-01-02",
    "time": "12:30",
    "location_purok": "Purok 3",
    "incident_summary": "NAGSADYA SI [PERSON_1] UPANG IPA-BLOTTER ANG PAGKUHA NG MANOK.",
    "action_taken": "Settlement / confrontation held at barangay",
    "status": "settled",
    "incident_type_primary": "Property Claim / Damage",
    "review_status": "accepted",
    "model_confidence": "0.8",
    "source_file_page": "20260506_130750.jpg",
}


def test_replayed_save_does_not_create_a_second_record(app, client):
    """The reported bug: one Save click, two identical records."""
    first = client.post("/records/new", data=DRAFT)
    second = client.post("/records/new", data=DRAFT)

    assert first.status_code == 302
    assert second.status_code == 302
    with app.app_context():
        assert IncidentRecord.query.count() == 1


def test_replay_redirects_to_the_record_that_was_actually_saved(app, client):
    """A swallowed replay must not look like a failure - the encoder pressed
    Save and a record exists, so send them to it and say what happened."""
    client.post("/records/new", data=DRAFT)
    with app.app_context():
        saved_id = IncidentRecord.query.one().id

    resp = client.post("/records/new", data=DRAFT, follow_redirects=True)
    assert f"Already saved as record #{saved_id}" in resp.data.decode("utf-8")


def test_a_genuinely_different_record_still_saves(app, client):
    """The guard keys on the narrative, so it must not swallow a second real
    incident logged in the same minute - a busy afternoon at the barangay hall
    is not a bug."""
    client.post("/records/new", data=DRAFT)
    client.post("/records/new",
                data=dict(DRAFT, incident_summary="NAWALA ANG ALAGANG ASO NI [PERSON_2]."))

    with app.app_context():
        assert IncidentRecord.query.count() == 2


def test_the_same_incident_re_entered_later_is_not_blocked(app, client):
    """The window is 60 seconds, not forever: an encoder who deliberately
    re-enters an identical narrative an hour later must not be silently
    refused."""
    client.post("/records/new", data=DRAFT)
    with app.app_context():
        record = IncidentRecord.query.one()
        record.created_at = datetime.utcnow() - timedelta(hours=1)
        db.session.commit()

    client.post("/records/new", data=DRAFT)
    with app.app_context():
        assert IncidentRecord.query.count() == 2


# --- respondent-name redaction ------------------------------------------------
# See bantay/narrative.py's redact_names and bantay/__init__.py's
# visible_narrative filter: the party name lines embedded in the
# free-text narrative are masked for every role except admin.

NAME_DRAFT = dict(DRAFT, reporting_party="Juan Dela Cruz", respondent="Pedro Santos",
                  incident_summary="Theft of poultry reported by Juan Dela Cruz.")


def test_non_admin_sees_redacted_party_names(app, client):
    client.post("/records/new", data=NAME_DRAFT)
    with app.app_context():
        record_id = IncidentRecord.query.one().id

    resp = client.get(f"/records/{record_id}")
    body = resp.data.decode("utf-8")
    assert "Pedro Santos" not in body
    assert "B. PANGALAN NG INIREREKLAMO : [REDACTED]" in body


def test_admin_sees_the_real_party_names(app, client, admin_client):
    client.post("/records/new", data=NAME_DRAFT)
    with app.app_context():
        record_id = IncidentRecord.query.one().id

    resp = admin_client.get(f"/records/{record_id}")
    assert "Pedro Santos" in resp.data.decode("utf-8")


# --- matching the scan page's classification pipeline ------------------------
# See routes/records.py's _classify_narrative: SEA-LION first, legacy TF-IDF
# fallback, same priority order routes/scan.py already used for scanned pages.
# This is what backs both /records/predict (the new_record form's live
# preview) and new_record()'s own final classification on save.

def test_classify_narrative_prefers_sealion_when_available(monkeypatch):
    monkeypatch.setattr(sealion_classify, "available", lambda: True)
    monkeypatch.setattr(sealion_classify, "classify", lambda text, model=None: {
        "incident_type": "Neighbor Dispute", "confidence_label": "high",
        "category_group": "Community Disturbance", "reason": "loud argument",
    })
    result = records_route._classify_narrative("Ingay sa magkapitbahay.")
    assert result["incident_type"] == "Neighbor Dispute"
    assert result["confidence"] == 0.8
    assert result["category_group"] == "Community Disturbance"
    assert result["source"] == "sealion"


def test_classify_narrative_falls_back_to_tfidf_when_sealion_off(monkeypatch, app):
    monkeypatch.setattr(sealion_classify, "available", lambda: False)
    with app.app_context():
        result = records_route._classify_narrative("")
    # No SEA-LION, empty text -> nothing for the fallback classifier either.
    assert result["incident_type"] is None
    assert result["source"] == "none"


def test_predict_endpoint_returns_pnp_and_kp_screening(client, monkeypatch):
    monkeypatch.setattr(sealion_classify, "available", lambda: True)
    monkeypatch.setattr(sealion_classify, "classify", lambda text, model=None: {
        "incident_type": "Slight Physical Injury", "confidence_label": "high",
        "category_group": "Criminal/Penal Code", "reason": "minor injury reported",
    })
    resp = client.post("/records/predict", json={
        "narrative": "Sinaktan ng kapitbahay, sugat lang sa braso.",
        "date": "", "time": "", "location_purok": "",
    })
    data = resp.get_json()
    assert data["label"] == "Slight Physical Injury"
    assert data["source"] == "sealion"
    # Slight Physical Injury is an Index Crime that is nonetheless KP-Mediable
    # (RPC Art. 266 caps at arresto menor) - see narrative.py/normalize.py §3.2.4.
    assert data["pnp_tier"] == "Index Crime"
    assert data["kp_status"] == "KP-Mediable"


def test_new_record_form_fields_render_through_the_same_template_as_scan(app, client):
    """The New Record page now posts the same slot shape the scan draft does
    (reporting_party/respondent/incident_summary), so it renders through
    narrative.render() and gets the redaction/KP/PNP handling scan-created
    records already get - see NAME_DRAFT above for the redaction half."""
    client.post("/records/new", data=NAME_DRAFT)
    with app.app_context():
        record = IncidentRecord.query.one()
        assert "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO : Juan Dela Cruz" in record.narrative
        assert "B. PANGALAN NG INIREREKLAMO : Pedro Santos" in record.narrative
        assert "E. SALAYSAY" in record.narrative
        # incident_type_primary was posted manually in DRAFT, so KP/PNP/category
        # are still derived automatically from it exactly as scan.py does.
        assert record.kp_status is not None
        assert record.pnp_classification is not None
        assert record.category_group is not None


def test_editing_a_record_does_not_wipe_its_existing_persons_involved_masked(app, admin_client):
    """The New Record/Edit form dropped the Persons Involved (Masked) field
    from the UI, but existing records (e.g. from CSV import) may still carry
    a value in that column - editing an unrelated field must not silently
    null it out just because the form no longer posts it."""
    with app.app_context():
        record = IncidentRecord(
            date="2026-01-02", narrative="X", persons_involved_masked="P1 (complainant)",
        )
        db.session.add(record)
        db.session.commit()
        record_id = record.id

    edit_data = dict(DRAFT)
    del edit_data["incident_summary"]
    edit_data["narrative"] = "Edited narrative text."
    admin_client.post(f"/records/{record_id}/edit", data=edit_data)

    with app.app_context():
        record = db.session.get(IncidentRecord, record_id)
        assert record.narrative == "Edited narrative text."
        assert record.persons_involved_masked == "P1 (complainant)"


def test_kp_status_filter_shows_only_matching_records(app, client):
    """Lets the Lupon Secretary pull up the KP-Mediable caseload directly,
    instead of only seeing aggregate counts on the dashboard - see the
    dashboard's Katarungang Pambarangay Referability panel, which now
    deep-links here with the same kp_status param."""
    with app.app_context():
        db.session.add(IncidentRecord(date="2026-01-01", narrative="Mediable one",
                                       incident_type_primary="Neighbor Dispute",
                                       kp_status="KP-Mediable"))
        db.session.add(IncidentRecord(date="2026-01-02", narrative="Excluded one",
                                       incident_type_primary="RA 9262 (VAWC)",
                                       kp_status="Excluded"))
        db.session.commit()

    resp = client.get("/records/?kp_status=KP-Mediable")
    body = resp.data.decode("utf-8")
    assert "Mediable one" in body
    assert "Excluded one" not in body

    resp_all = client.get("/records/")
    body_all = resp_all.data.decode("utf-8")
    assert "Mediable one" in body_all and "Excluded one" in body_all


def test_new_record_page_offers_the_evaluated_models(client):
    """The dropdown is the whole feature: if the template stops rendering it,
    the model is silently pinned to the default and nobody notices."""
    html = client.get("/records/new").get_data(as_text=True)
    assert 'id="pred-model"' in html
    for entry in sealion_classify.EVALUATED_MODELS:
        assert entry["tag"] in html


def test_predict_honours_the_selected_model(client, monkeypatch):
    """A chosen checkpoint must reach classify() as the primary call and be
    reported back, so the encoder can tell which model produced the
    suggestion on screen. classify_with_agreement() also makes a second,
    check-model call (§ the confidence-agreement feature), so `seen` records
    every call and this asserts on the first - the primary - not the last."""
    seen = []

    def fake(text, model=None):
        seen.append(model)
        return {"incident_type": "Theft", "category_group": "Criminal/Penal Code",
                "confidence_label": "high", "reason": "took an item", "status": "ok"}

    monkeypatch.setattr(sealion_classify, "available", lambda: True)
    monkeypatch.setattr(sealion_classify, "classify", fake)

    picked = "aisingapore/Gemma-SEA-LION-v4-4B-VL"
    resp = client.post("/records/predict",
                       json={"narrative": "NAGNAKAW NG MANOK SA KATABING BAHAY.",
                             "model": picked})
    assert seen[0] == picked
    assert resp.get_json()["model"] == picked


def test_predict_ignores_a_model_that_is_not_on_the_evaluated_list(client, monkeypatch):
    """Untrusted field: an off-list tag must never reach the Ollama call.
    See test_predict_honours_the_selected_model for why `seen` is a list."""
    seen = []

    def fake(text, model=None):
        seen.append(model)
        return {"incident_type": "Theft", "category_group": "Criminal/Penal Code",
                "confidence_label": "high", "reason": "", "status": "ok"}

    monkeypatch.setattr(sealion_classify, "available", lambda: True)
    monkeypatch.setattr(sealion_classify, "classify", fake)

    client.post("/records/predict",
                json={"narrative": "NAGNAKAW NG MANOK SA KATABING BAHAY.",
                      "model": "attacker/whatever:latest"})
    assert seen[0] == sealion_classify._DEFAULT_MODEL
