import io
import os

import pytest

from bantay import create_app, db as _db
from bantay.models import IncidentRecord, User
from bantay.routes import records as records_route
from bantay.routes.records import IMPORT_COLUMNS, import_csv_rows

PDF_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "bantay", "static",
                            "templates", "blotter_import_template.pdf")


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "test.db"
    flask_app = create_app(
        overrides={"SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}", "TESTING": True}
    )
    yield flask_app


def test_import_skips_blank_and_duplicate_rows_and_keeps_unreadable_out_of_ml(app):
    rows = [
        {"date": "2023-01-01", "incident_summary": "Loud noise complaint from neighbor",
         "readability": "readable", "incident_type_primary": "Theft"},
        {"date": "2023-01-01", "incident_summary": "Loud noise complaint from neighbor",
         "readability": "readable", "incident_type_primary": "Theft"},  # duplicate
        {"date": "2023-01-02", "incident_summary": "", "readability": "readable",
         "incident_type_primary": "Theft"},  # no narration - the form would refuse it
        {"date": "2023-01-03", "incident_summary": "Illegible handwriting sample text",
         "readability": "unreadable", "incident_type_primary": "Theft"},
        {"date": "2023-01-04", "incident_summary": "Reported theft of bicycle in purok 2",
         "readability": "readable", "incident_type_primary": "Theft"},
    ]
    with app.app_context():
        stats = import_csv_rows(rows)
        assert stats == {"saved": 3, "excluded": 1, "skipped_blank": 1, "skipped_existing": 1}
        assert IncidentRecord.query.count() == 3
        unreadable = IncidentRecord.query.filter_by(readability="unreadable").one()
        assert unreadable.include_in_ml is False
        assert unreadable.review_status == "needs_review"


def test_import_renders_the_new_blotter_record_template(app):
    """An imported row is stored exactly like a New Blotter Record save: the
    slots rendered through narrative.render, type-derived PNP/KP fields."""
    rows = [{
        "date": "2023-05-10", "time": "2:30 PM", "location_purok": "Purok 4",
        "reporting_party": "Juan", "respondent": "Pedro",
        "incident_summary": "Complainant reported theft of goat from backyard.",
        "action_taken": "Summoned respondent", "status": "For Hearing",
        "incident_type_primary": "Theft", "batch_number": "batch-2023-05",
    }]
    with app.app_context():
        import_csv_rows(rows, encoded_by="Encoder A")
        r = IncidentRecord.query.one()
        assert r.narrative.startswith("BLOTTER FORM")
        assert "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO : Juan" in r.narrative
        assert "   ORAS  : 14:30" in r.narrative
        assert "E. SALAYSAY\nComplainant reported theft of goat" in r.narrative
        assert (r.date, r.time, r.location_purok, r.status) == ("2023-05-10", "14:30", "Purok 4", "For Hearing")
        assert r.batch_number == "BATCH-2023-05"
        assert r.encoded_by == "Encoder A"           # form default fills the blank column
        assert r.incident_type_primary == "Theft"
        assert r.pnp_classification and r.kp_status and r.category_group


def test_import_reads_encoding_schema_headers(app):
    """data/raw/gold_blotter_final.csv-style headers used to import as empty
    narratives - every row skipped the snake_case lookups."""
    rows = [{
        "Date": "2026-05-05", "Time": "22:59", "Location/Purok": "Purok 2, Don Juico St.",
        "Reporting Party": "Ana", "Respondent": "Ben",
        "Narrative/Summary": "Nawala ang cellphone sa KTV.", "Action Taken": "Referred to police",
        "Status": "Referred", "Incident Type": "Theft", "Include in ML": "FALSE",
    }]
    with app.app_context():
        assert import_csv_rows(rows)["saved"] == 1
        r = IncidentRecord.query.one()
        assert "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO : Ana" in r.narrative
        assert "Nawala ang cellphone sa KTV." in r.narrative
        assert r.location_purok == "Purok 2"
        assert r.incident_type_primary == "Theft"
        assert r.include_in_ml is False


def test_import_twice_does_not_duplicate(app):
    rows = [{"date": "2023-06-01", "incident_summary": "Altercation over a fence.",
             "incident_type_primary": "Theft"}]
    with app.app_context():
        import_csv_rows(rows)
        assert import_csv_rows(rows)["skipped_existing"] == 1
        assert IncidentRecord.query.count() == 1


def test_import_classifies_rows_without_a_type(app, monkeypatch):
    """Same as the New Blotter Record form: no type given, the classifier picks it."""
    monkeypatch.setattr(records_route, "_classify_narrative", lambda text, **kw: {
        "incident_type": "Theft", "confidence": 0.8})
    with app.app_context():
        import_csv_rows([{"incident_summary": "Ninakaw ang manok ni Juan."}])
        r = IncidentRecord.query.one()
        assert r.incident_type_primary == "Theft"
        assert r.model_confidence == 0.8
        assert r.review_status == "accepted"


def test_import_old_schema_incident_type_splits(app):
    """Old-schema 'incident_type' column should auto-split into primary/secondary."""
    rows = [{"date": "2023-06-01", "narrative": "Altercation over a boundary fence dispute.",
             "incident_type": "Physical Injury/Property Dispute"}]
    with app.app_context():
        import_csv_rows(rows)
        r = IncidentRecord.query.first()
        assert r.incident_type_primary is not None
        assert "Altercation over a boundary fence dispute." in r.narrative


def _admin_client(app):
    with app.app_context():
        admin = User(username="admin1", role="admin")
        admin.set_password("pw")
        _db.session.add(admin)
        _db.session.commit()
    client = app.test_client()
    client.post("/login", data={"username": "admin1", "password": "pw"})
    return client


def test_pdf_template_imports_through_the_route(app):
    """The shipped PDF template round-trips: its filled example is saved, its
    two blank entries are skipped for having no narration."""
    client = _admin_client(app)
    with open(PDF_TEMPLATE, "rb") as f:
        resp = client.post("/records/import", data={"file": (io.BytesIO(f.read()), "t.pdf")},
                           content_type="multipart/form-data", follow_redirects=True)
    assert b"Saved 1 blotter record(s). Skipped 2 with no narration of facts." in resp.data
    with app.app_context():
        r = IncidentRecord.query.one()
        assert "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO : Juan Dela Cruz" in r.narrative
        # Form-only fields survive the PDF round trip, including a sub-label
        # (TIRAHAN) that only its section tells apart from B's and the witness's.
        assert "C. REKLAMO : Pagkawala ng manok" in r.narrative
        assert "   TIRAHAN     : Purok 5, Brgy. Anunas" in r.narrative
        assert r.location_purok == "Purok 3"


def test_csv_template_download_matches_import_columns(app):
    client = _admin_client(app)
    body = client.get("/records/import/template.csv").data.decode("utf-8-sig")
    assert body.splitlines()[0] == ",".join(IMPORT_COLUMNS)
    page = client.get("/records/import").data.decode()
    assert all(col in page for col in IMPORT_COLUMNS)
