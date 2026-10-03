"""At-rest encryption of the blotter columns (bantay/crypto.py, §3.2.1 / RA 10173).

The one that matters for a demonstration is
test_database_file_holds_no_plaintext: it writes a record through the app, then
opens instance-style bantay.db with raw sqlite3 and shows the narrative is not
in it, while the same record reads back intact through the ORM.

The primitive-level properties - round trip, nonce randomness, tamper detection,
column binding, digest stability, refusal without a key - are asserted by the
self-check inside bantay/crypto.py. test_crypto_module_self_check runs it rather
than restating it here, so there is one copy of those assertions and it is the
one a developer gets from `python -m bantay.crypto`.
"""
import os
import sqlite3
import subprocess
import sys

import pytest
from cryptography.exceptions import InvalidTag

from bantay import create_app, db
from bantay.crypto import is_encrypted, narrative_digest
from bantay.models import IncidentRecord

NARRATIVE = "NAGSUMBONG SI JUAN DELA CRUZ TUNGKOL SA AWAY KAPITBAHAY SA PUROK 3."
REMARKS = "TUMANGGI ANG RESPONDENT NA PUMIRMA."


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def app(tmp_path, db_file):
    return create_app(overrides={
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file}",
        "TESTING": True,
        "MODEL_DIR": str(tmp_path / "models"),
    })


@pytest.fixture
def record(app):
    """One saved incident, with the app context left open for the test."""
    with app.app_context():
        rec = IncidentRecord(
            narrative=NARRATIVE, remarks=REMARKS,
            date="2026-01-02", location_purok="Purok 3",
            incident_type_primary="Property Claim / Damage",
        )
        db.session.add(rec)
        db.session.commit()
        yield rec
        db.session.remove()
        db.engine.dispose()


def test_crypto_module_self_check():
    """`python -m bantay.crypto` - round trip, random nonce, tamper detection,
    column binding, digest stability, and refusal without a key."""
    done = subprocess.run([sys.executable, "-m", "bantay.crypto"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "self-check OK" in done.stdout


def test_database_file_holds_no_plaintext(record, db_file):
    """Copy the .db off the machine and you get ciphertext."""
    raw = db_file.read_bytes()
    assert NARRATIVE.encode() not in raw
    assert REMARKS.encode() not in raw

    stored = sqlite3.connect(db_file).execute(
        "select narrative, remarks from incident_record").fetchone()
    assert is_encrypted(stored[0]) and is_encrypted(stored[1])

    # ...and the application still reads it back.
    assert db.session.get(IncidentRecord, record.id).narrative == NARRATIVE


def test_filterable_columns_stay_plaintext(record, db_file):
    """The dashboard groups and filters on these in SQL, and none of them is
    personal information - encrypting them would cost a full-table decrypt per
    chart and buy nothing."""
    assert b"Purok 3" in db_file.read_bytes()
    assert IncidentRecord.query.filter(
        IncidentRecord.location_purok == "Purok 3",
        IncidentRecord.date >= "2026-01-01",
    ).count() == 1


def test_digest_tracks_the_narrative_with_no_caller_help(record):
    """models.IncidentRecord keeps it in step on every write path. A stale digest
    does not raise - it silently stops the duplicate-submission guard matching."""
    assert record.narrative_digest == narrative_digest(NARRATIVE)

    record.narrative = "IBANG SALAYSAY NA ITO."
    db.session.commit()
    assert record.narrative_digest == narrative_digest("IBANG SALAYSAY NA ITO.")


def test_digest_finds_an_exact_match(record):
    """What the replay guard in routes/records.py compares on, now that a random
    nonce makes `WHERE narrative = ?` never match."""
    assert IncidentRecord.query.filter(
        IncidentRecord.narrative_digest == narrative_digest(NARRATIVE)).one().id == record.id


def test_search_reaches_inside_the_encrypted_column(record):
    assert [r.id for r in IncidentRecord.search_narrative(
        IncidentRecord.query, "kapitbahay")] == [record.id]
    assert IncidentRecord.search_narrative(IncidentRecord.query, "wala dito").count() == 0
    # The SQL it replaced is now blind, which is why search_narrative exists.
    assert IncidentRecord.query.filter(
        IncidentRecord.narrative.ilike("%kapitbahay%")).count() == 0


def test_wrong_key_raises_rather_than_returning_garbage(record, tmp_path, db_file):
    """Loud failure, not a barangay record quietly rendered as noise."""
    import base64
    good = os.environ["BANTAY_ENC_KEY"]
    os.environ["BANTAY_ENC_KEY"] = base64.b64encode(bytes(range(1, 33))).decode()
    try:
        with pytest.raises(InvalidTag):
            create_app(overrides={
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file}",
                "TESTING": True,
                "MODEL_DIR": str(tmp_path / "models2"),
            })
    finally:
        os.environ["BANTAY_ENC_KEY"] = good


def test_missing_key_refuses_to_start(record, tmp_path, db_file):
    good = os.environ.pop("BANTAY_ENC_KEY")
    try:
        with pytest.raises(RuntimeError, match="BANTAY_ENC_KEY is not set"):
            create_app(overrides={
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file}",
                "TESTING": True,
                "MODEL_DIR": str(tmp_path / "models3"),
            })
    finally:
        os.environ["BANTAY_ENC_KEY"] = good
