"""Review item R7: security defaults.

Debug off unless BANTAY_DEBUG=1, forced password change on seeded/legacy
accounts, the change-password rules, the in-place upgrade of an existing
database, and an append-only audit log that never carries narrative or name
text.
"""
import io
import os
import sqlite3
import subprocess
import sys

import pytest
from werkzeug.security import generate_password_hash

from bantay import create_app, db
from bantay.ml import sealion_classify
from bantay.models import AuditLog, IncidentRecord, User
from bantay.seed import LEGACY_DEFAULT_PASSWORD, seed_default_users

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Distinctive strings that must never reach the audit log.
MARKER = "ZQXJMARKER"
MARKER_NAME = "Kalabaw-Dimaculangan"
MARKER_NARRATIVE = (f"COMPLAINANT: Juanito {MARKER_NAME}\nRESPONDENT: Pedro {MARKER}\n"
                    f"Nagsumbong si Juanito {MARKER_NAME} tungkol sa {MARKER} na away.")


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    # Saving a record classifies it; keep these tests off the network.
    monkeypatch.setattr(sealion_classify, "available", lambda: False)


def _make_app(tmp_path, name="test.db"):
    return create_app(overrides={
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / name}",
        "TESTING": True,
        "MODEL_DIR": str(tmp_path / "models"),
    })


@pytest.fixture
def app(tmp_path):
    flask_app = _make_app(tmp_path)
    with flask_app.app_context():
        for username, role, flagged in [("boss", "admin", False), ("rev", "reviewer", False),
                                        ("enc", "encoder", False), ("fresh", "encoder", True)]:
            u = User(username=username, role=role, must_change_password=flagged)
            u.set_password(f"{username}-password")
            db.session.add(u)
        db.session.add(IncidentRecord(narrative=MARKER_NARRATIVE, location_purok="Purok 3",
                                      incident_type_primary="Physical Injury",
                                      review_status="needs_review"))
        db.session.commit()
    yield flask_app


def _login(app, username, password=None):
    c = app.test_client()
    c.post("/login", data={"username": username, "password": password or f"{username}-password"})
    return c


def _audit_rows(app):
    with app.app_context():
        return AuditLog.query.order_by(AuditLog.id).all()


def _csrf(client):
    client.get("/change-password")
    with client.session_transaction() as s:
        return s["csrf_token"]


# --- 1. Debug off by default -------------------------------------------------

def test_debug_is_off_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("BANTAY_DEBUG", raising=False)
    monkeypatch.setenv("FLASK_DEBUG", "1")   # Flask's own switch must not turn it on
    assert _make_app(tmp_path).debug is False


def test_debug_is_on_with_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("BANTAY_DEBUG", "1")
    assert _make_app(tmp_path).debug is True


@pytest.mark.parametrize("env_value, expected", [(None, "False False"), ("1", "True True")])
def test_app_py_entry_point_debugger_and_reloader(tmp_path, env_value, expected):
    """Runs app.py as `python app.py` would, with the server call stubbed out,
    and reads what it would have passed to Werkzeug."""
    env = {k: v for k, v in os.environ.items() if k not in ("BANTAY_DEBUG", "FLASK_DEBUG")}
    env["DATABASE_URL"] = f"sqlite:///{tmp_path / 'entry.db'}"
    if env_value:
        env["BANTAY_DEBUG"] = env_value
    script = (
        "import runpy, werkzeug.serving as s\n"
        "s.run_simple = lambda host, port, app, **kw: print('RUN', kw['use_debugger'], kw['use_reloader'])\n"
        "runpy.run_path('app.py', run_name='__main__')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert f"RUN {expected}" in out.stdout


# --- 2. Seeding and the forced change ---------------------------------------

def test_seeded_accounts_get_random_one_time_passwords(tmp_path, capsys):
    app = _make_app(tmp_path)
    with app.app_context():
        created = seed_default_users()
        assert [u for u, _ in created] == ["admin", "reviewer", "encoder"]
        passwords = [p for _, p in created]
        assert len(set(passwords)) == 3
        assert LEGACY_DEFAULT_PASSWORD not in passwords
        for username, password in created:
            user = User.query.filter_by(username=username).one()
            assert user.must_change_password is True
            assert user.check_password(password)
        assert seed_default_users() == []   # second start: nothing new, nothing printed again
    printed = capsys.readouterr().out
    assert all(p in printed for p in passwords)
    assert printed.count(passwords[0]) == 1


def _all_route_urls(app):
    """Every registered URL rule and method, with record_id=1 filled in."""
    for rule in app.url_map.iter_rules():
        if rule.endpoint in ("static", "auth.change_password", "auth.logout"):
            continue
        url = rule.rule.replace("<int:record_id>", "1")
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            yield rule.endpoint, method, url


def test_flagged_user_is_sent_to_change_password_from_every_route(app):
    c = _login(app, "fresh")
    urls = list(_all_route_urls(app))
    assert len(urls) > 20   # the whole app, not a sample
    for endpoint, method, url in urls:
        resp = c.open(url, method=method, json={} if url.endswith("/predict") else None)
        assert resp.status_code == 302, (endpoint, method, resp.status_code)
        assert resp.headers["Location"].endswith("/change-password"), (endpoint, method)
    # None of those requests reached its view: the record survived delete-all.
    with app.app_context():
        assert IncidentRecord.query.count() == 1
    assert c.get("/change-password").status_code == 200
    assert c.get("/static/css/style.css").status_code == 200
    assert c.get("/logout").headers["Location"].endswith("/login")


def test_unflagged_user_is_not_redirected(app):
    assert _login(app, "enc").get("/records/").status_code == 200


# --- 3. Change-password rules ------------------------------------------------

def _change(client, current, new, confirm=None, token=None):
    return client.post("/change-password", data={
        "csrf_token": token if token is not None else _csrf(client),
        "current_password": current, "new_password": new,
        "confirm_password": new if confirm is None else confirm,
    })


@pytest.mark.parametrize("current, new, confirm, message", [
    ("fresh-password", "short9chr", None, "at least 10 characters"),
    ("fresh-password", "fresh-password", None, "different from the current one"),
    ("wrong-password", "a-brand-new-password", None, "Current password is incorrect"),
    ("fresh-password", "a-brand-new-password", "something-else-1", "do not match"),
    ("fresh-password", LEGACY_DEFAULT_PASSWORD, None, "old shared default"),
])
def test_change_password_rejections_keep_the_flag(app, current, new, confirm, message):
    c = _login(app, "fresh")
    resp = _change(c, current, new, confirm)
    assert resp.headers["Location"].endswith("/change-password")
    assert message in c.get("/change-password").get_data(as_text=True)
    with app.app_context():
        user = User.query.filter_by(username="fresh").one()
        assert user.must_change_password is True
        assert user.check_password("fresh-password")


def test_change_password_requires_the_csrf_token(app):
    c = _login(app, "fresh")
    _csrf(c)
    assert _change(c, "fresh-password", "a-brand-new-password", token="forged").status_code == 400
    assert _change(c, "fresh-password", "a-brand-new-password", token="").status_code == 400


def test_change_password_success_clears_the_flag_and_rotates_the_session(app):
    c = _login(app, "fresh")
    old_token = _csrf(c)
    resp = _change(c, "fresh-password", "a-brand-new-password", token=old_token)
    assert resp.headers["Location"].endswith("/")
    with c.session_transaction() as s:
        assert s.get("csrf_token") != old_token   # old session contents dropped
    assert c.get("/records/").status_code == 200   # no longer redirected
    with app.app_context():
        user = User.query.filter_by(username="fresh").one()
        assert user.must_change_password is False
        assert user.check_password("a-brand-new-password")
        # Same hashing scheme as everywhere else (werkzeug's default).
        assert user.password_hash.split("$")[0] == generate_password_hash("x").split("$")[0]
    assert _login(app, "fresh", "a-brand-new-password").get("/records/").status_code == 200
    assert _login(app, "fresh").get("/records/").status_code == 302   # old password refused


# --- Existing-database upgrade -------------------------------------------------

def test_existing_database_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "user" (id INTEGER PRIMARY KEY, username VARCHAR(80) UNIQUE NOT NULL, '
                "password_hash VARCHAR(255) NOT NULL, role VARCHAR(20))")
    old_hash = generate_password_hash(LEGACY_DEFAULT_PASSWORD)
    rows = [(1, "admin", old_hash, "admin"),
            (2, "kag.santos", generate_password_hash("already-changed-1"), "encoder")]
    con.executemany('INSERT INTO "user" VALUES (?, ?, ?, ?)', rows)
    con.commit()
    con.close()

    for _ in range(2):   # and again: re-running the upgrade is a no-op
        app = _make_app(tmp_path, "old.db")
        with app.app_context():
            users = {u.username: u for u in User.query.all()}
            assert set(users) == {"admin", "kag.santos"}
            assert users["admin"].password_hash == old_hash   # data untouched
            assert users["admin"].must_change_password is True
            assert users["kag.santos"].must_change_password is False
            assert AuditLog.query.count() == 0   # table created
    c = app.test_client()
    c.post("/login", data={"username": "admin", "password": LEGACY_DEFAULT_PASSWORD})
    assert c.get("/").headers["Location"].endswith("/change-password")


# --- 4. Audit log ----------------------------------------------------------------

def _csv_upload():
    return io.BytesIO(("date,location_purok,narrative\n"
                       f"2026-01-05,Purok 2,Ninakaw ang manok ni {MARKER_NAME}.\n").encode())


def _run_every_audited_action(app):
    boss = _login(app, "boss")
    _login(app, "boss", "wrong")
    _login(app, MARKER_NAME, "whatever")
    rev = _login(app, "rev")
    enc = _login(app, "enc")

    boss.get("/records/1")
    boss.get("/records/1/export.docx")
    boss.get(f"/records/export.docx?q={MARKER}")
    boss.post("/records/new", data={"narrative": f"Bagong ulat {MARKER}",
                                    "incident_type_primary": "Theft"})
    rev.post("/records/1/edit", data={"narrative": MARKER_NARRATIVE + " edited"})
    rev.post("/review/1", data={"incident_type_primary": "Physical Injury"})
    boss.post("/records/1/send-to-review")
    rev.post("/records/import", data={"file": (_csv_upload(), "batch.csv")},
             content_type="multipart/form-data")
    enc.get("/records/1/edit")                 # role_required denial
    enc.post("/records/delete/1")              # inline admin-check denial
    enc.post("/reload-model")
    assert enc.get("/audit").status_code == 403
    boss.post("/records/delete/2")
    boss.post("/records/delete-all", data={"confirm": "DELETE ALL"})
    _change(_login(app, "fresh"), "fresh-password", "a-brand-new-password")
    boss.get("/logout")


def test_audit_entries_for_every_listed_action(app):
    _run_every_audited_action(app)
    got = [(r.username, r.action, r.target_type, r.target_id, r.outcome) for r in _audit_rows(app)]
    expected = [
        ("boss", "login", None, None, "success"),
        ("boss", "login", None, None, "failure"),
        (None, "login", None, None, "failure"),       # unknown name: not recorded
        ("boss", "record.view", "record", 1, "success"),
        ("boss", "record.export", "record", 1, "success"),
        ("boss", "record.export_bulk", None, None, "success"),
        ("boss", "record.create", "record", 2, "success"),
        ("rev", "record.edit", "record", 1, "success"),
        ("rev", "record.review", "record", 1, "success"),
        ("boss", "record.send_to_review", "record", 1, "success"),
        ("rev", "record.import", None, None, "success"),
        ("enc", "records.edit_record", "record", 1, "denied"),
        ("enc", "records.delete_record", "record", 1, "denied"),
        ("enc", "dashboard.reload_model", None, None, "denied"),
        ("enc", "auth.audit_log", None, None, "denied"),
        ("boss", "record.delete", "record", 2, "success"),
        ("boss", "record.delete_all", None, None, "success"),
        ("fresh", "password_change", None, None, "success"),
        ("boss", "logout", None, None, "success"),
    ]
    # The rev/enc/fresh sign-ins are interleaved above; compare without them.
    assert [g for g in got if not (g[1] == "login" and g[0] in ("rev", "enc", "fresh"))] == expected
    roles = {"boss": "admin", "rev": "reviewer", "enc": "encoder", "fresh": "encoder", None: None}
    for row in _audit_rows(app):
        assert row.timestamp is not None and row.ip == "127.0.0.1"
        assert row.role == roles[row.username]


def test_audit_log_never_contains_narrative_or_name_text(app, tmp_path):
    _run_every_audited_action(app)
    _login(app, MARKER, MARKER)
    con = sqlite3.connect(tmp_path / "test.db")
    rows = con.execute("SELECT * FROM audit_log").fetchall()
    con.close()
    assert len(rows) > 15
    dump = repr(rows)
    for needle in (MARKER, MARKER_NAME, "Juanito", "Pedro", "Nagsumbong", "manok"):
        assert needle not in dump


def test_audit_page_is_admin_only_and_filters(app):
    assert _login(app, "rev").get("/audit").status_code == 403
    assert _login(app, "enc").get("/audit").status_code == 403
    boss = _login(app, "boss")
    boss.get("/records/1")
    page = boss.get("/audit").get_data(as_text=True)
    assert "<td>record.view</td>" in page and "<td>auth.audit_log</td>" in page
    only_views = boss.get("/audit?action=record.view").get_data(as_text=True)
    assert "<td>record.view</td>" in only_views and "<td>auth.audit_log</td>" not in only_views
    assert "<td>rev</td>" not in boss.get("/audit?user=boss").get_data(as_text=True)
    assert "No entries match" in boss.get("/audit?date=2001-01-01").get_data(as_text=True)


def test_audit_log_is_append_only(app):
    _login(app, "boss")
    with app.app_context():
        for sql in ("UPDATE audit_log SET outcome = 'success'", "DELETE FROM audit_log"):
            with pytest.raises(Exception, match="append-only"):
                db.session.execute(db.text(sql))
            db.session.rollback()
        assert AuditLog.query.count() == 1
    # The only audit route is the read-only page.
    audit_routes = [(r.rule, sorted(r.methods - {"HEAD", "OPTIONS"}))
                    for r in app.url_map.iter_rules() if "audit" in r.rule]
    assert audit_routes == [("/audit", ["GET"])]
