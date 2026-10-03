import pytest

from bantay import create_app, db
from bantay.models import User


@pytest.fixture
def app(tmp_path):
    flask_app = create_app(
        overrides={
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
            "TESTING": True,
            "MODEL_DIR": str(tmp_path / "models"),
        }
    )
    with flask_app.app_context():
        admin = User(username="admin", role="admin")
        admin.set_password("adminpass")
        staff = User(username="staff", role="staff")
        staff.set_password("staffpass")
        db.session.add_all([admin, staff])
        db.session.commit()
    yield flask_app


def test_reload_model_requires_admin(app):
    client = app.test_client()
    client.post("/login", data={"username": "staff", "password": "staffpass"})
    resp = client.post("/reload-model", follow_redirects=True)
    assert b"Only admins can reload the model" in resp.data


def test_reload_model_admin_with_no_trained_model(app):
    client = app.test_client()
    client.post("/login", data={"username": "admin", "password": "adminpass"})
    resp = client.post("/reload-model", follow_redirects=True)
    assert b"No trained model found" in resp.data
