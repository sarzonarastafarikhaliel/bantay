from . import db
from .models import User


def seed_admin(username="admin", password="changeme123"):
    if User.query.filter_by(username=username).first():
        return
    admin = User(username=username, role="admin")
    admin.set_password(password)
    db.session.add(admin)
    db.session.commit()


# One default account per role (encoder / reviewer / admin) so the RBAC
# rollout ships with a working login for each permission tier instead of
# leaving reviewer/encoder accounts to be created by hand with no UI for it.
DEFAULT_USERS = [
    ("admin", "changeme123", "admin"),
    ("reviewer", "changeme123", "reviewer"),
    ("encoder", "changeme123", "encoder"),
]


def seed_default_users():
    for username, password, role in DEFAULT_USERS:
        if User.query.filter_by(username=username).first():
            continue
        user = User(username=username, role=role)
        user.set_password(password)
        db.session.add(user)
    db.session.commit()
