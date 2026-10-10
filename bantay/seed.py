import secrets

from . import db
from .models import User

# The shared password every account used to be seeded with. Kept only so
# upgrade_schema() can find accounts still on it, and so the change-password
# form can refuse it as a new password.
LEGACY_DEFAULT_PASSWORD = "changeme123"

# One default account per role (encoder / reviewer / admin) so the RBAC
# rollout ships with a working login for each permission tier instead of
# leaving reviewer/encoder accounts to be created by hand with no UI for it.
DEFAULT_USERS = [
    ("admin", "admin"),
    ("reviewer", "reviewer"),
    ("encoder", "encoder"),
]


def seed_default_users():
    """Create any missing default account with its own random one-time password.

    The passwords are printed once, to the console of the first start, and are
    not stored anywhere else. Each account is flagged must_change_password, so
    the first sign-in goes straight to the change-password screen. Returns the
    (username, password) pairs it created.
    """
    created = []
    for username, role in DEFAULT_USERS:
        if User.query.filter_by(username=username).first():
            continue
        password = secrets.token_urlsafe(9)   # 12 characters
        user = User(username=username, role=role, must_change_password=True)
        user.set_password(password)
        db.session.add(user)
        created.append((username, password))
    db.session.commit()
    if created:
        lines = "\n".join(f"    {u:<10} {p}" for u, p in created)
        print("\n" + "=" * 64 +
              "\nBANTAY first start - one-time passwords (shown only this once):\n"
              f"{lines}\n"
              "Each account must set a new password at its first sign-in.\n" +
              "=" * 64 + "\n", flush=True)
    return created


def upgrade_schema():
    """Bring an existing SQLite database up to the current schema. Safe to re-run.

    db.create_all() only creates missing tables (audit_log, on an old database);
    it never alters an existing one, so the new user column is added here. No
    data is rewritten except the must_change_password flag itself.
    """
    if db.engine.dialect.name == "sqlite":
        with db.engine.begin() as con:
            cols = {row[1] for row in con.exec_driver_sql('PRAGMA table_info("user")')}
            if "must_change_password" not in cols:
                con.exec_driver_sql('ALTER TABLE "user" ADD COLUMN '
                                    "must_change_password BOOLEAN NOT NULL DEFAULT 0")
            # Append-only at the database level too, so a bulk query.delete()
            # or a hand-written UPDATE cannot rewrite history either.
            for op in ("UPDATE", "DELETE"):
                con.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS audit_log_no_{op.lower()} "
                    f"BEFORE {op} ON audit_log "
                    "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END")

    # Any account still on the old shared default is forced to change it.
    for user in User.query.filter_by(must_change_password=False).all():
        if user.check_password(LEGACY_DEFAULT_PASSWORD):
            user.must_change_password = True
    db.session.commit()
