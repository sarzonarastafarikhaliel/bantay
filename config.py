import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_instance_env():
    """Read instance/.env into the environment, if it exists.

    BANTAY_ENC_KEY has no default and the app refuses to start without it (see
    bantay/crypto.py), so it has to come from somewhere on every run. Five lines
    of stdlib here instead of a python-dotenv dependency for the same job.

    A real environment variable always wins over the file, so CI, a service
    manager, or a one-off `set BANTAY_ENC_KEY=...` still overrides it.

    instance/ is gitignored, which keeps the key out of the repository - but note
    it then sits in the same folder as bantay.db, so anyone who copies the whole
    folder has both. What this protects is a copied *database*: a backup, an
    emailed .db, a stray commit. Not a copied machine.
    """
    path = os.path.join(BASE_DIR, "instance", ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def _secret_key():
    """Flask's session-signing key. No default, deliberately.

    The old fallback, "dev-secret-change-me", is in this repository's history:
    anyone holding it can forge a session cookie for any account, admin included,
    which makes @login_required and the RBAC checks in bantay/rbac.py decorative.
    A default that works is a default that ships.
    """
    value = os.environ.get("SECRET_KEY")
    if not value:
        raise RuntimeError(
            "SECRET_KEY is not set. It signs the session cookie - with a value "
            "anyone can look up, anyone can forge a login. Generate one with:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(48))"\n'
            "then add it to instance/.env (gitignored) as SECRET_KEY=..."
        )
    return value


_load_instance_env()


class Config:
    SECRET_KEY = _secret_key()
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "instance", "bantay.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.6))
    # Phone-camera blotter photos run up to 6.4 MB in the real corpus (55/104
    # of data/raw/Blotter Pics exceed the old 5 MB limit) - 16 MB gives
    # headroom above the observed max without opening the door to arbitrarily
    # large uploads.
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
