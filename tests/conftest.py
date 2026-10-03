"""Shared pytest setup.

Encryption has no plaintext fallback (see bantay/crypto.py), so every test that
touches the database needs a key. It is set here, at import time, before any test
module imports the application.

Deliberately a throwaway key, and deliberately unconditional: config.py loads
instance/.env with os.environ.setdefault, so whatever is already in the
environment wins. Setting it first means the suite can never open the live
corpus with the production key, and a test that somehow points at
instance/bantay.db fails loudly instead of quietly decrypting real records.

Fixed rather than random so a failure reproduces on the next run.
"""
import base64
import os

os.environ["BANTAY_ENC_KEY"] = base64.b64encode(bytes(range(32))).decode()
os.environ["SECRET_KEY"] = "test-secret-not-used-outside-the-suite"
