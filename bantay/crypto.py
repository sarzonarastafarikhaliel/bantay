"""AES-256-GCM at-rest encryption for the sensitive free-text blotter columns.

Blotter narratives carry names, addresses, allegations and family-related
concerns (Chapter 3 §3.2.1; RA 10173 applies), and `instance/bantay.db` is a
single file that walks off a laptop as easily as any other. Encrypting the four
free-text columns means a copied database is ciphertext, while the columns the
dashboard actually filters and groups on (date, purok, incident type, PNP tier)
stay queryable in SQL - a category label is not personal information, and
encrypting it would force a full-table decrypt for every chart.

GCM rather than CBC or plain CTR because the tag makes tampering *detectable*:
a row edited in place by someone with file access fails to decrypt instead of
quietly returning altered text into a barangay record.

Encryption is wired in as a SQLAlchemy TypeDecorator (see EncryptedText below)
rather than encrypt/decrypt calls in the routes. Doing it at the call sites
means finding every one of them forever, and the one that gets missed writes
plaintext without saying anything.
"""
import base64
import binascii
import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import Text, TypeDecorator

# Version tag on every stored value. Nothing rotates keys today, but a v2 branch
# in process_result_value is the whole cost of rotation later, and it is what
# lets the backfill tool tell an encrypted row from a pre-migration one.
_PREFIX = "v1:"

_KEY_ENV = "BANTAY_ENC_KEY"
_NONCE_BYTES = 12   # 96-bit nonce - the size AES-GCM is specified for
_KEY_BYTES = 32     # AES-256


def _key():
    """The AES-256 key, from the environment. No default, no fallback.

    A default key or a silent plaintext fallback is how a system that is
    "encrypted at rest" ends up shipping a plaintext database - the failure is
    invisible until someone opens the file. Refusing to start is louder.
    """
    raw = os.environ.get(_KEY_ENV)
    if not raw:
        raise RuntimeError(
            f"{_KEY_ENV} is not set. Blotter records are encrypted at rest and "
            "there is no plaintext fallback. Generate a key with:\n"
            "  python -c \"import base64, secrets; "
            "print(base64.b64encode(secrets.token_bytes(32)).decode())\"\n"
            "then set it in the environment (or instance/.env, which is gitignored). "
            "Losing this key means losing every encrypted narrative - back it up "
            "somewhere other than this machine."
        )
    try:
        # validate=True so a mistyped key errors here instead of having its
        # stray characters silently discarded into a different 32-byte key.
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"{_KEY_ENV} is not valid base64: {exc}") from exc
    if len(key) != _KEY_BYTES:
        raise RuntimeError(
            f"{_KEY_ENV} must decode to {_KEY_BYTES} bytes for AES-256, got {len(key)}"
        )
    return key


def is_encrypted(value):
    """True if `value` is already in the stored ciphertext form.

    Lets tools/encrypt_backfill.py tell a converted row from a pre-migration one
    without reaching into _PREFIX, and makes a second backfill run a no-op.
    """
    return isinstance(value, str) and value.startswith(_PREFIX)


def narrative_digest(text):
    """Deterministic keyed digest of a narrative, for exact-match lookups.

    GCM uses a fresh nonce per write, so the same narrative encrypts to a
    different ciphertext every time and `WHERE narrative = ?` stops matching -
    which would silently disable the duplicate-submission guard in
    routes/records.py (the one that caught records 6 and 7, 6 ms apart).
    This digest is what that comparison moves to.

    Keyed, and with a subkey derived from the AES key rather than the AES key
    itself, so the two uses stay cryptographically separate. Not a bare SHA-256:
    an unkeyed digest of a blotter narrative is guessable by anyone who can
    approximate the wording.
    """
    subkey = hashlib.blake2b(key=_key(), person=b"bantay-dig", digest_size=32).digest()
    return hmac.new(subkey, text.strip().lower().encode(), hashlib.sha256).hexdigest()


class EncryptedText(TypeDecorator):
    """AES-256-GCM on the way into the database, plaintext on the way out.

    Stored form: "v1:" + base64(nonce[12] || ciphertext || tag[16]).

    The column name is passed in as the GCM additional authenticated data, so a
    ciphertext lifted out of `narrative` will not decrypt into `remarks` - the
    tag check fails. Binding to the row id would also block swapping values
    between rows, but the id does not exist yet at INSERT time.
    ponytail: column-level binding only; add row-id AAD via an after-insert
    re-encrypt if cross-row swapping ever matters.

    Usage:  narrative = db.Column(EncryptedText("narrative"), nullable=False)
    """
    impl = Text
    cache_ok = True

    def __init__(self, aad, **kwargs):
        super().__init__(**kwargs)
        # Kept as the str it was passed as: SQLAlchemy builds this type's
        # compiled-statement cache key from the constructor argument names, and
        # expects to find each one as an attribute of the same name.
        self.aad = aad

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        nonce = os.urandom(_NONCE_BYTES)
        sealed = AESGCM(_key()).encrypt(nonce, value.encode(), self.aad.encode())
        return _PREFIX + base64.b64encode(nonce + sealed).decode()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if not value.startswith(_PREFIX):
            # A row written before the migration. Readable, but not yet
            # protected - tools/encrypt_backfill.py converts these in place.
            return value
        blob = base64.b64decode(value[len(_PREFIX):])
        nonce, sealed = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
        return AESGCM(_key()).decrypt(nonce, sealed, self.aad.encode()).decode()


if __name__ == "__main__":
    # Self-check: python -m bantay.crypto
    # Runs without a database - this is the crypto layer on its own.
    import secrets
    from cryptography.exceptions import InvalidTag

    os.environ[_KEY_ENV] = base64.b64encode(secrets.token_bytes(32)).decode()
    col = EncryptedText("narrative")
    plain = "NAGSUMBONG SI JUAN DELA CRUZ TUNGKOL SA AWAY KAPITBAHAY."

    stored = col.process_bind_param(plain, None)
    assert stored.startswith(_PREFIX)
    assert plain not in stored, "plaintext leaked into the stored value"
    assert col.process_result_value(stored, None) == plain

    assert col.process_bind_param(plain, None) != stored, "nonce is not random"
    assert col.process_bind_param(None, None) is None
    assert col.process_result_value(None, None) is None
    assert col.process_result_value("legacy plaintext row", None) == "legacy plaintext row"

    # Tampering is detected, not silently returned.
    blob = bytearray(base64.b64decode(stored[len(_PREFIX):]))
    blob[-1] ^= 0x01
    tampered = _PREFIX + base64.b64encode(bytes(blob)).decode()
    try:
        col.process_result_value(tampered, None)
        raise AssertionError("tampered ciphertext decrypted")
    except InvalidTag:
        pass

    # A narrative ciphertext will not decrypt as a remarks value.
    try:
        EncryptedText("remarks").process_result_value(stored, None)
        raise AssertionError("AAD did not bind the value to its column")
    except InvalidTag:
        pass

    # Digest is stable for the same text, different for different text, and
    # insensitive to the whitespace/case noise that CSV imports carry.
    assert narrative_digest(plain) == narrative_digest("  " + plain.lower() + "\n")
    assert narrative_digest(plain) != narrative_digest(plain + " X")

    del os.environ[_KEY_ENV]
    try:
        _key()
        raise AssertionError("missing key did not refuse")
    except RuntimeError:
        pass

    print("bantay.crypto self-check OK")
