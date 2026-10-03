from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from sqlalchemy.orm import validates

from . import db
from .crypto import EncryptedText, narrative_digest as compute_narrative_digest
from .normalize import canonical_location


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="encoder")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class IncidentRecord(db.Model):
    """Structured barangay incident record.

    Schema aligned with Chapter 3 §3.2.4 Data Description table.

    incident_type_primary  — single canonical label; the ML classification target.
    incident_type_secondary — optional overlapping concern (e.g. a dispute that
                              also involves physical injury), recorded separately
                              rather than forcing multi-label on the primary model
                              (§3.3.1 – practical middle ground per Vieira et al., 2025).
    """
    id = db.Column(db.Integer, primary_key=True)

    # --- Traceability (§3.2.4) ---
    batch_number = db.Column(db.String(40))         # groups records by scanning batch
    source_file_page = db.Column(db.String(120))    # reference to original image/page

    # --- Core incident fields ---
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    # location_purok is the canonical place analytics groups on; whatever was
    # typed, scanned or imported is kept verbatim in location_detail. Both are
    # set by _canonicalize_location below - assign location_purok only.
    location_purok = db.Column(db.String(120))
    location_detail = db.Column(EncryptedText("location_detail"))  # often a home address

    # --- Incident type: primary (ML target) + secondary (optional overlap) ---
    incident_type_primary = db.Column(db.String(120))
    incident_type_secondary = db.Column(db.String(120))

    # --- PNP crime-classification tier, derived from incident_type_primary
    # via normalize.get_pnp_classification() (Index Crime / Non-Index Crime /
    # Non-Criminal Complaint). Stored so it can be filtered/aggregated in SQL
    # and factored into per-tier model evaluation, not just rendered ad hoc. ---
    pnp_classification = db.Column(db.String(40))

    # --- Category group, derived from incident_type_primary via
    # normalize.get_category_group() (Civil Dispute / Criminal-Penal Code /
    # Property-Lost Items / Family-Domestic / Community Disturbance /
    # Administrative-Referral / Catch-all). Separate axis from
    # pnp_classification above — this groups by *kind of matter*, not legal
    # tier. ---
    category_group = db.Column(db.String(40))

    # --- Katarungang Pambarangay referability, derived from
    # incident_type_primary via normalize.get_kp_status() (KP-Mediable /
    # Conditional / Excluded). A third independent axis: RA 7160 Sec. 408 turns
    # on the penalty ceiling, not on whether the matter is a crime, so this does
    # NOT track pnp_classification - Slight Physical Injury is an Index Crime
    # and still KP-Mediable. Stored so Lupon workload can be filtered and
    # counted in SQL. ---
    kp_status = db.Column(db.String(20))

    # --- Narrative and response ---
    narrative = db.Column(EncryptedText("narrative"), nullable=False)
    # Keyed digest of the narrative, maintained by _sync_narrative_digest below.
    # AES-GCM uses a fresh nonce per write, so the same narrative encrypts to a
    # different ciphertext every time and `WHERE narrative = ?` no longer matches -
    # this is what exact-match lookups (the duplicate-submission guard in
    # routes/records.py) compare on instead.
    narrative_digest = db.Column(db.String(64), index=True)
    action_taken = db.Column(EncryptedText("action_taken"))
    status = db.Column(db.String(40))

    # --- Privacy / anonymization (§3.2.1) ---
    persons_involved_masked = db.Column(EncryptedText("persons_involved_masked"))  # anonymized ref to parties

    # --- ML quality flags ---
    readability = db.Column(db.String(20), default="readable")
    include_in_ml = db.Column(db.Boolean, default=True)
    model_confidence = db.Column(db.Float)
    review_status = db.Column(db.String(20), default="accepted")

    # --- Encoding provenance (§3.2.4) ---
    encoded_by = db.Column(db.String(120))          # name/ID of encoder
    date_encoded = db.Column(db.String(20))         # date of manual encoding

    # --- Miscellaneous ---
    remarks = db.Column(EncryptedText("remarks"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"))

    @validates("narrative")
    def _sync_narrative_digest(self, key, value):
        """Keep narrative_digest in lockstep with narrative, on every write path.

        Setting the digest at each call site instead would mean three of them
        today (new_record, edit_record, import_csv_rows) and silent breakage the
        first time a fourth is added - a stale digest does not raise, it just
        stops the duplicate guard from ever matching.
        """
        self.narrative_digest = compute_narrative_digest(value) if value else None
        return value

    @validates("location_purok")
    def _canonicalize_location(self, key, value):
        """Store the canonical place, keep the raw text beside it.

        Here rather than at each call site for the same reason as the digest
        above: new_record, edit_record and import_csv_rows all write this
        column, and one that skips normalizing splits a place in two on the
        Analytics page. Re-runnable from location_detail by
        tools/migrate_location.py after an alias is added.
        """
        value = str(value or "").strip()
        self.location_detail = value or None
        return canonical_location(value)

    @classmethod
    def search_narrative(cls, query, needle):
        """Substring search over the encrypted narrative column.

        ponytail: decrypts the already-filtered set in Python, then hands SQL the
        surviving ids so ordering and pagination still happen in SQL. O(n) over the
        corpus - a few hundred to a few thousand rows at barangay scale, which is
        milliseconds. Add a token-level blind index only if it outgrows that.
        """
        needle = needle.lower()
        matches = query.with_entities(cls.id, cls.narrative)
        ids = [rid for rid, text in matches if needle in (text or "").lower()]
        return query.filter(cls.id.in_(ids))
