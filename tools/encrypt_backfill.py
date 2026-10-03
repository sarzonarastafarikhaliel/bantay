"""encrypt_backfill.py — convert an existing bantay.db to encrypted blotter records.

Run once, after bantay/crypto.py and the EncryptedText columns in bantay/models.py
are in place, to convert rows that were written before encryption existed:

    python tools/encrypt_backfill.py                    # back up, migrate, verify
    python tools/encrypt_backfill.py --dry-run          # report only, write nothing
    python tools/encrypt_backfill.py --db path/to.db

Two jobs, because there is no Alembic here and db.create_all() only ever creates
tables it does not already have - it will not add a column to an existing one:

  1. Schema: add narrative_digest and its index to incident_record.
  2. Data:   encrypt narrative, action_taken, persons_involved_masked and remarks
             in place, and fill in the digest.

Follows the same shape as the earlier Unused/uncertain/migrate_schema.py (argparse
--db, PRAGMA table_info guard, safe to re-run) and adds what encryption needs that
a column migration does not: a backup, a per-row read-back verification, and an
abort that leaves the database untouched if any row fails to decrypt.

Safe to re-run: an already-encrypted value is recognised by its "v1:" prefix and
skipped, so a second run is a no-op and an interrupted run can simply be repeated.

BANTAY_ENC_KEY must be set to the SAME key the application will use. Encrypting
with one key and running the app with another leaves every narrative unreadable,
and there is no way back except the backup this script writes.

Stop the Flask app before running - SQLite will not stop you from writing to a
database another process has open, and a mid-flight write during this rewrite is
the one thing the backup cannot help with.
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402,F401  - importing it loads instance/.env
from bantay.crypto import EncryptedText, is_encrypted, narrative_digest  # noqa: E402

TABLE = "incident_record"
ENCRYPTED_COLUMNS = ("narrative", "action_taken", "persons_involved_masked", "remarks")
DIGEST_COLUMN = "narrative_digest"
# Matches what SQLAlchemy generates for index=True on that column, so a database
# migrated here and one built fresh by db.create_all() end up identical.
INDEX_NAME = f"ix_{TABLE}_{DIGEST_COLUMN}"


def _backup(db_path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = db_path.with_name(f"{db_path.name}.pre-encrypt-{stamp}")
    shutil.copy2(db_path, dest)
    print(f"[encrypt] backup written: {dest}")
    return dest


def _add_schema(cur, existing, dry_run):
    added = False
    if DIGEST_COLUMN in existing:
        print(f"[encrypt] column {DIGEST_COLUMN} already present")
    elif dry_run:
        print(f"[encrypt] would add column {DIGEST_COLUMN} and index {INDEX_NAME}")
        return False, False
    else:
        cur.execute(
            f"ALTER TABLE {TABLE} ADD COLUMN {DIGEST_COLUMN} VARCHAR(64) DEFAULT NULL"
        )
        added = True
        print(f"[encrypt] added column {DIGEST_COLUMN}")
    if not dry_run:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE} ({DIGEST_COLUMN})")
        print(f"[encrypt] index {INDEX_NAME} present")
    return True, added


def _plan_row(row, codecs, has_digest):
    """What this row needs: {column: new_stored_value}, empty if already done."""
    updates = {}
    plain_narrative = None

    for col in ENCRYPTED_COLUMNS:
        value = row[col]
        if value is None:
            continue
        if is_encrypted(value):
            # Already converted - but a run interrupted between the UPDATE and
            # the digest still needs the plaintext to compute one.
            if col == "narrative":
                plain_narrative = codecs[col].process_result_value(value, None)
            continue
        if col == "narrative":
            plain_narrative = value
        updates[col] = codecs[col].process_bind_param(value, None)

    # Mirrors the @validates hook in models.IncidentRecord: an empty narrative
    # gets no digest, so a fresh insert and a migrated row agree.
    current_digest = row[DIGEST_COLUMN] if has_digest else None
    if plain_narrative and current_digest is None:
        updates[DIGEST_COLUMN] = narrative_digest(plain_narrative)

    return updates, plain_narrative


def _verify_row(cur, row_id, codecs, expected):
    """Read the row back and decrypt it, before anything is committed.

    Catches a mangled write while the transaction can still be rolled back,
    rather than after the operator has deleted the backup.
    """
    cols = ", ".join(ENCRYPTED_COLUMNS)
    stored = cur.execute(
        f"SELECT {cols} FROM {TABLE} WHERE id = ?", (row_id,)
    ).fetchone()
    for col in ENCRYPTED_COLUMNS:
        got = codecs[col].process_result_value(stored[col], None)
        if got != expected[col]:
            raise RuntimeError(
                f"row {row_id}, column {col}: does not read back as it went in"
            )


def migrate(db_path, dry_run=False):
    db_path = Path(db_path)
    if not db_path.exists():
        raise SystemExit(f"[encrypt] no such database: {db_path}")

    codecs = {col: EncryptedText(col) for col in ENCRYPTED_COLUMNS}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    existing = {r[1] for r in cur.execute(f"PRAGMA table_info({TABLE})")}
    if not existing:
        raise SystemExit(f"[encrypt] {db_path} has no {TABLE} table")
    missing = [c for c in ENCRYPTED_COLUMNS if c not in existing]
    if missing:
        raise SystemExit(f"[encrypt] {TABLE} is missing columns: {missing}")

    print(f"[encrypt] database: {db_path}")
    if not dry_run:
        _backup(db_path)

    has_digest, schema_added = _add_schema(cur, existing, dry_run)

    select_cols = ["id", *ENCRYPTED_COLUMNS] + ([DIGEST_COLUMN] if has_digest else [])
    rows = cur.execute(f"SELECT {', '.join(select_cols)} FROM {TABLE}").fetchall()

    changed = untouched = 0
    columns_written = 0
    digests_written = 0

    try:
        for row in rows:
            updates, _ = _plan_row(row, codecs, has_digest)
            if not updates:
                untouched += 1
                continue
            changed += 1
            columns_written += sum(1 for c in updates if c != DIGEST_COLUMN)
            digests_written += 1 if DIGEST_COLUMN in updates else 0

            if dry_run:
                print(f"[encrypt] would update row {row['id']}: {sorted(updates)}")
                continue

            assignments = ", ".join(f"{c} = ?" for c in updates)
            cur.execute(
                f"UPDATE {TABLE} SET {assignments} WHERE id = ?",
                [*updates.values(), row["id"]],
            )
            # What the row should decrypt to: whatever it held before, since an
            # already-encrypted column was left alone and reads back unchanged.
            expected = {
                col: codecs[col].process_result_value(row[col], None)
                for col in ENCRYPTED_COLUMNS
            }
            _verify_row(cur, row["id"], codecs, expected)
    except Exception:
        con.rollback()
        con.close()
        # Deliberately does not claim "nothing was changed": sqlite3 runs DDL in
        # autocommit, so the ALTER TABLE above is already on disk and the rollback
        # cannot take it back. Harmless - the column is nullable and the run is
        # idempotent - but saying otherwise sends people looking in the wrong place.
        print("[encrypt] ABORTED - no record data was changed.")
        if schema_added:
            print(f"[encrypt] note: the {DIGEST_COLUMN} column and its index were added "
                  "before the failure and remain (DDL is not covered by the rollback). "
                  "Fixing the cause and re-running completes the migration.")
        raise

    if dry_run:
        con.rollback()
        print(f"[encrypt] dry run: {changed} row(s) would change, {untouched} already done")
    else:
        con.commit()
        print(
            f"[encrypt] {changed} row(s) updated "
            f"({columns_written} column value(s) encrypted, {digests_written} digest(s) set), "
            f"{untouched} already done"
        )
        print("[encrypt] every updated row was read back and decrypted successfully.")

        # An in-place UPDATE does not erase what was there before: SQLite moves the
        # old row to the free list and leaves its bytes in the page. Without this,
        # every narrative this script "encrypted" is still sitting in the file in
        # plaintext, recoverable with a hex editor - which is precisely the attacker
        # the encryption is for. VACUUM rebuilds the file from the live rows only.
        #
        # File-level, not disk-level: the pages the old file occupied may still be
        # readable from the raw device until they are reused. Deleting the .db is
        # not the same as shredding it.
        con.execute("VACUUM")
        print("[encrypt] VACUUM done - freed pages holding the old plaintext are gone")
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Encrypt existing BANTAY blotter records")
    parser.add_argument("--db", default="instance/bantay.db", help="Path to bantay.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing.")
    args = parser.parse_args()
    try:
        migrate(args.db, dry_run=args.dry_run)
    except RuntimeError as exc:
        # Chiefly a missing or wrong BANTAY_ENC_KEY, which carries its own
        # instructions - a traceback would only bury them.
        raise SystemExit(f"[encrypt] {exc}")


if __name__ == "__main__":
    main()
