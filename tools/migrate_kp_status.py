"""
migrate_kp_status.py — add and backfill the Katarungang Pambarangay column.

    python tools/migrate_kp_status.py [--db instance/bantay.db] [--dry-run]

db.create_all() only ever CREATEs; it never ALTERs an existing table, and there
is no Alembic here — so an existing bantay.db needs this once. Same shape as
Unused/uncertain/migrate_schema.py (PRAGMA table_info guard, safe to re-run).

Backfills kp_status from incident_type_primary via normalize.get_kp_status(),
so a row migrated here and a row written fresh by routes/records.py agree.
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bantay.normalize import get_kp_status  # noqa: E402

TABLE = "incident_record"
COLUMN = "kp_status"


def migrate(db_path, dry_run=False):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    existing = {row[1] for row in cur.execute(f"PRAGMA table_info({TABLE})")}
    if not existing:
        print(f"[migrate] No {TABLE} table in {db_path} — nothing to do.")
        con.close()
        return

    if COLUMN in existing:
        print(f"[migrate] Column {COLUMN} already present.")
    elif dry_run:
        print(f"[migrate] (dry-run) would add column {COLUMN}")
    else:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(20) DEFAULT NULL")
        print(f"[migrate] Added column: {COLUMN}")

    if COLUMN not in existing and dry_run:
        print("[migrate] (dry-run) column absent, skipping backfill preview counts")
        con.close()
        return

    # Backfill every row whose type is known, including rows an earlier run
    # already touched — get_kp_status is pure, so re-running is a no-op unless
    # the KP_STATUS map itself changed, in which case a refresh is what we want.
    rows = cur.execute(
        f"SELECT id, incident_type_primary FROM {TABLE} WHERE incident_type_primary IS NOT NULL"
    ).fetchall()
    updates = [(get_kp_status(t)[0], rid) for rid, t in rows]

    if dry_run:
        counts = {}
        for status, _ in updates:
            counts[status] = counts.get(status, 0) + 1
        print(f"[migrate] (dry-run) would backfill {len(updates)} rows: {counts}")
    else:
        cur.executemany(f"UPDATE {TABLE} SET {COLUMN} = ? WHERE id = ?", updates)
        con.commit()
        print(f"[migrate] Backfilled {COLUMN} for {len(updates)} rows.")

    con.close()
    print("[migrate] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BANTAY kp_status migration")
    parser.add_argument("--db", default="instance/bantay.db", help="Path to bantay.db")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()
    migrate(args.db, dry_run=args.dry_run)
