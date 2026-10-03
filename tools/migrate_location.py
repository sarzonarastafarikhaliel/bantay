"""
migrate_location.py — add location_detail and canonicalize location_purok.

    python tools/migrate_location.py [--db instance/bantay.db] [--dry-run]

db.create_all() never ALTERs an existing table and there is no Alembic here, so
an existing bantay.db needs this once - before the app starts, or every query
fails on the missing column. Same shape as migrate_kp_status.py.

Then re-derives every row's canonical place from the raw text kept in
location_detail (the first run copies the old location_purok there). Safe and
meant to be re-run: after adding an alias to data/purok_coordinates.json, run it
again and existing records regroup under the new canonical name.
"""
import argparse
import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bantay.normalize import canonical_location  # noqa: E402

TABLE = "incident_record"
COLUMN = "location_detail"


def _report(pairs):
    """pairs: (old location_purok, new location_purok) per row."""
    for (old, new), n in sorted(Counter(pairs).items(), key=lambda kv: str(kv[0])):
        mark = "  " if old == new else "->"
        print(f"  {old!r:45} {mark} {new!r}  ({n})")
    print(f"[migrate] distinct places: {len({o for o, _ in pairs})} before, "
          f"{len({n for _, n in pairs})} after")


def migrate(db_path, dry_run=False):
    con = sqlite3.connect(db_path)
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({TABLE})")}
    if not existing:
        print(f"[migrate] No {TABLE} table in {db_path} — nothing to do.")
        con.close()
        return

    if COLUMN not in existing:
        if dry_run:
            print(f"[migrate] (dry-run) would add column {COLUMN}")
            rows = con.execute(f"SELECT location_purok FROM {TABLE}").fetchall()
            _report([(old, canonical_location(old)) for (old,) in rows])
            con.close()
            return
        con.execute(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} TEXT DEFAULT NULL")
        con.commit()
        print(f"[migrate] Added column: {COLUMN}")
    con.close()

    # location_detail is encrypted, so the backfill goes through the model
    # (and its _canonicalize_location validator) rather than raw SQL.
    from bantay import create_app, db
    from bantay.models import IncidentRecord

    app = create_app(overrides={"SQLALCHEMY_DATABASE_URI": "sqlite:///" + os.path.abspath(db_path)})
    with app.app_context():
        pairs = []
        for record in IncidentRecord.query.all():
            old = record.location_purok
            record.location_purok = record.location_detail or old
            pairs.append((old, record.location_purok))
        _report(pairs)
        if dry_run:
            db.session.rollback()
            print("[migrate] (dry-run) nothing written.")
        else:
            db.session.commit()
            print(f"[migrate] Canonicalized {len(pairs)} rows.")
    print("[migrate] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BANTAY location canonicalization migration")
    parser.add_argument("--db", default="instance/bantay.db", help="Path to bantay.db")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()
    migrate(args.db, dry_run=args.dry_run)
