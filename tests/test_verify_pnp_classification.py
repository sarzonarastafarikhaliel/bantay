import json

from bantay.ml.verify_pnp_classification import verify


def test_verify_passes_against_the_real_reference_file():
    assert verify() == []


def test_verify_flags_a_deliberate_mismatch(tmp_path):
    bad_reference = {
        "categories": {
            "Theft": {"tier": "Non-Index Crime", "citation": "made up citation"},
        }
    }
    path = tmp_path / "bad_reference.json"
    path.write_text(json.dumps(bad_reference), encoding="utf-8")

    mismatches = verify(str(path))

    fields = {m["field"] for m in mismatches}
    assert "tier" in fields
    assert "citation" in fields
