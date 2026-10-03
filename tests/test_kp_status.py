"""Katarungang Pambarangay referability (RA 7160 Sec. 408).

The point of these tests is the independence claim: KP status is NOT a
relabelling of the PNP tier. If someone later "simplifies" get_kp_status into a
lookup on pnp_classification, test_kp_is_not_a_function_of_pnp_tier fails.
"""
from bantay.normalize import (CANONICAL_CATEGORIES, FALLBACK_CATEGORY, KP_STATUS,
                              KP_STATUSES, clean_row, get_kp_status,
                              get_pnp_classification)


def test_every_canonical_category_is_mapped():
    """A category with no KP entry would silently read as Unclassified."""
    missing = [name for name, _ in CANONICAL_CATEGORIES if name not in KP_STATUS]
    assert missing == [], f"unmapped categories: {missing}"


def test_statuses_are_from_the_declared_vocabulary():
    for name, (status, note) in KP_STATUS.items():
        assert status in KP_STATUSES, f"{name} has unknown status {status!r}"
        assert note.strip(), f"{name} has an empty note"


def test_conditional_notes_state_a_test_to_apply():
    """A "Conditional" verdict is useless to an investigator without the rule."""
    for name, (status, note) in KP_STATUS.items():
        if status == "Conditional":
            assert "EXCLUDED" in note or "ceiling" in note or "Sec. 408[d]" in note, \
                f"{name} is Conditional but its note names no test"


def test_kp_is_not_a_function_of_pnp_tier():
    """Same PNP tier, different KP status - the axes are genuinely independent.

    Slight Physical Injury is the load-bearing case: an Index Crime that the
    Lupon may still mediate, because RPC Art. 266 is arresto menor and falls
    within the Sec. 408[c] ceiling.
    """
    assert get_pnp_classification("Slight Physical Injury")[0] == "Index Crime"
    assert get_kp_status("Slight Physical Injury")[0] == "KP-Mediable"

    assert get_pnp_classification("Robbery")[0] == "Index Crime"
    assert get_kp_status("Robbery")[0] == "Excluded"


def test_vawc_is_barred_outright():
    """RA 9262 Sec. 33 is an express prohibition, not a penalty computation."""
    status, note = get_kp_status("RA 9262 (VAWC)")
    assert status == "Excluded"
    assert "9262" in note and "33" in note


def test_non_criminal_tier_still_contains_kp_exclusions():
    """The old conflation this feature exists to fix.

    "Referral to Police/Other Agency" sits in the Non-Criminal Complaint tier
    but is definitionally outside Lupon authority.
    """
    assert get_pnp_classification("Referral to Police/Other Agency")[0] == "Non-Criminal Complaint"
    assert get_kp_status("Referral to Police/Other Agency")[0] == "Excluded"


def test_unknown_and_fallback_categories_are_unclassified():
    assert get_kp_status(FALLBACK_CATEGORY) == ("Unclassified", "—")
    assert get_kp_status(None)[0] == "Unclassified"
    assert get_kp_status("Not A Real Category")[0] == "Unclassified"


def test_clean_row_emits_kp_status():
    row = clean_row({"incident_type_primary": "Neighbor Dispute", "narrative": "x"})
    assert row["kp_status"] == "KP-Mediable"

    row = clean_row({"incident_type_primary": "Theft", "narrative": "x"})
    assert row["kp_status"] == "Conditional"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all KP checks passed")
