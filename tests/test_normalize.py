from bantay.normalize import (
    clean_row,
    fix_mojibake,
    get_category_group,
    get_pnp_classification,
    join_labels,
    normalize_batch_number,
    normalize_date,
    normalize_incident_types,
    normalize_purok,
    normalize_readability,
    normalize_time,
    split_labels,
    strip_citations,
)


def test_strip_citations():
    assert strip_citations("Cash Collection[cite: 19, 20]") == "Cash Collection"
    assert strip_citations("No citation here") == "No citation here"


def test_fix_mojibake_repairs_utf8_read_as_latin1():
    assert fix_mojibake("MalabaÃ±as") == "Malabañas"


def test_fix_mojibake_is_noop_on_already_correct_text():
    assert fix_mojibake("Malabañas") == "Malabañas"


def test_normalize_date_handles_varied_formats():
    assert normalize_date("May-06-2026") == "2026-05-06"
    assert normalize_date("Nov. 15, 2023") == "2023-11-15"
    assert normalize_date("01-17-2026") == "2026-01-17"
    assert normalize_date("November 21, 2023") == "2023-11-21"


def test_normalize_date_handles_missing_values():
    assert normalize_date("Not Indicated") is None
    assert normalize_date("") is None
    assert normalize_date(None) is None


def test_normalize_time_hhmm():
    assert normalize_time("09:30") == "09:30"


def test_normalize_time_12h_pm():
    assert normalize_time("2:30 PM") == "14:30"


def test_normalize_time_12h_am():
    assert normalize_time("12:00 AM") == "00:00"


def test_normalize_time_compact():
    assert normalize_time("0930") == "09:30"


def test_normalize_time_none():
    assert normalize_time(None) is None
    assert normalize_time("") is None


def test_normalize_batch_number():
    assert normalize_batch_number("  batch-001  ") == "BATCH-001"
    assert normalize_batch_number(None) is None
    assert normalize_batch_number("") is None


def test_normalize_purok_extracts_purok_number():
    assert normalize_purok("Purok 3 (Arko Brgy. Anunas) / Purok 2, Brgy. Anunas") == "Purok 3"
    assert normalize_purok("5/0 Purok 4, Brgy. Anunas") == "Purok 4"


def test_normalize_purok_falls_back_to_first_segment():
    assert normalize_purok("Batibot, Brgy. Anunas") == "Batibot"
    assert normalize_purok("") == "Unspecified"


def test_normalize_incident_types_returns_primary_secondary():
    primary, secondary = normalize_incident_types("Robbery (Panloloob)")
    assert primary == "Robbery"
    assert secondary is None


def test_normalize_incident_types_compound_returns_primary_and_secondary():
    primary, secondary = normalize_incident_types("For Blotter (Theft/Property Damage)")
    assert primary == "Theft"
    assert secondary == "Malicious Mischief / Property Damage"


def test_normalize_incident_types_fallback():
    primary, secondary = normalize_incident_types("Blotter")
    assert primary == "Others / Miscellaneous"
    assert secondary is None


def test_split_and_join_labels_round_trip():
    labels = ["Theft", "Physical Injury"]
    joined = join_labels(labels)
    assert joined == "Theft|Physical Injury"
    assert split_labels(joined) == labels


def test_split_labels_handles_empty():
    assert split_labels("") == []
    assert split_labels(None) == []


def test_normalize_readability():
    assert normalize_readability("Legible") == "readable"
    assert normalize_readability("Partially Legible") == "partial"
    assert normalize_readability("Moderately Legible") == "partial"
    assert normalize_readability("Illegible") == "unreadable"


def test_clean_row_new_schema():
    """clean_row should produce incident_type_primary and incident_type_secondary."""
    row = {
        "date": "2023-01-01",
        "location_purok": "Purok 3",
        "incident_type_primary": "Theft",
        "incident_type_secondary": "",
        "narrative": "Bicycle reported stolen.",
        "readability": "readable",
        "batch_number": "batch-001",
    }
    once = clean_row(row)
    assert once["incident_type_primary"] == "Theft"
    assert once["category_group"] == "Criminal/Penal Code"
    assert once["batch_number"] == "BATCH-001"
    twice = clean_row(once)
    assert once == twice


def test_get_pnp_classification_known_and_fallback():
    tier, basis = get_pnp_classification("Theft")
    assert tier == "Index Crime"
    assert "Theft" in basis

    tier, basis = get_pnp_classification("Marital Relation")
    assert tier == "Non-Criminal Complaint"

    tier, basis = get_pnp_classification("Not A Real Category")
    assert (tier, basis) == ("Unclassified", "—")


def test_get_category_group_known_and_fallback():
    assert get_category_group("Collection of Sum of Money") == "Civil Dispute"
    assert get_category_group("Theft") == "Criminal/Penal Code"
    assert get_category_group("Lost Property / Document") == "Property/Lost Items"
    assert get_category_group("RA 9262 (VAWC)") == "Family/Domestic"
    assert get_category_group("Noise Complaint") == "Community Disturbance"
    assert get_category_group("Missing Person Report") == "Administrative/Referral"
    assert get_category_group("Others / Miscellaneous") == "Catch-all"
    assert get_category_group("Not A Real Category") == "Catch-all"


def test_clean_row_old_schema_splits_multilabel():
    """clean_row on old incident_type column should produce primary + secondary."""
    row = {
        "date": "2023-01-01",
        "location_purok": "Purok 2",
        "incident_type": "For Blotter (Theft/Property Damage)",
        "narrative": "Something happened.",
        "readability": "readable",
    }
    cleaned = clean_row(row)
    assert cleaned["incident_type_primary"] == "Theft"
    assert cleaned["incident_type_secondary"] == "Malicious Mischief / Property Damage"


def test_find_purok_number_reads_short_and_roman_forms():
    from bantay.normalize import find_purok_number
    assert find_purok_number("P-4, Brgy. Anunas") == "Purok 4"
    assert find_purok_number("PRK. 3") == "Purok 3"
    assert find_purok_number("PUROK I, MANARANG COMPOUND") == "Purok 1"
    assert find_purok_number("PUROK VILLASOL") is None
    assert find_purok_number("Bayad P500") is None
