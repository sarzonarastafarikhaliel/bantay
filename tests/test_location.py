"""One place, one Analytics row: spelling variants of a Lugar must collapse to
the same canonical location_purok. Before this, "Anunas Hall" and "Anunas Brgy.
Hall", or "#2040 Oregon St." and "2040 Oregon St.", counted as separate places.
"""
import json

import pytest

from bantay.models import IncidentRecord
from bantay.normalize import canonical_location, place_names


@pytest.fixture
def gazetteer(tmp_path):
    path = tmp_path / "places.json"
    path.write_text(json.dumps({
        "_meta": {"adjacent_barangays": ["Margot"]},
        "puroks": {"Villasol Subdivision": [15.16, 120.55]},
        "aliases": {"Anunas Hall": "Barangay Hall", "Barangay Office": "Barangay Hall",
                    "Villasol": "Villasol Subdivision", "Brgy. Anunas": None},
    }), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("variants,expected", [
    (["Anunas Hall", "ANUNAS HALL.", "Brgy. Anunas (Barangay office)"], "Barangay Hall"),
    (["4th St. Villasol Subd.", "4-19 COLORADO VILLASOL", "Villasol Subdivision"], "Villasol Subdivision"),
    (["#2040 Oregon St.", "2040 OREGON STREET", "Oregon St"], "Oregon St"),     # unlisted: still one group
    (["Purok 4B", "PRK. 4 B", "P-4B, Manarang"], "Purok 4B"),
    (["Brgy. Margot"], "Outside Anunas"),
    (["Brgy Anunas", "", None, "Unspecified"], None),                            # names no place
])
def test_variants_collapse_to_one_place(gazetteer, variants, expected):
    assert {canonical_location(v, gazetteer) for v in variants} == {expected}


def test_null_alias_does_not_swallow_a_longer_address(gazetteer):
    assert canonical_location("Brgy. Anunas Barangay Office", gazetteer) == "Barangay Hall"


def test_place_names_lists_canonical_targets_only(gazetteer):
    assert place_names(gazetteer) == ["Barangay Hall", "Outside Anunas", "Villasol Subdivision"]


def test_record_keeps_raw_text_and_stores_canonical():
    record = IncidentRecord(location_purok="  Anunas Brgy. Hall ")
    assert record.location_purok == "Barangay Hall"     # live gazetteer alias
    assert record.location_detail == "Anunas Brgy. Hall"
