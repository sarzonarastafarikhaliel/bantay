"""Self-check for geocode.ensure_coordinates: a looked-up coordinate must
pass the barangay boundary check before it's trusted, a miss must be cached
so it isn't retried, and a name already known must never trigger a lookup.
"""
import json

import pytest

from bantay import geocode

BOUNDARY = geocode.BOUNDARY_PATH  # real data/anunas_boundary.geojson - read-only here

BASE_META = {
    "boundary_extent": {"north": 15.166041, "south": 15.148995, "east": 120.567464, "west": 120.523497},
}


def _write_coords(path, puroks=None, misses=None):
    meta = dict(BASE_META)
    if misses is not None:
        meta["geocode_misses"] = misses
    path.write_text(json.dumps({"_meta": meta, "puroks": puroks or {}}), encoding="utf-8")


def test_valid_result_is_added_and_persisted(tmp_path, monkeypatch):
    coords_path = tmp_path / "coords.json"
    _write_coords(coords_path)
    monkeypatch.setattr(geocode, "_geocode", lambda name, bbox: (15.16, 120.55))  # inside Anunas

    result = geocode.ensure_coordinates(["Purok 8"], coords_path=str(coords_path), boundary_path=BOUNDARY)

    assert result["Purok 8"] == [15.16, 120.55]
    on_disk = json.loads(coords_path.read_text(encoding="utf-8"))
    assert on_disk["puroks"]["Purok 8"] == [15.16, 120.55]


def test_result_outside_boundary_is_rejected_and_cached_as_miss(tmp_path, monkeypatch):
    coords_path = tmp_path / "coords.json"
    _write_coords(coords_path)
    monkeypatch.setattr(geocode, "_geocode", lambda name, bbox: (15.27, 120.44))  # ~15km away, real past bug

    result = geocode.ensure_coordinates(["Purok 3"], coords_path=str(coords_path), boundary_path=BOUNDARY)

    assert "Purok 3" not in result
    on_disk = json.loads(coords_path.read_text(encoding="utf-8"))
    assert "Purok 3" not in on_disk["puroks"]
    assert "Purok 3" in on_disk["_meta"]["geocode_misses"]


def test_already_known_or_missed_name_is_never_looked_up(tmp_path, monkeypatch):
    coords_path = tmp_path / "coords.json"
    _write_coords(coords_path, puroks={"Purok 1": [15.16, 120.55]}, misses=["Purok 3"])
    calls = []
    monkeypatch.setattr(geocode, "_geocode", lambda name, bbox: calls.append(name) or None)

    geocode.ensure_coordinates(["Purok 1", "Purok 3", "Unknown"],
                                coords_path=str(coords_path), boundary_path=BOUNDARY)

    assert calls == []


def test_no_result_is_cached_as_a_miss_not_retried_forever(tmp_path, monkeypatch):
    coords_path = tmp_path / "coords.json"
    _write_coords(coords_path)
    monkeypatch.setattr(geocode, "_geocode", lambda name, bbox: None)  # Nominatim found nothing

    geocode.ensure_coordinates(["Sitio Wala"], coords_path=str(coords_path), boundary_path=BOUNDARY)

    on_disk = json.loads(coords_path.read_text(encoding="utf-8"))
    assert "Sitio Wala" in on_disk["_meta"]["geocode_misses"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
