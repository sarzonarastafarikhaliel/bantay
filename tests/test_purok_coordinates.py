"""Every coordinate in data/purok_coordinates.json must fall inside Barangay
Anunas.

This exists because four of the file's original ten entries did not: Purok 3
sat ~15 km away in the Sapangbato/Porac uplands, Puroks 6 and 7 sat inside
Barangay Amsic, and Mountain View ~4 km east near Balibago. Nothing caught it -
the hotspot map (routes/analytics.py) plots whatever the file says, so those
incidents were being mapped into the wrong barangay entirely, and the error is
invisible unless you know where Anunas actually ends.

A point-in-polygon check at request time would have hidden the bad data instead
of fixing it, and charged every page view for the privilege. This is the same
check run once, against the boundary in data/anunas_boundary.geojson
(OpenStreetMap relation 21180548, ODbL), so a bad coordinate fails the suite
rather than reaching a map.
"""
import json
import os

import pytest

from bantay.boundary import point_in_boundary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORDS = os.path.join(ROOT, "data", "purok_coordinates.json")
BOUNDARY = os.path.join(ROOT, "data", "anunas_boundary.geojson")


def _inside(lat, lon):
    """Point-in-polygon check, shared with the geocode-and-validate runtime
    path (bantay/geocode.py) via bantay/boundary.py so both enforce the
    identical boundary."""
    return point_in_boundary(lat, lon, BOUNDARY)


def _load():
    with open(COORDS, encoding="utf-8") as fh:
        return json.load(fh)


def test_every_purok_coordinate_is_inside_barangay_anunas():
    outside = [(name, latlon) for name, latlon in _load()["puroks"].items()
               if not _inside(latlon[0], latlon[1])]
    assert not outside, (
        "These coordinates fall outside Barangay Anunas and would plot in the "
        f"wrong barangay on the hotspot map: {outside}. Do not guess a "
        "replacement - no purok-boundary dataset for Anunas is published. "
        "Either survey the real location or leave the entry out; a place with "
        "no coordinate still appears in the bar-chart ranking.")


def test_default_map_center_is_inside_barangay_anunas():
    """The map opens here. The previous value was an Angeles City placeholder."""
    lat, lon = _load()["_meta"]["default_center"]
    assert _inside(lat, lon), (
        f"default_center {[lat, lon]} is outside the barangay - the hotspot map "
        "would open looking at somewhere else.")


@pytest.mark.parametrize("name,latlon", sorted(_load()["puroks"].items()))
def test_coordinate_is_well_formed(name, latlon):
    """[lat, lon] order, Philippine range. Swapped lat/lon is the classic way a
    coordinate ends up in the Pacific, and the polygon check above would catch
    it only by accident."""
    assert len(latlon) == 2, f"{name}: expected [lat, lon]"
    lat, lon = latlon
    assert 4.0 <= lat <= 21.5, f"{name}: latitude {lat} is outside the Philippines"
    assert 116.0 <= lon <= 127.0, f"{name}: longitude {lon} is outside the Philippines"
