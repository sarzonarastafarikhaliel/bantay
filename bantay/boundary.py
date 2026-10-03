"""Point-in-polygon check against a GeoJSON boundary.

Ray casting, dependency-free on purpose: pulling in shapely for one boundary
check costs more than it's worth. Shared by tests/test_purok_coordinates.py
(checks the curated coordinate file) and geocode.py (validates a freshly
looked-up one before it's trusted).
"""
import json


def _rings(boundary_path):
    with open(boundary_path, encoding="utf-8") as fh:
        geom = json.load(fh)["features"][0]["geometry"]
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]      # MultiPolygon


def point_in_boundary(lat, lon, boundary_path):
    for ring in _rings(boundary_path):
        hit, n = False, len(ring)
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            if ((y1 > lat) != (y2 > lat)) and (lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1):
                hit = not hit
        if hit:
            return True
    return False
