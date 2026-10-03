"""Fill in missing purok/place coordinates via OpenStreetMap Nominatim.

Purok names extracted from blotters (narrative.py, routes/scan.py) or typed
into the record form have no coordinate until someone adds one to
data/purok_coordinates.json - see that file's _meta for why a raw geocoder
result isn't trusted outright: 6 of the original 12 entries were geocoded
wrong, one 15 km off in the wrong municipality. ensure_coordinates() looks a
missing name up, keeps it only if it lands inside the real barangay boundary
(boundary.py), and caches a miss too so a name Nominatim can't resolve isn't
retried on every page load.
"""
import json
import os
import urllib.parse
import urllib.request

from .boundary import point_in_boundary

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORDS_PATH = os.path.join(_ROOT, "data", "purok_coordinates.json")
BOUNDARY_PATH = os.path.join(_ROOT, "data", "anunas_boundary.geojson")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "BANTAY-Prototype-hotspot-map/1.0 (contact: novembergraduates@gmail.com)"


def _geocode(name, bbox):
    query = f"{name}, Barangay Anunas, Angeles City, Pampanga, Philippines"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "bounded": 1,
        # Nominatim viewbox order is left,top,right,bottom = west,north,east,south.
        "viewbox": f"{bbox['west']},{bbox['north']},{bbox['east']},{bbox['south']}",
    }
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=5) as resp:
        results = json.load(resp)
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def ensure_coordinates(names, coords_path=COORDS_PATH, boundary_path=BOUNDARY_PATH):
    """Geocode any name in `names` that isn't already mapped or already a
    known miss, validate it against the barangay boundary, and persist the
    result either way. Returns the resulting {name: [lat, lon]} map. Never
    raises - a network or API failure just leaves that name unmapped, same
    as a name nobody has surveyed yet.
    """
    try:
        with open(coords_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    puroks = data.setdefault("puroks", {})
    meta = data.setdefault("_meta", {})
    misses = meta.setdefault("geocode_misses", [])
    bbox = meta.get("boundary_extent")
    if not bbox:
        return puroks

    to_try = [n for n in names if n and n not in puroks and n not in misses and n != "Unknown"]
    if not to_try:
        return puroks

    changed = False
    for name in to_try:
        try:
            result = _geocode(name, bbox)
        except OSError:
            continue  # network hiccup - retry next time, don't cache it as a miss
        changed = True
        if result and point_in_boundary(result[0], result[1], boundary_path):
            puroks[name] = [round(result[0], 7), round(result[1], 7)]
        else:
            misses.append(name)

    if changed:
        with open(coords_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    return puroks
