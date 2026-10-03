"""
Analytics route — advanced analytics views per Chapter 3 §3.1.1 Phase 7.

Provides:
  - Cross-tabulation tables (category × purok, category × month)
  - Trend-based risk indicator summary
  - Hotspot analysis by purok
"""
import json
import os
from collections import defaultdict

from flask import Blueprint, render_template
from flask_login import login_required

from ..geocode import ensure_coordinates
from ..models import IncidentRecord

analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")

# data/purok_coordinates.json — same "data/<file>.json" convention as
# data/pnp_legal_reference.json (see bantay/ml/verify_pnp_classification.py).
PUROK_COORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "purok_coordinates.json",
)
DEFAULT_MAP_CENTER = [15.147181, 120.584733]  # Angeles City, Pampanga — placeholder


def _load_purok_coordinates():
    try:
        with open(PUROK_COORDS_PATH) as f:
            data = json.load(f)
        return data.get("puroks", {}), data.get("_meta", {}).get("default_center", DEFAULT_MAP_CENTER)
    except (OSError, json.JSONDecodeError):
        return {}, DEFAULT_MAP_CENTER


def _cross_tab_data(records):
    cat_purok = defaultdict(lambda: defaultdict(int))
    cat_month = defaultdict(lambda: defaultdict(int))
    puroks = set()
    months = set()
    for r in records:
        cat = r.incident_type_primary or "Uncategorized"
        purok = r.location_purok or "Unknown"
        month = (r.date or "")[:7]
        cat_purok[cat][purok] += 1
        puroks.add(purok)
        if month:
            cat_month[cat][month] += 1
            months.add(month)
    return {
        "cat_purok": dict(cat_purok),
        "cat_month": dict(cat_month),
        "puroks": sorted(puroks),
        "months": sorted(months),
        "categories": sorted(set(r.incident_type_primary for r in records if r.incident_type_primary)),
    }


def _hotspot_data(records, coords=None):
    """Rank puroks by incident concentration (§ Hotspot Analysis).

    Severity tiers are relative to the busiest purok in the current data
    (>=66% of max = high, >=33% = medium) rather than fixed counts, so the
    ranking stays meaningful regardless of total record volume.

    `coords` (purok name -> [lat, lon]) is optional; puroks without a known
    coordinate get lat/lon = None and are simply left off the map, not
    plotted at a guessed location.
    """
    coords = coords or {}
    counts = defaultdict(int)
    cat_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        purok = r.location_purok or "Unknown"
        counts[purok] += 1
        if r.incident_type_primary:
            cat_counts[purok][r.incident_type_primary] += 1

    if not counts:
        return []

    max_count = max(counts.values())
    hotspots = []
    for purok, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        ratio = count / max_count
        severity = "high" if ratio >= 0.66 else ("medium" if ratio >= 0.33 else "low")
        top_category = max(cat_counts[purok].items(), key=lambda kv: kv[1])[0] if cat_counts[purok] else "—"
        latlon = coords.get(purok)
        hotspots.append({
            "purok": purok,
            "count": count,
            "pct": round(ratio * 100, 1),
            "severity": severity,
            "top_category": top_category,
            "lat": latlon[0] if latlon else None,
            "lon": latlon[1] if latlon else None,
        })
    return hotspots


def _risk_indicators(records):
    from collections import Counter
    monthly_cat = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.date and r.incident_type_primary:
            monthly_cat[r.date[:7]][r.incident_type_primary] += 1
    months = sorted(monthly_cat.keys())
    if len(months) < 2:
        return {}
    last, prev = months[-1], months[-2]
    cats = {r.incident_type_primary for r in records if r.incident_type_primary}
    result = {}
    for cat in cats:
        l, p = monthly_cat[last].get(cat, 0), monthly_cat[prev].get(cat, 0)
        result[cat] = {
            "status": "rising" if l > p else ("declining" if l < p else "stable"),
            "last": l,
            "prev": p,
            "delta": l - p,
            "last_month": last,
            "prev_month": prev,
        }
    return result


@analytics_bp.route("/")
@login_required
def index():
    records = IncidentRecord.query.all()
    cross_tabs = _cross_tab_data(records)
    risk_indicators = _risk_indicators(records)
    purok_coords, map_center = _load_purok_coordinates()
    # Any purok/place named on a record but missing from the coordinate file
    # gets one geocode attempt here, validated against the barangay boundary
    # and cached to disk either way - see geocode.py for why a raw lookup
    # isn't trusted without that check.
    purok_names = {r.location_purok for r in records if r.location_purok}
    purok_coords = ensure_coordinates(purok_names) or purok_coords
    hotspots = _hotspot_data(records, purok_coords)
    mapped_hotspots = [h for h in hotspots if h["lat"] is not None]

    return render_template(
        "analytics.html",
        cross_tabs=cross_tabs,
        risk_indicators=risk_indicators,
        hotspots=hotspots,
        mapped_hotspots=mapped_hotspots,
        map_center=map_center,
    )
