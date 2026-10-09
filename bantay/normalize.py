import json
import os
import re

import pandas as pd

_CITE_RE = re.compile(r"\[cite:[^\]]*\]", re.IGNORECASE)
# Optional trailing letter (Purok 1B, Purok 4 B EXT): 7 of the 10 surveyed
# puroks split into A/B sub-puroks in the real corpus, and the old digit-only
# pattern silently dropped that letter, collapsing "Purok 4B" to "Purok 4".
# Also the short forms encoders write in an address ("P-4", "PRK. 4") and Roman
# numerals ("PUROK I, MANARANG COMPOUND") - all three appear in the gold and
# none were read before, so those pages scored as a missing purok. "P-" needs
# the hyphen: a bare "P4" is as likely to be pesos as a purok.
_PUROK_RE = re.compile(r"(?:purok|\bprk\.?|\bp-)\s*#?\s*(\d+|[ivx]{1,4}(?![a-z]))\s*([a-z])?\b",
                       re.IGNORECASE)
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
_NO_DATE_VALUES = {"", "not indicated", "n/a", "na", "unknown", "unspecified"}

# Real barangay blotter exports carry loose, inconsistent incident labels
# ("Robbery (Panloloob)", "For Theft", "Blotter (Theft)"...). This keyword
# taxonomy is a first-pass heuristic mapping to canonical categories, checked
# most-specific first (a "theft" match wins over a later "property" match).
#
# Taxonomy aligned with IncidentType-CategoryGroup.csv — the 34-item
# encoded-incident-type list actually used in bantay_master_MASKED_formatted.csv
# and bantay_synthetic_seed_2023_1.csv (§3.2.4). Four extra categories
# (Peeping/Voyeurism, Curfew Violation, Amicable Settlement, Work/Service
# Dispute) are kept from the pre-CSV taxonomy — they don't appear in the CSV
# but do appear in raw free-text imports (data/raw/incident_reports.csv) and
# already have researched PNP_CLASSIFICATION citations below.
CANONICAL_CATEGORIES = [
    ("Robbery", ["robbery", "hold-up", "holdap"]),
    ("Theft", ["theft", "nakaw", "stolen"]),
    ("Slight Physical Injury", ["slight physical injury", "slight injury"]),
    ("Physical Injury", [
        "physical injury", "pisikal na pinsala", "fistfight", "punch",
        "pag-aaway", "quarrel", "altercation", "stone throwing",
    ]),
    ("Lost Property / Document", ["lost property", "lost document", "nawawalang ari-arian", "nawala"]),
    ("Ejectment", ["ejectment", "vacate", "paalis"]),
    ("RA 9262 (VAWC)", ["ra 9262", "vawc", "violence against women"]),
    ("Online Defamation / Threat", ["online defamation", "cyber libel", "online threat", "facebook threat"]),
    ("Oral Defamation", ["oral defamation", "slander", "mura", "harassment"]),
    ("Marital Relation", ["marital relation", "asawa", "live-in partner", "jealousy"]),
    ("Malicious Mischief / Property Damage", ["malicious mischief", "vandalism", "property damage", "pagkasira ng ari-arian"]),
    ("Property Claim / Damage", ["property claim", "damage claim"]),
    ("Grave Threats / Light Threats", ["grave threat", "light threat", "banta"]),
    ("Unjust Vexation", ["unjust vexation", "vexation"]),
    ("Alarms and Scandals", ["alarms and scandals", "alarming scandal", "public scandal"]),
    ("Trespassing", ["trespassing", "panghihimasok"]),
    ("Noise Complaint", ["noise", "ingay", "videoke", "karaoke"]),
    ("Neighbor Dispute", ["neighbor dispute", "kapitbahay"]),
    ("Estafa (Swindling)", ["estafa", "swindling", "panloloko"]),
    ("Boundary Dispute", ["boundary dispute", "boundary", "hangganan"]),
    ("Animal-Related Incident", ["animal-related", "aso", "hayop", "dog bite"]),
    ("Coercion", ["coercion", "pamimilit"]),
    ("Using Fictitious Names / False Certificates", ["fictitious name", "false certificate", "pekeng pangalan"]),
    ("Breach of Contract", ["breach of contract", "kontrata"]),
    ("Demand for Specific Performance", ["specific performance"]),
    ("Family/Sibling Dispute", ["sibling dispute", "magkapatid"]),
    ("Child Custody-Adjacent Concern", ["child custody", "custody ng bata"]),
    ("Vehicular Incident", ["motor accident", "motor vehicle collision", "vehicular", "traffic collision", "hit & run", "hit and run"]),
    ("Public Disturbance", ["public disturbance", "gulo"]),
    ("Drug-Related Concern", ["drug-related", "shabu", "droga"]),
    ("Missing Person Report", ["missing person"]),
    ("Referral to Police/Other Agency", ["referral to police", "referred to police", "for referral"]),
    ("Collection of Sum of Money", ["sum of money", "collection", "debt", "utang"]),
    # -- Kept from the pre-CSV taxonomy (not in the CSV, still seen in raw imports) --
    ("Peeping/Voyeurism", ["peeping", "paninilip"]),
    ("Curfew Violation", ["curfew"]),
    ("Amicable Settlement", ["kasunduan", "amicable settlement", "agreement", "settlement"]),
    ("Work/Service Dispute", ["work instruction", "job order"]),
]
FALLBACK_CATEGORY = "Others / Miscellaneous"

# ---------------------------------------------------------------------------
# Category Group — coarse grouping tier from IncidentType-CategoryGroup.csv
# (Civil Dispute / Criminal-Penal Code / Property-Lost Items / Family-Domestic /
# Community Disturbance / Administrative-Referral / Catch-all). Independent of
# the PNP tier below: category group is about *what kind* of matter it is,
# PNP tier is about *how it's legally classified*.
# ---------------------------------------------------------------------------
CATEGORY_GROUPS = {
    # -- Civil Dispute --
    "Collection of Sum of Money": "Civil Dispute",
    "Ejectment": "Civil Dispute",
    "Property Claim / Damage": "Civil Dispute",
    "Boundary Dispute": "Civil Dispute",
    "Breach of Contract": "Civil Dispute",
    "Demand for Specific Performance": "Civil Dispute",
    "Work/Service Dispute": "Civil Dispute",
    # -- Criminal/Penal Code --
    "Theft": "Criminal/Penal Code",
    "Robbery": "Criminal/Penal Code",
    "Physical Injury": "Criminal/Penal Code",
    "Slight Physical Injury": "Criminal/Penal Code",
    "Oral Defamation": "Criminal/Penal Code",
    "Online Defamation / Threat": "Criminal/Penal Code",
    "Grave Threats / Light Threats": "Criminal/Penal Code",
    "Unjust Vexation": "Criminal/Penal Code",
    "Alarms and Scandals": "Criminal/Penal Code",
    "Malicious Mischief / Property Damage": "Criminal/Penal Code",
    "Trespassing": "Criminal/Penal Code",
    "Estafa (Swindling)": "Criminal/Penal Code",
    "Coercion": "Criminal/Penal Code",
    "Using Fictitious Names / False Certificates": "Criminal/Penal Code",
    "Peeping/Voyeurism": "Criminal/Penal Code",
    # -- Property/Lost Items --
    "Lost Property / Document": "Property/Lost Items",
    # -- Family/Domestic --
    "RA 9262 (VAWC)": "Family/Domestic",
    "Marital Relation": "Family/Domestic",
    "Family/Sibling Dispute": "Family/Domestic",
    "Child Custody-Adjacent Concern": "Family/Domestic",
    # -- Community Disturbance --
    "Noise Complaint": "Community Disturbance",
    "Neighbor Dispute": "Community Disturbance",
    "Animal-Related Incident": "Community Disturbance",
    "Vehicular Incident": "Community Disturbance",
    "Public Disturbance": "Community Disturbance",
    "Drug-Related Concern": "Community Disturbance",
    "Curfew Violation": "Community Disturbance",
    # -- Administrative/Referral --
    "Missing Person Report": "Administrative/Referral",
    "Referral to Police/Other Agency": "Administrative/Referral",
    "Amicable Settlement": "Administrative/Referral",
    # -- Catch-all --
    FALLBACK_CATEGORY: "Catch-all",
}
CATEGORY_GROUP_NAMES = [
    "Civil Dispute", "Criminal/Penal Code", "Property/Lost Items",
    "Family/Domestic", "Community Disturbance", "Administrative/Referral", "Catch-all",
]


def get_category_group(category):
    """Return the Category Group for a canonical incident category.

    Falls back to "Catch-all" for FALLBACK_CATEGORY or any unmapped category.
    """
    return CATEGORY_GROUPS.get(category, "Catch-all")


# ---------------------------------------------------------------------------
# PNP crime-classification tier + legal basis per canonical category.
#
# Tiers follow the PNP's official crime-statistics split:
#   - "Index Crime"     — the 8 crimes tracked in PNP crime statistics
#                          (Murder, Homicide, Physical Injury, Rape, Robbery
#                          against persons; Theft, Carnapping, Cattle Rustling
#                          against property).
#   - "Non-Index Crime" — all other Revised Penal Code offenses, special-law
#                          violations, and local ordinance violations.
#   - "Non-Criminal Complaint" — not a crime; a dispute resolved through
#                          barangay conciliation under Katarungang Pambarangay
#                          (PD 1508 / RA 7160), which is how most blotter
#                          entries in this dataset actually resolve.
# ---------------------------------------------------------------------------
PNP_CLASSIFICATION = {
    "Theft": ("Index Crime", "RPC Art. 308-311 (Theft)"),
    "Robbery": ("Index Crime", "RPC Art. 293-299 (Robbery)"),
    "Physical Injury": ("Index Crime", "RPC Art. 262 (Mutilation), 263 (Serious), 265 (Less Serious) Physical Injuries"),
    "Slight Physical Injury": ("Index Crime", "RPC Art. 266 (Slight Physical Injuries)"),

    "RA 9262 (VAWC)": ("Non-Index Crime", "RA 9262 (Anti-Violence Against Women and Their Children Act)"),
    "Online Defamation / Threat": ("Non-Index Crime", "RA 10175 (Cybercrime Prevention Act) in relation to RPC Art. 355 (Libel) / Art. 282 (Grave Threats)"),
    "Grave Threats / Light Threats": ("Non-Index Crime", "RPC Art. 282 (Grave Threats); Art. 285 (Light Threats)"),
    "Unjust Vexation": ("Non-Index Crime", "RPC Art. 287 (Unjust Vexation)"),
    "Alarms and Scandals": ("Non-Index Crime", "RPC Art. 155 (Alarms and Scandals)"),
    "Malicious Mischief / Property Damage": ("Non-Index Crime", "RPC Art. 327-331 (Malicious Mischief)"),
    "Trespassing": ("Non-Index Crime", "RPC Art. 280 (Qualified Trespass to Dwelling); Art. 281 (Other Forms of Trespass)"),
    "Estafa (Swindling)": ("Non-Index Crime", "RPC Art. 315 (Estafa)"),
    "Coercion": ("Non-Index Crime", "RPC Art. 286 (Grave Coercion); Art. 287 (Light Coercion)"),
    "Using Fictitious Names / False Certificates": ("Non-Index Crime", "RPC Art. 178 (Using Fictitious Name); Art. 171-172 (Falsification)"),
    "Vehicular Incident": ("Non-Index Crime", "RPC Art. 365 (Reckless Imprudence); RA 4136 (Land Transportation and Traffic Code)"),
    "Public Disturbance": ("Non-Index Crime", "RPC Art. 153 (Tumults and Other Disturbances of Public Order)"),
    "Drug-Related Concern": ("Non-Index Crime", "RA 9165 (Comprehensive Dangerous Drugs Act of 2002)"),
    "Noise Complaint": ("Non-Index Crime", "Local noise ordinance, enacted under RA 7160 Sec. 16 (General Welfare Clause)"),
    "Curfew Violation": ("Non-Index Crime", "Local curfew ordinance (adults only) — RA 10630 exempts minors from punishment as a status offense"),
    "Peeping/Voyeurism": ("Non-Index Crime", "RA 9995 (Anti-Photo and Video Voyeurism Act) if photo/video capture occurred; else RPC Art. 287 (Unjust Vexation)"),

    "Oral Defamation": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); may escalate to RPC Art. 358 (Oral Defamation) if unresolved"),
    "Lost Property / Document": ("Non-Criminal Complaint", "RA 7160 (barangay blotter recording); Civil Code Art. 719-720 (finder of lost movable property)"),
    "Ejectment": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 1673 (grounds for judicial ejectment)"),
    "Marital Relation": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Family Code Art. 149 et seq. (The Family)"),
    "Property Claim / Damage": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 2176 (quasi-delict)"),
    "Neighbor Dispute": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay)"),
    "Boundary Dispute": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 434 (accion reivindicatoria)"),
    "Animal-Related Incident": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 2183 (possessor of animal's liability)"),
    "Breach of Contract": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 1170 et seq. (Breach of Obligations)"),
    "Demand for Specific Performance": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 1191 / 1233 et seq. (Specific Performance)"),
    "Family/Sibling Dispute": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Family Code provisions on family relations"),
    "Child Custody-Adjacent Concern": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay, referral); Family Code Art. 213 (custody of children)"),
    "Missing Person Report": ("Non-Criminal Complaint", "RA 7160 (barangay blotter recording/referral) — not itself an offense"),
    "Referral to Police/Other Agency": ("Non-Criminal Complaint", "RA 7160 Sec. 408-409 (Katarungang Pambarangay jurisdictional exceptions requiring referral)"),
    "Collection of Sum of Money": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 1156 et seq. (Obligations)"),
    "Amicable Settlement": ("Non-Criminal Complaint", "RA 7160 Sec. 411-422 (Katarungang Pambarangay Settlement)"),
    "Work/Service Dispute": ("Non-Criminal Complaint", "RA 7160 Sec. 399-422 (Katarungang Pambarangay); Civil Code Art. 1713 et seq. (Contract for a Piece of Work)"),
}
PNP_TIERS = ["Index Crime", "Non-Index Crime", "Non-Criminal Complaint", "Unclassified"]


def get_pnp_classification(category):
    """Return (tier, legal_basis) for a canonical incident category.

    Falls back to ("Unclassified", "—") for FALLBACK_CATEGORY or any
    category not in the PNP_CLASSIFICATION map.
    """
    return PNP_CLASSIFICATION.get(category, ("Unclassified", "—"))


# ---------------------------------------------------------------------------
# Katarungang Pambarangay (KP) referability — third, independent axis.
#
# Not derivable from the PNP tier. RA 7160 Sec. 408 gives the Lupon authority
# over disputes between residents of the same city/municipality *regardless of
# whether the matter is a crime*; what removes a case is the penalty ceiling
# (imprisonment > 1 year OR fine > P5,000), the absence of a private offended
# party, or an express statutory bar. So Slight Physical Injury is a crime and
# still KP-mediable (arresto menor), while Online Defamation is KP-excluded
# because RA 10175 Sec. 6 raises the penalty one degree past the ceiling.
#
# Three statuses, not a boolean, because the middle one is where the barangay
# investigator actually has to decide:
#   "KP-Mediable" - squarely within Lupon authority.
#   "Conditional" - mediable only if a stated threshold holds. The note names
#                   the exact test to apply (damage value, injury severity,
#                   amount defrauded). This is the investigator aid.
#   "Excluded"    - outside Lupon authority; the note says why.
#
# The system reports the applicable rule. It does not decide the case -
# certification to file action is issued by the Lupon/Pangkat secretary and attested by the chairman (RA 7160 Sec. 412).
# ---------------------------------------------------------------------------
_KP_CEILING = "penalty ceiling: imprisonment > 1 yr or fine > P5,000 (RA 7160 Sec. 408[c])"

KP_STATUS = {
    # -- KP-Mediable: within Sec. 408, no threshold to check --
    "Collection of Sum of Money": ("KP-Mediable", "Civil dispute between residents — RA 7160 Sec. 408."),
    "Ejectment": ("KP-Mediable", "Civil dispute; prior barangay conciliation is a condition precedent to filing (RA 7160 Sec. 412[a])."),
    "Property Claim / Damage": ("KP-Mediable", "Civil claim under Civil Code Art. 2176 — RA 7160 Sec. 408."),
    "Boundary Dispute": ("KP-Mediable", "Civil dispute; real property within the same city/municipality (RA 7160 Sec. 408[e])."),
    "Breach of Contract": ("KP-Mediable", "Civil dispute — RA 7160 Sec. 408."),
    "Demand for Specific Performance": ("KP-Mediable", "Civil dispute — RA 7160 Sec. 408."),
    "Work/Service Dispute": ("KP-Mediable", "Civil dispute over a contract for a piece of work — RA 7160 Sec. 408."),
    "Neighbor Dispute": ("KP-Mediable", "Core Lupon subject matter — RA 7160 Sec. 408."),
    "Family/Sibling Dispute": ("KP-Mediable", "RA 7160 Sec. 408; Family Code Art. 151 requires earnest efforts toward a compromise."),
    "Marital Relation": ("KP-Mediable", "RA 7160 Sec. 408 — but re-screen for RA 9262 (VAWC). If any element of violence appears, KP is barred."),
    "Animal-Related Incident": ("KP-Mediable", "Civil liability of the animal's possessor, Civil Code Art. 2183 — RA 7160 Sec. 408."),
    "Noise Complaint": ("KP-Mediable", "Ordinance violation with a private complainant; fine within the Sec. 408[c] ceiling."),
    "Slight Physical Injury": ("KP-Mediable", "RPC Art. 266 — arresto menor (1–30 days), within the Sec. 408[c] ceiling. A crime, but still KP-mediable."),
    "Unjust Vexation": ("KP-Mediable", "RPC Art. 287 — arresto menor / fine, within the Sec. 408[c] ceiling."),
    "Amicable Settlement": ("KP-Mediable", "The KP outcome itself — RA 7160 Sec. 411-422."),

    # -- Conditional: mediable only if the stated test passes --
    "Physical Injury": ("Conditional", "Mediable if LESS SERIOUS (RPC Art. 265, arresto mayor). EXCLUDED if SERIOUS (Art. 263) or mutilation (Art. 262) — those exceed the " + _KP_CEILING + ". Check the medical certificate and days of incapacity."),
    "Theft": ("Conditional", "Penalty scales with the value taken (RPC Art. 309 as amended by RA 10951). Mediable only for small values falling within the " + _KP_CEILING + ". Establish the value first."),
    "Estafa (Swindling)": ("Conditional", "Penalty scales with the amount defrauded (RPC Art. 315 as amended by RA 10951). Establish the amount, then apply the " + _KP_CEILING + "."),
    "Malicious Mischief / Property Damage": ("Conditional", "Penalty scales with the damage value (RPC Art. 328-329). Establish the value, then apply the " + _KP_CEILING + "."),
    "Trespassing": ("Conditional", "Mediable if OTHER FORMS of trespass (RPC Art. 281, arresto menor). EXCLUDED if QUALIFIED trespass to dwelling (Art. 280, prision correccional)."),
    "Coercion": ("Conditional", "Mediable if LIGHT coercion (RPC Art. 287). EXCLUDED if GRAVE coercion (Art. 286, prision correccional)."),
    "Grave Threats / Light Threats": ("Conditional", "Mediable if LIGHT threats (RPC Art. 285, arresto menor). GRAVE threats (Art. 282) depend on whether a condition was attached and achieved — apply the " + _KP_CEILING + "."),
    "Using Fictitious Names / False Certificates": ("Conditional", "Mediable if use of a fictitious name (RPC Art. 178, arresto mayor). EXCLUDED if falsification (Art. 171-172, prision correccional or higher)."),
    "Vehicular Incident": ("Conditional", "Reckless imprudence (RPC Art. 365) — the penalty follows the resulting harm. Damage to property alone is normally mediable; EXCLUDED where death or serious physical injury resulted."),
    "Oral Defamation": ("Conditional", "Mediable if SLIGHT slander (RPC Art. 358, arresto menor). SERIOUS slander reaches prision correccional minimum and exceeds the " + _KP_CEILING + "."),
    "Alarms and Scandals": ("Conditional", "RPC Art. 155 is within the penalty ceiling, but check RA 7160 Sec. 408[d] — with no private offended party, KP does not apply."),
    "Public Disturbance": ("Conditional", "RPC Art. 153. Check RA 7160 Sec. 408[d] — a purely public-order matter with no private offended party is outside KP."),
    "Peeping/Voyeurism": ("Conditional", "Mediable if it amounts only to unjust vexation (RPC Art. 287). EXCLUDED if photo or video capture occurred — RA 9995 carries 3 to 7 years."),

    # -- Excluded: outside Lupon authority --
    "RA 9262 (VAWC)": ("Excluded", "EXPRESS STATUTORY BAR — RA 9262 Sec. 33 prohibits barangay conciliation of VAWC cases. Do not mediate; assist with a Barangay Protection Order (Sec. 14) and refer."),
    "Robbery": ("Excluded", "RPC Art. 293-299 — prision correccional to reclusion perpetua, far beyond the " + _KP_CEILING + "."),
    "Drug-Related Concern": ("Excluded", "RA 9165 — penalties far beyond the ceiling, and no private offended party (RA 7160 Sec. 408[d]). Refer to the PNP / PDEA."),
    "Online Defamation / Threat": ("Excluded", "RA 10175 Sec. 6 raises the penalty one degree above RPC Art. 355, exceeding the " + _KP_CEILING + "."),
    "Child Custody-Adjacent Concern": ("Excluded", "Custody falls within the exclusive original jurisdiction of the Family Courts (RA 8369 Sec. 5). Record and refer; coordinate with the DSWD / VAWC desk."),
    "Curfew Violation": ("Excluded", "Ordinance violation with no private offended party (RA 7160 Sec. 408[d]); handled administratively. RA 10630 bars punishing minors for status offenses."),
    "Referral to Police/Other Agency": ("Excluded", "Already a Sec. 408-409 jurisdictional exception — this label exists to record the referral."),
    "Missing Person Report": ("Excluded", "Not a dispute between parties; there is nothing to conciliate. Blotter recording and referral only."),
    "Lost Property / Document": ("Excluded", "Not a dispute between parties. Blotter recording only; Civil Code Art. 719-720 governs found property."),
}
KP_STATUSES = ["KP-Mediable", "Conditional", "Excluded", "Unclassified"]


def get_kp_status(category):
    """Return (kp_status, note) for a canonical incident category.

    Falls back to ("Unclassified", "—") for FALLBACK_CATEGORY or any category
    not in the map — an incident with no assigned type cannot be screened for
    KP referability until a human assigns one.
    """
    return KP_STATUS.get(category, ("Unclassified", "—"))


LABEL_SEP = "|"


def split_labels(raw):
    if not raw:
        return []
    return [label.strip() for label in raw.split(LABEL_SEP) if label.strip()]


def join_labels(labels):
    return LABEL_SEP.join(labels)


def fix_mojibake(text):
    """Undo UTF-8-read-as-Latin-1 corruption."""
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def strip_citations(text):
    """Strip PDF/citation-export artifacts like '[cite: 19, 20]'."""
    if not text:
        return text
    return re.sub(r"\s+", " ", _CITE_RE.sub("", text)).strip()


def normalize_date(raw):
    """Parse loose real-world date strings into YYYY-MM-DD, or None."""
    if raw is None:
        return None
    text = strip_citations(str(raw)).strip()
    if not text or text.lower() in _NO_DATE_VALUES:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def normalize_time(raw):
    """Normalize time strings to HH:MM format, or None if unparseable.

    Handles common barangay blotter formats: '09:30', '9:30 AM', '930', etc.
    """
    if not raw:
        return None
    text = str(raw).strip()
    # Already HH:MM
    if re.match(r"^\d{2}:\d{2}$", text):
        return text
    # H:MM AM/PM
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", text, re.IGNORECASE)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        period = (m.group(3) or "").upper()
        if period == "PM" and hour != 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    # HHMM compact
    m2 = re.match(r"^(\d{2})(\d{2})$", text)
    if m2:
        return f"{m2.group(1)}:{m2.group(2)}"
    return None


def normalize_batch_number(raw):
    """Normalize batch_number: strip whitespace, uppercase for consistency."""
    if not raw:
        return None
    return str(raw).strip().upper() or None


def find_purok_number(text):
    """Search free text for a 'Purok N[letter]' mention, anywhere in the string.

    Shared by normalize_purok (a structured address field) and
    routes/scan.py's guess_fields (a whole OCR'd page) so the two intake paths
    can't drift into recognising different purok spellings.
    """
    match = _PUROK_RE.search(text or "")
    if not match:
        return None
    number = match.group(1).upper()
    number = _ROMAN.get(number, number)
    if not str(number).isdigit():
        return None                        # "XIV"-style noise, not a purok
    suffix = (match.group(2) or "").upper()
    return f"Purok {number}{suffix}"


def normalize_purok(raw):
    """Extract a 'Purok N' label from free-text addresses."""
    if not raw:
        return "Unspecified"
    text = strip_citations(fix_mojibake(str(raw)))
    found = find_purok_number(text)
    if found:
        return found
    first_segment = re.split(r"[/,]", text)[0].strip()
    return first_segment or "Unspecified"


# The place gazetteer: `puroks` (names with a map coordinate) plus `aliases`
# (spelling variant -> canonical name, or null for text that names no place).
# Same file the hotspot map and geocode.py read, so a canonical name and its
# map pin can't drift apart.
PLACES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "purok_coordinates.json")
_PLACE_WORDS = {"street": "st", "avenue": "ave", "subdivision": "subd", "barangay": "brgy",
                "compound": "comp"}
_HOUSE_NO_RE = re.compile(r"^(?:\d+[a-z]?\s+)+(?:[a-z]\s+)?")   # "#2040 ", "56-A ", "5-29 A "
_NO_PLACE_KEYS = {"", "unspecified", "unknown", "n a", "na", "none", "not indicated"}


def _place_key(text):
    """Lowercase, punctuation-free, abbreviation-folded, house number dropped:
    "#2040 Oregon St." and "2040 OREGON STREET" both become "oregon st"."""
    words = re.sub(r"[\W_]+", " ", str(text or "").casefold()).split()
    key = " ".join(_PLACE_WORDS.get(w, w) for w in words)
    return _HOUSE_NO_RE.sub("", key + " ").strip()


def _load_places(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    lookup = {_place_key(name): name for name in data.get("puroks", {})}
    for b in data.get("_meta", {}).get("adjacent_barangays", []):
        lookup[_place_key(f"brgy {b}")] = "Outside Anunas"
    lookup.update({_place_key(alias): target for alias, target in data.get("aliases", {}).items()})
    return lookup


def place_names(path=PLACES_PATH):
    """Every canonical place name, for the form's <datalist> suggestions."""
    return sorted({name for name in _load_places(path).values() if name})


def canonical_location(raw, path=PLACES_PATH):
    """The one place name analytics groups on, for whatever the Lugar says.

    Order: a purok number anywhere in the text; an exact gazetteer match; a
    gazetteer name inside the text, longest first ("4-19 COLORADO VILLASOL" ->
    Villasol Subdivision); else the cleaned text itself, so an unlisted place
    still groups across case, punctuation and house numbers. Returns None for
    text that names no place - analytics shows that as "Unknown".

    ponytail: re-reads the gazetteer per call (a few KB) so an alias added to
    the JSON applies without a restart; cache it if bulk imports get slow.
    """
    found = find_purok_number(strip_citations(fix_mojibake(str(raw or ""))))
    if found:
        return found
    key = _place_key(fix_mojibake(str(raw or "")))
    if key in _NO_PLACE_KEYS:
        return None
    lookup = _load_places(path)
    if key in lookup:
        return lookup[key]
    # A null alias means "this whole text names no place" - exact match only,
    # or "brgy anunas" would swallow "brgy anunas brgy office".
    for k in sorted((k for k, v in lookup.items() if k and v), key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", key):
            return lookup[k]
    return " ".join(w.capitalize() for w in key.split())


def normalize_incident_types(raw):
    """Multi-label: returns every canonical category whose keywords match.

    Returns (primary, secondary) where:
      primary   — first/best match (the ML target label)
      secondary — additional matches (optional overlapping concern, §3.2.4)
    """
    if not raw:
        return FALLBACK_CATEGORY, None
    text = strip_citations(str(raw)).lower()
    matches = [canonical for canonical, keywords in CANONICAL_CATEGORIES if any(kw in text for kw in keywords)]
    if not matches:
        return FALLBACK_CATEGORY, None
    primary = matches[0]
    secondary = matches[1] if len(matches) > 1 else None
    return primary, secondary


def normalize_readability(raw):
    if not raw:
        return "readable"
    text = str(raw).strip().lower()
    if "unreadable" in text or "illegible" in text:
        return "unreadable"
    if "partial" in text or "moderate" in text:
        return "partial"
    return "readable"


def clean_row(row):
    """Apply all field-level normalization to one raw CSV/import row dict.

    Updated to handle both old (incident_type) and new
    (incident_type_primary / incident_type_secondary) column schemas.
    Idempotent — safe to run on already-clean data.
    """
    cleaned = dict(row)
    for key in ("narrative", "action_taken", "status", "remarks", "source_file_page",
                 "persons_involved_masked", "encoded_by",
                 "incident_summary", "reporting_party", "respondent"):
        if cleaned.get(key):
            cleaned[key] = strip_citations(fix_mojibake(str(cleaned[key]))).strip()

    cleaned["date"] = normalize_date(cleaned.get("date"))
    cleaned["time"] = normalize_time(cleaned.get("time"))
    cleaned["location_purok"] = normalize_purok(cleaned.get("location_purok"))
    cleaned["batch_number"] = normalize_batch_number(cleaned.get("batch_number"))
    cleaned["readability"] = normalize_readability(cleaned.get("readability"))

    # Primary/secondary incident type handling
    if "incident_type_primary" in cleaned and cleaned.get("incident_type_primary"):
        # New schema — normalize primary directly, secondary optionally
        primary, _ = normalize_incident_types(cleaned["incident_type_primary"])
        cleaned["incident_type_primary"] = primary
        if cleaned.get("incident_type_secondary"):
            sec_primary, _ = normalize_incident_types(cleaned["incident_type_secondary"])
            cleaned["incident_type_secondary"] = sec_primary
    elif "incident_type" in cleaned:
        # Old schema — split multi-label into primary + secondary
        primary, secondary = normalize_incident_types(cleaned.get("incident_type"))
        cleaned["incident_type_primary"] = primary
        cleaned["incident_type_secondary"] = secondary
        cleaned["incident_type"] = join_labels([primary] + ([secondary] if secondary else []))
    else:
        cleaned["incident_type_primary"] = FALLBACK_CATEGORY
        cleaned["incident_type_secondary"] = None

    cleaned["pnp_classification"] = get_pnp_classification(cleaned["incident_type_primary"])[0]
    cleaned["category_group"] = get_category_group(cleaned["incident_type_primary"])
    cleaned["kp_status"] = get_kp_status(cleaned["incident_type_primary"])[0]

    return cleaned
