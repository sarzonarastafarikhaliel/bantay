"""Canonical narrative template for scanned blotter records.

Why a template at all: the logbook is written by whoever was on duty, so the
same incident lands as a one-line note on one page and three paragraphs on the
next. That variance is invisible when you read one record and corrosive
everywhere downstream - analytics cannot count a fact nobody wrote down, and
two encoders reading the same page produce two different rows.

The template fixes the SHAPE, never the content. Every slot is filled from what
was actually read off the page, and a slot the page never stated renders as an
explicit BLANK marker rather than silently vanishing. A missing fact must look
missing: an encoder can fill a blank they can see, and cannot fill one that was
quietly dropped.

Note for the classifier: the rendered template is what gets STORED in
IncidentRecord.narrative, but it is NOT what gets classified. Feeding a model
this scaffolding would hand it a constant labelled prefix that carries no
incident signal and dilutes the prose that does. routes/scan.py classifies the
`incident_summary` slot alone. Keep those two uses separate if you edit this file.

Layout note: the entry follows the barangay's paper BLOTTER FORM
(Template/Template Blotter.png) section for section - NO., Petsa ng
Pagblotter, A-E, the statement writer, the witness, the desk officer - under
the form's own Filipino labels, so an encoder can read a stored entry against
the paper page line by line. status and action_taken are not on the paper form;
they trail the entry under their own rule because the barangay records them
after the walk-in. Prose is NOT hard-wrapped - the textarea and the <pre> that
display it both soft-wrap, and a second hard wrap at a fixed column would fight
that and leave ragged half-lines at every window width.
"""
import re

from .ml.mask import mask_names
from .normalize import find_purok_number, normalize_date, normalize_purok, normalize_time

# Order matters: this is both the render order and the form field order. The
# original eight keys keep their names (CSV columns, eval tools and stored gold
# all use them); the rest are the paper form's remaining blanks.
SLOTS = ("blotter_no", "date_reported",
         "reporting_party", "complainant_address", "complainant_contact", "complainant_age",
         "respondent", "respondent_address", "respondent_contact", "respondent_age",
         "complaint",
         "date", "time", "location_purok",
         "incident_summary",
         "statement_writer", "statement_date", "statement_time",
         "witness", "witness_age", "witness_address", "witness_contact",
         "desk_officer",
         "status", "action_taken")

# Slots without which a record is not usable as a record. A page missing any of
# these goes to the encoder rather than into the table - see routes/scan.py.
REQUIRED = ("date", "location_purok", "incident_summary")

BLANK = "(not stated on page)"

# A non-indented line opens a section; indented lines belong to the last one.
# That is what tells section A's TIRAHAN from section B's - see _label_map().
_TEMPLATE = """\
BLOTTER FORM
==================================================
NO.                 : {blotter_no}
PETSA NG PAGBLOTTER : {date_reported}

A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO : {reporting_party}
   TIRAHAN     : {complainant_address}
   CONTACT NO. : {complainant_contact}
   EDAD        : {complainant_age}

B. PANGALAN NG INIREREKLAMO : {respondent}
   TIRAHAN     : {respondent_address}
   CONTACT NO. : {respondent_contact}
   EDAD        : {respondent_age}

C. REKLAMO : {complaint}

D. KAGANAPAN NG PANGYAYARI
   PETSA : {date}
   ORAS  : {time}
   LUGAR : {location_purok}

E. SALAYSAY
{incident_summary}

LAGDA NG SUMULAT NG SALAYSAY
   PANGALAN NG SUMULAT NG SALAYSAY : {statement_writer}
   PETSA : {statement_date}
   ORAS  : {statement_time}

(MGA) TESTIGO
   PANGALAN    : {witness}
   EDAD        : {witness_age}
   TIRAHAN     : {witness_address}
   CONTACT NO. : {witness_contact}

PANGALAN NG BARANGAY DESK OFFICER / IMBESTIGADOR : {desk_officer}

==================================================
KATAYUAN : {status}

AKSYONG GINAWA
{action_taken}"""

# The layout every record stored before the form-shaped template. Never
# rendered any more; kept so parse() and redact_names() still read those
# records and the shipped PDF import template.
_LEGACY_TEMPLATE = """\
BARANGAY BLOTTER ENTRY
==================================================
DATE OF INCIDENT  : {date}
TIME OF INCIDENT  : {time}
PLACE OF INCIDENT : {location_purok}
COMPLAINANT       : {reporting_party}
RESPONDENT        : {respondent}
STATUS            : {status}

NARRATION OF FACTS
{incident_summary}

ACTION TAKEN
{action_taken}"""


def normalize_fields(raw):
    """Run model-extracted slots through the same normalizers the CSV import uses.

    Reuses normalize.py rather than reparsing: a date that arrives as "JAN. 3,
    2026" must become the same string whether it came from a scanned page or a
    spreadsheet, or the two intake paths silently produce two date formats in
    one column.
    """
    raw = raw or {}
    out = {k: (str(raw.get(k) or "").strip()) for k in SLOTS}
    for key in ("date", "date_reported", "statement_date"):
        out[key] = normalize_date(out[key]) or ""
    for key in ("time", "statement_time"):
        out[key] = normalize_time(out[key]) or ""
    # normalize_purok returns "Unspecified" for empty input; an empty slot must
    # stay empty here so backfill() can still try the regex reader on the page.
    out["location_purok"] = normalize_purok(out["location_purok"]) if out["location_purok"] else ""
    if out["location_purok"] == "Unspecified":
        out["location_purok"] = ""
    return out


# status/action_taken are closed-vocabulary in this corpus (Filed/Settled/For
# Hearing; a fixed "Settlement..."/"Scheduled: ..." phrasing) - see
# OCR_ACCURACY_PLAN.md Lever 4. Measured via tools/run_gemini_arms.py: the
# regex classifier in routes/scan.py's guess_fields() beat BOTH Gemini arms on
# these two fields (35.3%/16.1% vs 11.8%/12.5% image, 0.0%/7.1% text), because
# a model answers in its own words ("SETTLEMENT", a copied narrative sentence)
# instead of the barangay's administrative convention. So for just these two
# slots, the deterministic reader outranks the model's own answer - the
# opposite of every other slot, where the model is trusted over the regex.
PREFER_FALLBACK = ("status", "action_taken")


def backfill(fields, fallback):
    """Fill the slots the model left empty. Never overwrites a real value -
    except PREFER_FALLBACK, where the deterministic reader wins outright."""
    merged = dict(fields)
    for key, value in (fallback or {}).items():
        if key not in merged or not value:
            continue
        if key in PREFER_FALLBACK or not merged.get(key):
            merged[key] = value
    return merged


def record_columns(fields):
    """The IncidentRecord `date` and `location_purok` columns for these slots.

    The narrative keeps the paper form's distinctions; the two columns the
    dashboards filter and count on keep the corpus's convention, which the gold
    was encoded under and the 2023 records already follow:

      date            Petsa ng Pagblotter, else D. Petsa. A logbook page dates
                      the ENTRY; the incident date is often never written, and
                      when it is, the gold still records the entry date.
                      Measured on the 151-page dev split (gemini_*_form runs):
                      D. Petsa alone 58-60%, this fallback 85-91%.
      location_purok  a purok named in D. Lugar, else in the complainant's
                      address, else the respondent's; only then D. Lugar's own
                      text ("SIDE NG BAHAY KO"). The gold records the purok - a
                      walk-in rarely places the incident apart from where the
                      parties live, and a descriptive Lugar names no purok.

    Only the columns borrow - the stored narrative still shows D. Lugar blank
    when the page left it blank, so nothing in the record claims the page said
    more than it did.
    """
    f = fields or {}
    lugar = str(f.get("location_purok") or "").strip()
    purok = next((p for p in (find_purok_number(str(f.get(k) or "")) for k in
                              ("location_purok", "complainant_address", "respondent_address")) if p), "")
    return {"date": str(f.get("date_reported") or f.get("date") or "").strip(),
            "location_purok": purok or lugar}


def missing(fields):
    """Required slots this page did not yield. Feeds the review gate. date and
    location count as present when record_columns() can fill the column."""
    have = dict(fields or {}, **record_columns(fields))
    return [k for k in REQUIRED if not have.get(k)]


def render(fields):
    """The stored narrative: fixed shape, explicit blanks, no invented content."""
    filled = {k: (str((fields or {}).get(k) or "").strip() or BLANK) for k in SLOTS}
    return _TEMPLATE.format(**filled)


# The inverse of render(). Labels are derived from the templates themselves
# rather than spelled out a second time, so a label edited in _TEMPLATE cannot
# drift from what parse() reads. Used by the CSV/PDF import so an entry typed in
# this layout lands in the same slots the New Blotter Record form posts.
def _label_map(template):
    """(section, LABEL) -> slot for every 'LABEL : {slot}' line, plus the labels
    that open a section (the non-indented ones). Sections are what let section
    A's TIRAHAN and section B's TIRAHAN land in different slots. parse() tracks
    them by label, not indentation, because PDF text extraction drops leading
    spaces."""
    labels, openers, section = {}, set(), ""
    for line in template.splitlines():
        label = line.split(":")[0].strip().upper()
        if line.strip() and not line[:1].isspace():
            section = label
            openers.add(label)
        slot = re.search(r":\s*\{(\w+)\}\s*$", line)
        if slot:
            labels[(section, label)] = slot.group(1)
    return labels, openers


# Per layout: its entry heading, its label map, and its prose blocks as
# (slot, heading, next heading or None for end-of-entry). The trailing block
# comes first so it is cut out before the one above it is read to "the end".
_LAYOUTS = {
    "BLOTTER FORM": (_label_map(_TEMPLATE),
                     (("action_taken", "AKSYONG GINAWA", None),
                      ("incident_summary", r"E\.\s*SALAYSAY", "LAGDA NG SUMULAT NG SALAYSAY"))),
    "BARANGAY BLOTTER ENTRY": (_label_map(_LEGACY_TEMPLATE),
                               (("action_taken", "ACTION TAKEN", None),
                                ("incident_summary", "NARRATION OF FACTS", None))),
}
_ENTRY_SPLIT = re.compile(r"^\s*(BLOTTER FORM|BARANGAY BLOTTER ENTRY)\s*$",
                          re.MULTILINE | re.IGNORECASE)


def _clean(value):
    value = re.sub(r"^\s*=+\s*$", "", value or "", flags=re.MULTILINE).strip()
    return "" if value == BLANK else value


def parse(text):
    """Split text holding one or more BLOTTER FORM (or legacy BARANGAY BLOTTER
    ENTRY) blocks into slot dicts (SLOTS keys). The BLANK marker and ==== rules
    read back as empty. Text with no entry heading yields no entries."""
    parts = _ENTRY_SPLIT.split(text or "")
    entries = []
    for heading, chunk in zip(parts[1::2], parts[2::2]):
        (labels, openers), prose = _LAYOUTS[heading.strip().upper()]
        fields = dict.fromkeys(SLOTS, "")
        # Prose first, and cut out of the chunk: a narration line that happens
        # to read "PETSA: ..." is the writer's words, not a labelled field.
        for slot, start, end in prose:
            stop = r"(?=^\s*%s\s*:?\s*$|\Z)" % end if end else r"\Z"
            m = re.search(r"^\s*%s\s*:?\s*$(.*?)%s" % (start, stop), chunk,
                          re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if m:
                fields[slot] = _clean(m.group(1))
                chunk = chunk[:m.start()] + chunk[m.end():]
        section = ""
        for line in chunk.splitlines():
            label, sep, value = line.partition(":")
            label = label.strip().upper()
            if label in openers:
                section = label
            slot = labels.get((section, label))
            if sep and slot:
                fields[slot] = _clean(value)
        entries.append(fields)
    return entries


# Lines whose value identifies a private person, in either layout. Names are
# also hunted down inside the prose (see redact_names); addresses and contact
# numbers are masked on their own lines, and a contact number wherever else it
# is repeated. The desk officer is barangay staff acting officially, not a
# party, and is left visible.
_NAME_LABELS = ("COMPLAINANT", "RESPONDENT",
                "A. PANGALAN NG NAG-BLO-BLOTTER/NAGREREKLAMO", "B. PANGALAN NG INIREREKLAMO",
                "PANGALAN NG SUMULAT NG SALAYSAY", "PANGALAN")
_CONTACT_LABELS = ("CONTACT NO.",)
_ADDRESS_LABELS = ("TIRAHAN",)
_PII_LINE = re.compile(r"^(\s*(%s)\s*: ?)(.*)$" % "|".join(
    re.escape(l) for l in _NAME_LABELS + _CONTACT_LABELS + _ADDRESS_LABELS), re.MULTILINE)

_REDACTED = "[REDACTED]"


def redact_names(text):
    """Mask the parties' names, addresses and contact numbers throughout a
    rendered narrative.

    A name is not confined to its own labelled line - the encoder's free
    prose in E. SALAYSAY routinely repeats it ("NAGSADYA SI JUAN..."). So this
    masks every name/address/contact line itself, then reuses ml/mask.py's
    name-token matching (built for the same "Complainant: X; Respondent: Y"
    shape) to catch every other bare occurrence of those names - full name
    first, then individual tokens, so a later surname- or first-name-only
    reference is caught too. A stated contact number is masked wherever it is
    repeated.

    Reads both the form-shaped layout and the legacy COMPLAINANT/RESPONDENT
    one, since records stored before the change keep their old text.

    Used for every non-admin read surface (list, dashboard, record view,
    review queue) - see bantay/__init__.py's `visible_narrative` filter.
    Editing still sees the real text: a reviewer correcting OCR errors needs
    the actual name to fix a misread one.
    """
    if not text:
        return ""
    hits = [(m.group(2), m.group(3).strip()) for m in _PII_LINE.finditer(text)]
    hits = [(label, v) for label, v in hits if v and v != BLANK]

    out = _PII_LINE.sub(lambda m: f"{m.group(1)}{_REDACTED}", text)
    names = [v for label, v in hits if label in _NAME_LABELS]
    if names:
        # Label doesn't matter here - extract_names() only keys on finding
        # *a* recognized label before the colon, not on which one.
        remarks = "; ".join(f"Complainant: {v}" for v in names)
        out = mask_names(out, remarks, placeholder=_REDACTED)
    for label, v in hits:
        if label in _CONTACT_LABELS:
            out = re.sub(re.escape(v), _REDACTED, out, flags=re.IGNORECASE)
    return out
