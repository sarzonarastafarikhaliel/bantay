import re

_LABELS = (
    "complainants", "complainant", "respondents", "respondent",
    "nagsusumbong", "ipinagsusumbong", "parties",
)
_LABEL_RE = re.compile(r"(?:" + "|".join(_LABELS) + r")\s*:\s*([^;]+)", re.IGNORECASE)
_SPLIT_RE = re.compile(r",| and | & |/", re.IGNORECASE)
_TITLE_WORDS = {"mr", "mrs", "ms", "jr", "sr", "ii", "iii"}

# A bare one-word candidate is accepted as a name (a real narrative
# self-identifies with a nickname as often as a full name - "Ako po si ROSE"),
# but a placeholder an encoder leaves for an unnamed party must not be treated
# as one: masking "Unknown" or "Wala" would blank that ordinary word wherever
# it happens to appear in the narrative, not just the party it labels.
_NON_NAME_WORDS = {
    "unknown", "none", "n/a", "na", "n.a", "n.a.", "nil",
    "unidentified", "unspecified", "wala", "walang",
}


def extract_names(remarks):
    """Pull candidate person names out of the structured 'Complainant: X;
    Respondent: Y' style remarks the encoded blotter data already carries.

    Accepts 1-4 word candidates. A stricter 2-4 word floor originally excluded
    a bare first name or nickname entirely - which meant a header line like
    "Complainant: ROSE" still redacted correctly (bantay/narrative.py's
    _NAME_LINE substitution does not go through this function), but every
    later mention of "ROSE" inside the narration itself survived unmasked,
    since mask_names() only masks names this function actually returns
    (caught on the January 2026 field-evaluation batch, record #1). The
    _NON_NAME_WORDS stoplist and the length-2 floor below keep a placeholder
    like "Unknown" or short noise from being masked as if it were a name.
    """
    # pandas hands NaN (a float) for empty CSV cells, and every caller reads
    # remarks straight out of a DataFrame - so the type check belongs here, in
    # the shared function, not in each caller.
    if not remarks or not isinstance(remarks, str):
        return []
    names = []
    for match in _LABEL_RE.finditer(remarks):
        for candidate in _SPLIT_RE.split(match.group(1)):
            candidate = re.sub(r"\(.*?\)", "", candidate).strip().strip(".")
            if not candidate:
                continue
            word_count = len(candidate.split())
            if not (1 <= word_count <= 4):
                continue
            if word_count == 1 and (
                len(candidate) <= 2 or candidate.lower() in _NON_NAME_WORDS
            ):
                continue
            names.append(candidate)
    return names


def _name_tokens(name):
    """Individual first-name/surname tokens, so a later bare reference ('...Riniedo
    must pay...') gets masked even though only the full name appeared in remarks."""
    tokens = []
    for token in name.split():
        token = token.strip(".,")
        if len(token) <= 2 or token.lower() in _TITLE_WORDS:
            continue
        tokens.append(token)
    return tokens


def mask_names(narrative, remarks, placeholder="[PERSON]"):
    """Replace names identified in `remarks` with a placeholder (default
    [PERSON]) before this narrative is used for ML training, per the thesis's
    PII-anonymization requirement (doc section 3.2.1). Only affects the
    training feature text — the stored operational record keeps real names
    for barangay case handling. Masks full names first, then individual name
    tokens, so partial later references (surname- or first-name-only) don't
    leak through.

    The placeholder argument also lets bantay/narrative.py's redact_names
    reuse this same name-token logic for display-time redaction ([REDACTED])
    instead of duplicating it.
    """
    if not narrative:
        return narrative
    names = extract_names(remarks)
    replacements = set(names)
    for name in names:
        replacements.update(_name_tokens(name))

    masked = narrative
    for token in sorted(replacements, key=len, reverse=True):
        masked = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE).sub(placeholder, masked)
    return masked
