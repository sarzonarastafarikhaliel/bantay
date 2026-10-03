"""
Text and structured-feature preprocessing.

Shared by train.py and infer.py so that training and live classification
always see identical preprocessing (§3.2.5 – Transformation, §3.3.2 – Model
Deployment).  Changes here must be reflected in both callers.
"""
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Stop-word list (Chapter 3 §3.2.5 – explicit tokenisation stop-word step)
# ---------------------------------------------------------------------------
# Barangay narratives are typically Taglish, so we cover both English and
# Filipino function words.  This list is used in clean_text() BEFORE
# TF-IDF vectorisation, fulfilling the explicit stop-word-removal step
# required by §3.2.5 in addition to the TF-IDF-level stop-word filter.
TAGALOG_STOP_WORDS = {
    "ako", "ikaw", "ka", "siya", "kami", "tayo", "kayo", "sila",
    "ang", "ng", "nang", "sa", "na", "at", "ay", "mga",
    "ito", "iyon", "iyan", "dito", "doon", "diyan", "rito", "riyan", "roon",
    "kung", "kapag", "dahil", "para", "upang",
    "hindi", "wala", "meron", "mayroon", "may", "rin", "din",
    "lang", "lamang", "po", "opo", "ba", "naman", "pa", "pala", "kasi",
    "ni", "nina", "si", "sina", "kay", "kina",
    "akin", "iyo", "kanya", "amin", "inyo", "kanila", "atin",
    "sino", "ano", "saan", "kailan", "paano", "bakit", "alin",
    "yung", "yun",
}

# Minimal English stop-words (subset of sklearn's list) kept inline so this
# module has no sklearn import dependency at inference time.
ENGLISH_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "me", "my",
    "not", "no", "nor", "so", "yet", "both", "either", "each", "any",
    "all", "there", "here", "when", "where", "who", "what", "which", "how",
    "then", "than", "too", "very", "just", "also", "up", "out", "over",
    "after", "before", "between", "into", "through", "during", "about",
    "against", "as", "such", "other", "more", "some", "been",
}

ALL_STOP_WORDS = TAGALOG_STOP_WORDS | ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Text preprocessing (§3.2.5)
# ---------------------------------------------------------------------------

def tokenize(text):
    """Lowercase, strip punctuation, split into tokens (words)."""
    if not text:
        return []
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip().split()


def remove_stop_words(tokens):
    """Explicit pre-tokenisation stop-word removal step (§3.2.5)."""
    return [t for t in tokens if t not in ALL_STOP_WORDS]


def clean_text(text):
    """Full pipeline: lowercase → punctuation strip → tokenise →
    remove stop-words → rejoin.

    Used as the shared text-cleaning function for both training and inference.
    """
    tokens = remove_stop_words(tokenize(text))
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Structured feature extraction (§3.2.5 Transformation, §3.3.1)
# ---------------------------------------------------------------------------
# These features are combined with TF-IDF text features before model training,
# consistent with Balahadia et al. (2020) which found date/time/place
# significantly related to crime occurrence.

_TIME_PERIODS = {
    "Morning":    (6,  12),   # 06:00 – 11:59
    "Afternoon":  (12, 18),   # 12:00 – 17:59
    "Evening":    (18, 22),   # 18:00 – 21:59
    "Late Night": (22, 30),   # 22:00 – 05:59 (30 = wrap-around sentinel)
}


def extract_month(date_str):
    """Return integer month (1–12) or 0 if unparseable."""
    if not date_str:
        return 0
    try:
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d").month
    except ValueError:
        return 0


def extract_day_of_week(date_str):
    """Return integer day-of-week (0=Monday … 6=Sunday) or -1 if unparseable."""
    if not date_str:
        return -1
    try:
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d").weekday()
    except ValueError:
        return -1


def extract_time_period(time_str):
    """Return one of 'Morning', 'Afternoon', 'Evening', 'Late Night', or
    'Unknown' if the time field is empty or unreadable."""
    if not time_str:
        return "Unknown"
    try:
        t = datetime.strptime(str(time_str).strip()[:5], "%H:%M")
        h = t.hour
        # Late night wraps past midnight to 05:59
        if h >= 22 or h < 6:
            return "Late Night"
        for label, (start, end) in _TIME_PERIODS.items():
            if label == "Late Night":
                continue
            if start <= h < end:
                return label
    except ValueError:
        pass
    return "Unknown"


def normalize_purok_label(raw):
    """Normalise a purok string to a stable label for one-hot encoding.
    Falls back to 'Unspecified'."""
    if not raw:
        return "Unspecified"
    return str(raw).strip()
