"""Guards for the two scoring helpers in tools/eval_fields.py.

Both had a bug that made the metric flatter itself, which is the failure mode
that matters here: a scorer that is quietly wrong produces a Chapter 4 table
nobody can catch by reading it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.eval_fields import norm_value, token_recall


def test_blank_gold_is_blank():
    """float NaN is truthy, so `str(nan or '')` was scoring empty cells as 'NAN'.

    This made action_taken - filled on 22 of 73 gold rows - report n=58.
    """
    import math

    assert norm_value(float("nan")) == ""
    assert norm_value(math.nan) == ""
    assert norm_value(None) == ""
    assert norm_value("") == ""
    assert norm_value("  ") == ""


def test_norm_value_ignores_case_and_punctuation():
    assert norm_value("Purok 4B") == norm_value("purok  4b")
    assert norm_value("For Hearing") == norm_value("FOR HEARING.")
    assert norm_value("Purok 4") != norm_value("Purok 4B")


def test_token_recall_excludes_pii_masks():
    """Masks must leave the denominator, or the score moves with how many people
    a page names rather than with how well the page was read."""
    exact, fuzzy, n = token_recall("NAGSADYA DITO SA BARANGAY", "[PERSON_7] NAGSADYA DITO SA BARANGAY")
    # 3, not 4: MIN_TOKEN_LEN drops "SA" along with the mask, on both sides.
    assert n == 3 and exact == 1.0 and fuzzy == 1.0


def test_token_recall_fuzzy_catches_misreads():
    exact, fuzzy, n = token_recall("NAG5ADYA DITO", "NAGSADYA DITO")
    assert n == 2
    assert exact == 0.5           # NAG5ADYA is not NAGSADYA
    assert fuzzy == 1.0           # but it is within the fuzzy floor


def test_token_recall_empty_gold_is_unscoreable():
    assert token_recall("anything", "") == (None, None, 0)
    assert token_recall("anything", "[PERSON_1]") == (None, None, 0)
