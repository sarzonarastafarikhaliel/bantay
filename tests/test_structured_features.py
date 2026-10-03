"""
Tests for structured feature extraction functions (§3.2.5 Transformation, §3.3.1).

Covers: extract_month, extract_day_of_week, extract_time_period, normalize_purok_label,
and the combined build_structured_features() pipeline from train.py.
"""
import numpy as np
import pandas as pd
import pytest

from bantay.ml.preprocess import (
    extract_day_of_week,
    extract_month,
    extract_time_period,
    normalize_purok_label,
)


# ---------------------------------------------------------------------------
# Month extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date_str, expected", [
    ("2023-01-15", 1),
    ("2023-06-01", 6),
    ("2023-12-31", 12),
])
def test_extract_month_valid(date_str, expected):
    assert extract_month(date_str) == expected


@pytest.mark.parametrize("bad", ["", None, "not-a-date", "01/15/2023"])
def test_extract_month_invalid_returns_zero(bad):
    assert extract_month(bad) == 0


# ---------------------------------------------------------------------------
# Day of week extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date_str, expected_dow", [
    ("2023-01-02", 0),   # Monday
    ("2023-01-03", 1),   # Tuesday
    ("2023-01-07", 5),   # Saturday
    ("2023-01-08", 6),   # Sunday
])
def test_extract_day_of_week_valid(date_str, expected_dow):
    assert extract_day_of_week(date_str) == expected_dow


@pytest.mark.parametrize("bad", ["", None, "not-a-date"])
def test_extract_day_of_week_invalid_returns_minus_one(bad):
    assert extract_day_of_week(bad) == -1


# ---------------------------------------------------------------------------
# Time period classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time_str, expected", [
    ("06:00", "Morning"),
    ("09:30", "Morning"),
    ("11:59", "Morning"),
    ("12:00", "Afternoon"),
    ("15:00", "Afternoon"),
    ("17:59", "Afternoon"),
    ("18:00", "Evening"),
    ("21:59", "Evening"),
    ("22:00", "Late Night"),
    ("23:59", "Late Night"),
    ("00:00", "Late Night"),
    ("03:00", "Late Night"),
    ("05:59", "Late Night"),
])
def test_extract_time_period_boundaries(time_str, expected):
    assert extract_time_period(time_str) == expected


@pytest.mark.parametrize("bad", ["", None, "not-a-time", "25:00"])
def test_extract_time_period_invalid_returns_unknown(bad):
    assert extract_time_period(bad) == "Unknown"


# ---------------------------------------------------------------------------
# Purok label normalization
# ---------------------------------------------------------------------------

def test_normalize_purok_label_strips_whitespace():
    assert normalize_purok_label("  Purok 3  ") == "Purok 3"


def test_normalize_purok_label_none_returns_unspecified():
    assert normalize_purok_label(None) == "Unspecified"


def test_normalize_purok_label_empty_returns_unspecified():
    assert normalize_purok_label("") == "Unspecified"


# ---------------------------------------------------------------------------
# Combined structured feature pipeline (integration)
# ---------------------------------------------------------------------------

def test_build_structured_features_shape():
    """build_structured_features should return a sparse matrix with correct row count."""
    from bantay.ml.train import build_structured_features

    df = pd.DataFrame([
        {"date": "2023-01-02", "time": "09:00", "location_purok": "Purok 1"},
        {"date": "2023-06-15", "time": "14:30", "location_purok": "Purok 3"},
        {"date": "2023-11-20", "time": "22:00", "location_purok": "Purok 2"},
    ])

    structured, purok_enc, time_enc = build_structured_features(df, fit=True)
    assert structured.shape[0] == 3
    # At minimum: 2 numeric (month, day_of_week) + 5 time_period OHE + purok OHE
    assert structured.shape[1] >= 7


def test_build_structured_features_inference_uses_fitted_encoders():
    """Re-using fitted encoders (fit=False) should not error on new data."""
    from bantay.ml.train import build_structured_features

    df_train = pd.DataFrame([
        {"date": "2023-01-02", "time": "09:00", "location_purok": "Purok 1"},
        {"date": "2023-06-15", "time": "22:00", "location_purok": "Purok 3"},
    ])
    _, purok_enc, time_enc = build_structured_features(df_train, fit=True)

    df_test = pd.DataFrame([
        {"date": "2023-08-01", "time": "13:00", "location_purok": "Purok 2"},
    ])
    structured, _, _ = build_structured_features(df_test, purok_encoder=purok_enc,
                                                  time_period_encoder=time_enc, fit=False)
    assert structured.shape[0] == 1


def test_build_structured_features_unknown_purok_handled():
    """Unknown purok at inference should not raise (handle_unknown='ignore')."""
    from bantay.ml.train import build_structured_features

    df_train = pd.DataFrame([
        {"date": "2023-01-02", "time": "09:00", "location_purok": "Purok 1"},
    ])
    _, purok_enc, time_enc = build_structured_features(df_train, fit=True)

    df_test = pd.DataFrame([
        {"date": "2023-08-01", "time": "13:00", "location_purok": "Purok 99"},
    ])
    structured, _, _ = build_structured_features(df_test, purok_encoder=purok_enc,
                                                  time_period_encoder=time_enc, fit=False)
    # Should not raise; unknown purok → all-zero row in purok OHE block
    assert structured.shape[0] == 1
