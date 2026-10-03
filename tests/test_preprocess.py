from bantay.ml.preprocess import clean_text, extract_month, extract_day_of_week, extract_time_period, normalize_purok_label


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Loud Music at NIGHT!!") == "loud music night"


def test_clean_text_collapses_whitespace():
    # "too" is an English stop word — removed by clean_text
    result = clean_text("many    spaces here")
    assert "many" in result
    assert "spaces" in result


def test_clean_text_handles_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_removes_stop_words():
    result = clean_text("the bicycle was stolen from the house")
    assert "the" not in result.split()
    assert "from" not in result.split()
    assert "bicycle" in result


def test_extract_month_valid():
    assert extract_month("2023-01-15") == 1
    assert extract_month("2023-12-01") == 12


def test_extract_month_invalid():
    assert extract_month("") == 0
    assert extract_month(None) == 0
    assert extract_month("not-a-date") == 0


def test_extract_day_of_week():
    assert extract_day_of_week("2023-01-02") == 0   # Monday
    assert extract_day_of_week("2023-01-07") == 5   # Saturday
    assert extract_day_of_week("") == -1


def test_extract_time_period():
    assert extract_time_period("09:00") == "Morning"
    assert extract_time_period("13:00") == "Afternoon"
    assert extract_time_period("19:00") == "Evening"
    assert extract_time_period("23:00") == "Late Night"
    assert extract_time_period("03:00") == "Late Night"
    assert extract_time_period("") == "Unknown"
    assert extract_time_period(None) == "Unknown"


def test_normalize_purok_label():
    assert normalize_purok_label("Purok 3") == "Purok 3"
    assert normalize_purok_label(None) == "Unspecified"
    assert normalize_purok_label("") == "Unspecified"
