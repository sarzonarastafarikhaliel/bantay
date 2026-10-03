from bantay.ocr.reconcile import disagreements, fields_only, reconcile


def test_agree_auto_accepts():
    recon = reconcile({"date": "2023-11-19"}, {"date": "2023-11-19"})
    assert recon["date"] == {"value": "2023-11-19", "source": "both", "agree": True}


def test_disagree_prefers_regex_for_status_and_action():
    recon = reconcile({"status": "SETTLEMENT"}, {"status": "Settled"})
    assert recon["status"] == {"value": "Settled", "source": "regex", "agree": False}


def test_disagree_prefers_model_for_other_slots():
    recon = reconcile({"date": "2023-11-19"}, {"date": "2023-11-20"})
    assert recon["date"] == {"value": "2023-11-19", "source": "model", "agree": False}


def test_only_one_reader_answering_is_still_a_disagreement():
    """A field only one reader found is not a free pass - it goes to review
    same as an outright conflict, per OCR_ACCURACY_PLAN.md Lever 3's table."""
    recon = reconcile({"location_purok": "Purok 4B"}, {})
    assert recon["location_purok"] == {"value": "Purok 4B", "source": "model", "agree": False}

    recon = reconcile({}, {"location_purok": "Purok 4B"})
    assert recon["location_purok"] == {"value": "Purok 4B", "source": "regex", "agree": False}


def test_neither_reader_answering_is_a_blank_not_a_disagreement():
    recon = reconcile({"time": ""}, {"time": ""})
    assert recon["time"] == {"value": "", "source": "none", "agree": True}


def test_fields_only_drops_metadata():
    recon = reconcile({"date": "2023-11-19"}, {"date": "2023-11-19"})
    assert fields_only(recon) == {"date": "2023-11-19"}


def test_disagreements_excludes_blanks_and_agreements():
    recon = reconcile(
        {"date": "2023-11-19", "time": "", "status": "SETTLEMENT"},
        {"date": "2023-11-19", "time": "", "status": "Settled"})
    assert disagreements(recon) == ["status"]
