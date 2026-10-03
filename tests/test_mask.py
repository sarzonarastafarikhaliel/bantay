from bantay.ml.mask import extract_names, mask_names


def test_extract_names_from_labeled_remarks():
    remarks = "Complainant: Jane Doe; Respondents: John Smith, Mary Jones"
    assert extract_names(remarks) == ["Jane Doe", "John Smith", "Mary Jones"]


def test_extract_names_handles_tagalog_labels():
    remarks = "Nagsusumbong: Dong Joon Lee; Ipinagsusumbong: Nottwien Riniedo"
    assert extract_names(remarks) == ["Dong Joon Lee", "Nottwien Riniedo"]


def test_mask_names_masks_full_name_and_partial_later_references():
    narrative = "Nottwien Riniedo agreed to fix it. If he fails, Riniedo must pay."
    remarks = "Nagsusumbong: Dong Joon Lee; Ipinagsusumbong: Nottwien Riniedo"
    masked = mask_names(narrative, remarks)
    assert "Riniedo" not in masked
    assert "[PERSON]" in masked


def test_mask_names_handles_title_plus_surname():
    narrative = "Mr. Lee promised to pay Mrs. Toquero the amount owed."
    remarks = "Complainant: Segundina Toquero; Respondent: Donghyeob Lee"
    masked = mask_names(narrative, remarks)
    assert "Lee" not in masked
    assert "Toquero" not in masked


def test_mask_names_noop_without_remarks():
    assert mask_names("Some narrative text.", "") == "Some narrative text."
    assert mask_names("Some narrative text.", None) == "Some narrative text."


def test_mask_names_noop_without_narrative():
    assert mask_names("", "Complainant: Jane Doe") == ""


def test_extract_names_accepts_a_bare_one_word_name():
    # A narrative self-identifying with just a nickname ("Ako po si ROSE...")
    # must not be dropped the way the old 2-4 word floor dropped it - see
    # bantay/ml/mask.py's extract_names docstring for the field case that
    # caught this (record #1 of the January 2026 evaluation batch).
    remarks = "Complainant: ROSE; Respondent: Ronie Luinan"
    assert extract_names(remarks) == ["ROSE", "Ronie Luinan"]


def test_mask_names_masks_a_bare_one_word_name_in_the_narration():
    narrative = "Ako po si ROSE, inirereklamo ko si Ronie Luinan."
    remarks = "Complainant: ROSE; Respondent: Ronie Luinan"
    masked = mask_names(narrative, remarks)
    assert "ROSE" not in masked
    assert "Luinan" not in masked


def test_extract_names_ignores_placeholder_single_words():
    # A one-word candidate that is a placeholder for an unnamed party, not an
    # actual name, must not be masked - doing so would blank that ordinary
    # word everywhere it appears in the narrative, not just where it labels
    # a party.
    remarks = "Complainant: Unknown; Respondent: Wala"
    assert extract_names(remarks) == []
