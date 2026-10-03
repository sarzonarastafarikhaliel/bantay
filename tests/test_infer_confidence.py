import numpy as np

from bantay.ml.infer import Classifier, decide_review_status, select_labels
from bantay.ml.registry import save_model


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.9, 0.2, 0.05]])


class DummyVectorizer:
    def transform(self, texts):
        # Return a minimal sparse-compatible object
        import scipy.sparse
        return scipy.sparse.csr_matrix([[0.0] * 10])


class DummyLabelEncoder:
    classes_ = np.array(["Noise Complaint", "Physical Injury/Altercation", "Theft/Robbery"])


def test_decide_review_status_threshold_scalar():
    assert decide_review_status(0.8, 0.6) == "accepted"
    assert decide_review_status(0.3, 0.6) == "needs_review"


def test_decide_review_status_dict_backward_compat():
    assert decide_review_status({"Theft/Robbery": 0.8}, 0.6) == "accepted"
    assert decide_review_status({"Theft/Robbery": 0.3}, 0.6) == "needs_review"
    assert decide_review_status({}, 0.6) == "needs_review"


def test_select_labels_returns_confident_labels():
    confidences = {"Theft/Robbery": 0.8, "Physical Injury/Altercation": 0.7, "Noise Complaint": 0.1}
    labels = select_labels(confidences, 0.6)
    assert "Theft/Robbery" in labels
    assert "Physical Injury/Altercation" in labels


def test_select_labels_falls_back_to_top_guess_when_none_confident():
    confidences = {"Theft/Robbery": 0.4, "Noise Complaint": 0.1}
    assert select_labels(confidences, 0.6) == ["Theft/Robbery"]


def test_select_labels_empty_when_no_confidences():
    assert select_labels({}, 0.6) == []


def test_classifier_returns_empty_when_no_model_trained(tmp_path):
    clf = Classifier(str(tmp_path))
    assert clf.classify("some narrative") == {}


def test_classifier_predicts_with_loaded_model(tmp_path):
    save_model(str(tmp_path), DummyModel(), DummyVectorizer(), DummyLabelEncoder())
    clf = Classifier(str(tmp_path))
    confidences = clf.classify("loud noise videoke")
    assert "Noise Complaint" in confidences
    assert confidences["Noise Complaint"] == pytest.approx(0.9)


def test_classifier_accepts_structured_fields(tmp_path):
    save_model(str(tmp_path), DummyModel(), DummyVectorizer(), DummyLabelEncoder())
    clf = Classifier(str(tmp_path))
    # Should not raise even with structured fields — no encoder stored
    confidences = clf.classify("noise complaint", date="2023-01-01", time="22:00", location_purok="Purok 3")
    assert isinstance(confidences, dict)


import pytest
