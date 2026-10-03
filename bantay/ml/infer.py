"""
Live classification using trained model + TF-IDF vectorizer + structured features.

Updated for Chapter 3 revised methodology:
  - Accepts structured fields (date, time, purok) in addition to narrative.
  - Combines TF-IDF + structured features using the same pipeline as training.
  - Returns single-label primary prediction with confidence score.
  - Falls back gracefully when structured encoder is absent (older models).
"""
import scipy.sparse

from .preprocess import (
    clean_text,
    extract_day_of_week,
    extract_month,
    extract_time_period,
    normalize_purok_label,
)
from .registry import load_model


def decide_review_status(confidence, threshold):
    """Confidence-based human review gate (§3.3.2).
    Single-label: accepted when the top prediction clears threshold.
    Also accepts a dict for backward compatibility with existing call sites.
    """
    if isinstance(confidence, dict):
        # Backward-compatible: any label clears threshold
        return "accepted" if any(v >= threshold for v in confidence.values()) else "needs_review"
    return "accepted" if confidence >= threshold else "needs_review"


def select_labels(confidences, threshold):
    """Legacy helper kept for backward compatibility with records.py call sites.
    Returns a list of label strings (single-element since we are now single-label).
    confidences may be a dict {label: prob} or the {label: prob} returned by classify().
    """
    if not confidences:
        return []
    confident = [label for label, conf in confidences.items() if conf >= threshold]
    if confident:
        return confident
    return [max(confidences, key=confidences.get)]


class Classifier:
    """Wraps trained model + TF-IDF vectorizer + label encoder + optional
    structured encoder for live single-label classification.

    Returns {} until a model has been trained via `bantay.ml.train`.
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model, self.vectorizer, self.label_encoder, self.structured_encoder = load_model(model_dir)

    def reload(self):
        """Re-reads artifacts from disk. Returns True if a model was found."""
        self.model, self.vectorizer, self.label_encoder, self.structured_encoder = load_model(self.model_dir)
        return self.model is not None

    def classify(self, narrative, date=None, time=None, location_purok=None):
        """Returns {label: probability} for the top predicted primary incident type.

        When structured features are available (trained encoder on disk),
        date/time/location_purok are incorporated into the feature vector alongside
        the TF-IDF representation of the narrative (§3.2.5, §3.3.1).

        For backward compatibility, calling classify(narrative) with no structured
        args still works — structured features will default to zeros/unknowns.
        """
        if not self.model or not self.vectorizer or not self.label_encoder:
            return {}
        cleaned = clean_text(narrative)
        if not cleaned:
            return {}

        X_tfidf = self.vectorizer.transform([cleaned])

        if self.structured_encoder:
            import numpy as np
            import pandas as pd
            row_df = pd.DataFrame([{
                "date": date or "",
                "time": time or "",
                "location_purok": location_purok or "",
            }])
            month = extract_month(date or "")
            dow = extract_day_of_week(date or "")
            time_period = extract_time_period(time or "")
            purok = normalize_purok_label(location_purok or "")

            numeric = scipy.sparse.csr_matrix([[float(month), float(dow)]])
            time_enc = self.structured_encoder["time_period_encoder"].transform([[time_period]])
            purok_enc = self.structured_encoder["purok_encoder"].transform([[purok]])

            X = scipy.sparse.hstack([X_tfidf, numeric, time_enc, purok_enc])
        else:
            X = X_tfidf

        # Per-model input adjustments, matching bantay.ml.train._prepare_X:
        # GBT needs dense input; MultinomialNB needs non-negative input.
        model_name = type(self.model).__name__
        if model_name == "GradientBoostingClassifier":
            X_input = X.toarray() if scipy.sparse.issparse(X) else X
        elif model_name == "MultinomialNB":
            X_input = X.copy()
            X_input.data[X_input.data < 0] = 0
        else:
            X_input = X

        proba = self.model.predict_proba(X_input)[0]
        return {
            label: float(p)
            for label, p in zip(self.label_encoder.classes_, proba)
        }
