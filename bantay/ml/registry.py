import os
import joblib

MODEL_FILENAME = "trained_model.joblib"
VECTORIZER_FILENAME = "vectorizer.joblib"
LABEL_BINARIZER_FILENAME = "label_binarizer.joblib"
STRUCTURED_ENCODER_FILENAME = "structured_encoder.joblib"


def save_model(model_dir, model, vectorizer, label_binarizer, structured_encoder=None):
    """Persist model artifacts to disk.

    structured_encoder: optional dict of fitted encoders produced by the
    structured-feature extraction pipeline (e.g. OneHotEncoder for purok /
    time_period).  Stored separately so infer.py can reconstruct the same
    combined feature matrix at inference time (§3.3.2).
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, MODEL_FILENAME))
    joblib.dump(vectorizer, os.path.join(model_dir, VECTORIZER_FILENAME))
    joblib.dump(label_binarizer, os.path.join(model_dir, LABEL_BINARIZER_FILENAME))
    if structured_encoder is not None:
        joblib.dump(structured_encoder, os.path.join(model_dir, STRUCTURED_ENCODER_FILENAME))


def load_model(model_dir):
    """Load model artifacts from disk.

    Returns (model, vectorizer, label_binarizer, structured_encoder).
    structured_encoder is None when no structured encoder file exists (e.g.
    models trained before this feature was added).
    """
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    vectorizer_path = os.path.join(model_dir, VECTORIZER_FILENAME)
    binarizer_path = os.path.join(model_dir, LABEL_BINARIZER_FILENAME)
    encoder_path = os.path.join(model_dir, STRUCTURED_ENCODER_FILENAME)

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(binarizer_path)):
        return None, None, None, None

    structured_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    return (
        joblib.load(model_path),
        joblib.load(vectorizer_path),
        joblib.load(binarizer_path),
        structured_encoder,
    )
