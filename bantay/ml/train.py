"""
ML training pipeline — revised for Chapter 3 alignment.

Key changes vs. original:
  - Single-label classification (incident_type_primary) instead of multi-label.
  - Combined feature matrix: TF-IDF text features + structured features
    (month, day_of_week, time_period one-hot, purok one-hot), via
    scipy.sparse.hstack (§3.2.5, §3.3.1 — Balahadia et al., 2020).
  - Gradient Boosted Trees added as sixth model candidate (Asor et al., 2022).
  - Optional SMOTE oversampling for class imbalance (§3.2.5, §3.3.2).
  - Structured encoder (OneHotEncoder) saved alongside model artifacts.
  - Per-label confusion matrices shown for all models, not just best.
  - k-fold CV results stored for dashboard display.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .mask import mask_names
from .preprocess import (
    clean_text,
    extract_day_of_week,
    extract_month,
    extract_time_period,
    normalize_purok_label,
)
from .registry import save_model
from .verify_pnp_classification import verify as verify_pnp_classification
from ..normalize import FALLBACK_CATEGORY, clean_row, get_category_group, get_pnp_classification

# ---------------------------------------------------------------------------
# Stop-word list
# ---------------------------------------------------------------------------
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
STOP_WORDS = list(ENGLISH_STOP_WORDS.union(TAGALOG_STOP_WORDS))

# ---------------------------------------------------------------------------
# Model candidates — single-label (primary incident type is one canonical
# label per record, so standard multi-class classifiers are used directly).
# GradientBoostingClassifier doesn't support sparse matrices natively; we wrap
# it in a pipeline-compatible way by converting to dense inside _build_models().
# ---------------------------------------------------------------------------
TIME_PERIOD_CATEGORIES = ["Morning", "Afternoon", "Evening", "Late Night", "Unknown"]


def _prepare_X(X, name):
    """Per-model input adjustments before fit/predict.

    - Gradient Boosted Trees doesn't support sparse input; convert to dense.
    - MultinomialNB requires non-negative input, but the combined feature
      matrix can carry negative values (extract_day_of_week's -1 "unparseable
      date" sentinel). Clipped to 0 for this model only.
    # ponytail: clipping collides "unparseable date" with day_of_week==0
    # (Monday) for MultinomialNB only; fine since it's one candidate among
    # six, revisit if it becomes the consistent best model.
    """
    if name == "Gradient Boosted Trees":
        return X.toarray()
    if name == "Multinomial Naive Bayes":
        X = X.copy()
        X.data[X.data < 0] = 0
        return X
    return X


def _tier_metrics(y_true, y_pred, label_encoder):
    """Per-PNP-tier F1 (Index Crime / Non-Index Crime / Non-Criminal
    Complaint / Unclassified), computed from the same true/predicted label
    arrays already used for the per-model accuracy/precision/recall/F1 above.
    Surfaces whether the model is reliable specifically on Index Crimes
    (rare, high-stakes) vs. Non-Criminal Complaints (frequent, low-stakes).
    """
    true_tiers = [get_pnp_classification(label)[0] for label in label_encoder.inverse_transform(y_true)]
    pred_tiers = [get_pnp_classification(label)[0] for label in label_encoder.inverse_transform(y_pred)]
    tiers_present = sorted(set(true_tiers) | set(pred_tiers))
    if not tiers_present:
        return {}
    _, _, f1, _ = precision_recall_fscore_support(
        true_tiers, pred_tiers, labels=tiers_present, average=None, zero_division=0
    )
    return {tier: round(float(score), 4) for tier, score in zip(tiers_present, f1)}


def _group_metrics(y_true, y_pred, label_encoder):
    """Per-Category-Group F1 (Civil Dispute / Criminal-Penal Code /
    Property-Lost Items / Family-Domestic / Community Disturbance /
    Administrative-Referral / Catch-all), computed the same way as
    _tier_metrics above but keyed by the CSV-derived category group
    instead of the PNP legal tier.
    """
    true_groups = [get_category_group(label) for label in label_encoder.inverse_transform(y_true)]
    pred_groups = [get_category_group(label) for label in label_encoder.inverse_transform(y_pred)]
    groups_present = sorted(set(true_groups) | set(pred_groups))
    if not groups_present:
        return {}
    _, _, f1, _ = precision_recall_fscore_support(
        true_groups, pred_groups, labels=groups_present, average=None, zero_division=0
    )
    return {group: round(float(score), 4) for group, score in zip(groups_present, f1)}


def _subset_metrics(y_true, y_pred, mask, label_encoder):
    """Accuracy/precision/recall/F1 restricted to a subset of the test fold.

    Used to report real-record performance separately from synthetic-record
    performance. With a ~28:1 synthetic:real corpus, a single blended number is
    dominated by the templates and says almost nothing about the deployed task.
    """
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if not len(y_true):
        return {}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tier_f1": _tier_metrics(y_true, y_pred, label_encoder),
        "group_f1": _group_metrics(y_true, y_pred, label_encoder),
    }


def _build_models():
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Support Vector Machine": SVC(kernel="linear", probability=True, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200),
        "Gradient Boosted Trees": GradientBoostingClassifier(n_estimators=100, max_depth=4),
    }


# ---------------------------------------------------------------------------
# Structured feature builder
# ---------------------------------------------------------------------------

def build_structured_features(df, purok_encoder=None, time_period_encoder=None, fit=True):
    """Build a sparse structured-feature matrix from date/time/purok columns.

    Returns (sparse_matrix, purok_encoder, time_period_encoder).
    When fit=True the encoders are fitted on df (training); fit=False uses
    pre-fitted encoders (inference).
    """
    months = df["date"].apply(extract_month).values.reshape(-1, 1)          # (N, 1)
    days_of_week = df["date"].apply(extract_day_of_week).values.reshape(-1, 1)  # (N, 1)

    time_periods = df["time"].apply(extract_time_period).values.reshape(-1, 1)
    puroks = df["location_purok"].apply(normalize_purok_label).values.reshape(-1, 1)

    if fit:
        time_period_encoder = OneHotEncoder(
            categories=[TIME_PERIOD_CATEGORIES], handle_unknown="ignore", sparse_output=True
        )
        time_period_encoder.fit(time_periods)
        purok_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        purok_encoder.fit(puroks)

    time_period_ohe = time_period_encoder.transform(time_periods)   # (N, 5)
    purok_ohe = purok_encoder.transform(puroks)                      # (N, P)

    numeric = scipy.sparse.csr_matrix(
        np.hstack([months, days_of_week]).astype(float)
    )  # (N, 2)

    structured = scipy.sparse.hstack([numeric, time_period_ohe, purok_ohe])
    return structured, purok_encoder, time_period_encoder


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _normalize_columns(df):
    """
    Normalize DataFrame column names to lowercase snake_case and apply
    known column-name aliases so various CSV exports all converge to the
    same internal names expected by clean_row() and the ML pipeline.
    """
    # Lowercase + replace spaces and slashes with underscores
    df.columns = [
        col.strip().lower().replace("/", "_").replace(" ", "_")
        for col in df.columns
    ]
    # Known alias map (source_name → internal_name)
    aliases = {
        "narrative_summary":      "narrative",
        "summary":                "narrative",
        "incident_type":          "incident_type_primary",   # old single-label column
        "location_purok":         "location_purok",          # already correct
        "location":               "location_purok",
        "purok":                  "location_purok",
        "batch_number":           "batch_number",
        "batch":                  "batch_number",
        "source_file_page":       "source_file_page",
        "source_file_page_":      "source_file_page",
        "date_encoded":           "date_encoded",
        "encoded_by":             "encoded_by",
        "readability":            "readability",
        "include_in_ml":          "include_in_ml",
        "action_taken":           "action_taken",
        "persons_involved_masked": "persons_involved_masked",
        "remarks":                "remarks",
        "status":                 "status",
    }
    rename = {k: v for k, v in aliases.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)
    return df


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df = pd.DataFrame(clean_row(row) for row in df.to_dict(orient="records"))
    if "include_in_ml" in df.columns:
        # Accept rows explicitly marked Yes/True, as well as synthetic rows
        # (Pending Review / N/A (synthetic)) which are labelled and usable.
        # Only hard-exclude rows explicitly marked No/False/Excluded.
        excluded = {"no", "false", "0", "excluded", "exclude"}
        df = df[~df["include_in_ml"].astype(str).str.lower().isin(excluded)]

    # Support both old (incident_type) and new (incident_type_primary) column names
    if "incident_type_primary" not in df.columns and "incident_type" in df.columns:
        df["incident_type_primary"] = df["incident_type"].apply(
            lambda v: v.split("|")[0].strip() if v else FALLBACK_CATEGORY
        )
    elif "incident_type_primary" not in df.columns:
        df["incident_type_primary"] = FALLBACK_CATEGORY

    # Ensure required columns exist
    for col in ("date", "time", "location_purok"):
        if col not in df.columns:
            df[col] = None

    # Semi-synthetic corpus columns (see PIPELINE.md). Absent for legacy CSVs,
    # in which case the pipeline falls back to its old random-split behaviour.
    if "_split" not in df.columns:
        df["_split"] = None
    if "_provenance" not in df.columns:
        df["_provenance"] = "real"

    df = df.dropna(subset=["narrative", "incident_type_primary"])
    df = df[df["narrative"].str.strip() != ""]
    return df.reset_index(drop=True)


def merge_rare_categories(df, min_count=3, other_label=FALLBACK_CATEGORY):
    """Fold primary labels appearing in fewer than min_count rows into fallback."""
    df = df.copy()
    counts = df["incident_type_primary"].value_counts()
    rare = set(counts[counts < min_count].index)
    df["incident_type_primary"] = df["incident_type_primary"].apply(
        lambda v: other_label if v in rare else v
    )
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _has_prebuilt_splits(df):
    """True when the input carries the semi-synthetic corpus' _split column.

    The corpus splits synthetic rows by core clause variant so no sentence
    skeleton is shared between train and test (PIPELINE.md §5). Re-splitting it
    randomly puts the same template on both sides and produces an F1 that
    measures memorisation, not generalisation — so when the column is present
    it wins over train_test_split, always.
    """
    if "_split" not in df.columns:
        return False
    present = set(df["_split"].dropna().astype(str))
    return {"train", "test"}.issubset(present)


def train_and_compare(df, min_category_count=3, cv_folds=5, use_smote=False):
    prebuilt = _has_prebuilt_splits(df)

    if prebuilt:
        # The corpus is already balanced at 60 rows per canonical type, so
        # rare-class folding can only destroy classes it was meant to protect.
        print("Pre-built splits detected — skipping merge_rare_categories.")
        if use_smote:
            print("SMOTE disabled: corpus is already class-balanced, and "
                  "resampling would break row-to-split alignment.")
            use_smote = False
    else:
        df = merge_rare_categories(df, min_count=min_category_count)

    masked = df.apply(lambda r: mask_names(r["narrative"], r.get("remarks", "")), axis=1)
    texts = masked.apply(clean_text)

    labels = df["incident_type_primary"].tolist()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=STOP_WORDS)
    X_tfidf = vectorizer.fit_transform(texts)

    # Build structured features and combine with TF-IDF (§3.2.5, §3.3.1)
    structured, purok_encoder, time_period_encoder = build_structured_features(df, fit=True)

    X = scipy.sparse.hstack([X_tfidf, structured])

    structured_encoder = {
        "purok_encoder": purok_encoder,
        "time_period_encoder": time_period_encoder,
    }

    # Optional SMOTE (§3.2.5, §3.3.2)
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            X_dense = X.toarray()
            sm = SMOTE(random_state=42)
            X_dense, y = sm.fit_resample(X_dense, y)
            X = scipy.sparse.csr_matrix(X_dense)
            print("SMOTE applied.")
        except ImportError:
            print("imbalanced-learn not installed — skipping SMOTE.")

    min_class_count = int(np.bincount(y).min()) if len(y) else 0
    use_cv = (min_class_count < 2 or len(df) < 30) and not prebuilt

    models = _build_models()
    results = []
    best_name, best_model, best_score = None, None, -1.0

    if use_cv:
        folds = max(2, min(cv_folds, min_class_count))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        for name, model in models.items():
            try:
                X_fit = _prepare_X(X, name)
                y_pred = cross_val_predict(model, X_fit, y, cv=cv)
                accuracy = accuracy_score(y, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y, y_pred, average="weighted", zero_division=0
                )
                results.append({
                    "model": name,
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "cv_folds": folds,
                    "tier_f1": _tier_metrics(y, y_pred, label_encoder),
                    "group_f1": _group_metrics(y, y_pred, label_encoder),
                })
                if f1 > best_score:
                    best_score, best_name, best_model = f1, name, model
            except Exception as exc:
                print(f"  {name} failed: {exc}")
        if best_model is not None:
            X_fit = _prepare_X(X, best_name)
            best_model.fit(X_fit, y)
    else:
        if prebuilt:
            split = df["_split"].astype(str).to_numpy()
            train_mask = split == "train"
            test_mask = split == "test"
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            # Real records inside the test fold — the only rows that say
            # anything about performance on actual logbook language.
            real_test_mask = (
                df.loc[test_mask, "_provenance"].astype(str).to_numpy() == "real"
            )
            print(f"Using pre-built splits: {train_mask.sum()} train / "
                  f"{test_mask.sum()} test ({real_test_mask.sum()} real).")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            real_test_mask = None

        for name, model in models.items():
            try:
                X_tr = _prepare_X(X_train, name)
                X_te = _prepare_X(X_test, name)
                model.fit(X_tr, y_train)
                preds = model.predict(X_te)
                accuracy = accuracy_score(y_test, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, preds, average="weighted", zero_division=0
                )
                entry = {
                    "model": name,
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "cv_folds": None,
                    "tier_f1": _tier_metrics(y_test, preds, label_encoder),
                    "group_f1": _group_metrics(y_test, preds, label_encoder),
                }

                # Provenance breakout. The synthetic figure is a diagnostic;
                # the real figure is the one that belongs in the write-up.
                if real_test_mask is not None and real_test_mask.any():
                    entry["real_only"] = _subset_metrics(
                        y_test, preds, real_test_mask, label_encoder
                    )
                    entry["synthetic_only"] = _subset_metrics(
                        y_test, preds, ~real_test_mask, label_encoder
                    )

                results.append(entry)
                # Per-label confusion matrices for all models
                cm = confusion_matrix(y_test, preds)
                print(f"\nConfusion matrix — {name}:")
                print(cm)
                # Model selection is driven by the real-record subset when one
                # exists. Ranking on the synthetic majority would pick whichever
                # model best reproduces the phrase bank.
                score = entry.get("real_only", {}).get("f1", f1)
                if score > best_score:
                    best_score, best_name, best_model = score, name, model
            except Exception as exc:
                print(f"  {name} failed: {exc}")

    return best_name, best_model, vectorizer, label_encoder, structured_encoder, results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train and compare BANTAY incident classifiers.")
    parser.add_argument("--input", required=True, help="Path to encoded incidents CSV")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"),
    )
    parser.add_argument("--min-category-count", type=int, default=3)
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE for class imbalance")
    args = parser.parse_args()

    mismatches = verify_pnp_classification()
    if mismatches:
        print(f"WARNING: {len(mismatches)} PNP classification citation(s) have drifted from "
              f"data/pnp_legal_reference.json — run `python -m bantay.ml.verify_pnp_classification` for details.")

    df = load_dataset(args.input)
    if df.empty:
        print("No usable rows. Aborting.")
        sys.exit(1)

    best_name, best_model, vectorizer, label_encoder, structured_encoder, results = train_and_compare(
        df, args.min_category_count, use_smote=args.use_smote
    )

    print("\nModel comparison:")
    for r in results:
        print(r)

    if results and "real_only" in results[0]:
        print("\nReal-record test performance (the number to report):")
        for r in results:
            ro = r.get("real_only", {})
            print(f"  {r['model']:24s} n={ro.get('n', 0):3d}  "
                  f"acc={ro.get('accuracy', 0):.4f}  f1={ro.get('f1', 0):.4f}")
        print("\nNOTE: the top-level 'f1' above is dominated by synthetic rows "
              "and is a diagnostic only. For the headline metric run "
              "leave-one-out over the 73 real records "
              "(`python run_pipeline.py --loocv`).")

    print(f"\nSelected best model: {best_name}")

    save_model(args.model_dir, best_model, vectorizer, label_encoder, structured_encoder)
    print(f"Saved to {args.model_dir}")

    # Write comparison results for dashboard display
    results_path = os.path.join(args.model_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"Comparison results saved to {results_path}")


if __name__ == "__main__":
    main()
