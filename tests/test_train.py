from sklearn.preprocessing import LabelEncoder

from bantay.ml.train import _group_metrics, _tier_metrics


def test_tier_metrics_groups_by_pnp_tier():
    labels = ["Theft", "Marital Relation", "Noise Complaint", "Theft"]
    encoder = LabelEncoder().fit(labels)
    y_true = encoder.transform(labels)
    # One miss: "Noise Complaint" (Non-Index Crime) predicted as "Marital Relation" (Non-Criminal Complaint)
    y_pred = encoder.transform(["Theft", "Marital Relation", "Marital Relation", "Theft"])

    tier_f1 = _tier_metrics(y_true, y_pred, encoder)

    assert tier_f1["Index Crime"] == 1.0
    assert tier_f1["Non-Index Crime"] == 0.0
    assert "Non-Criminal Complaint" in tier_f1


def test_group_metrics_groups_by_category_group():
    labels = ["Theft", "Marital Relation", "Noise Complaint", "Theft"]
    encoder = LabelEncoder().fit(labels)
    y_true = encoder.transform(labels)
    # One miss: "Noise Complaint" (Community Disturbance) predicted as "Marital Relation" (Family/Domestic)
    y_pred = encoder.transform(["Theft", "Marital Relation", "Marital Relation", "Theft"])

    group_f1 = _group_metrics(y_true, y_pred, encoder)

    assert group_f1["Criminal/Penal Code"] == 1.0
    assert group_f1["Community Disturbance"] == 0.0
    assert "Family/Domestic" in group_f1
