"""tools/pool_188.py: pooling and the paired statistics behind review item R2."""
import math

import numpy as np
import pandas as pd
import pytest

from tools.pool_188 import (bootstrap_ci, holm, mcnemar_cc, mcnemar_exact,
                            paired_bootstrap_diff, pool, wilson)


def _toy():
    gold = pd.DataFrame({"source_file": ["c.jpg", "a.jpg", "b.jpg"],
                         "gold_type": ["Theft", "Physical Injury", "Ejectment"]})
    split = pd.DataFrame({"source_file": ["a.jpg", "b.jpg", "c.jpg"],
                          "split": ["dev", "dev", "holdout"]})
    preds = {
        "m1": pd.DataFrame({"source_file": ["a.jpg", "b.jpg", "c.jpg"],
                            "pred_type": ["Physical Injury", "Theft", None],
                            "pred_group": ["Criminal/Penal Code", "Criminal/Penal Code", None]}),
        "m2": pd.DataFrame({"source_file": ["c.jpg", "b.jpg", "a.jpg"],
                            "pred_type": ["Theft", "Ejectment", "Slight Physical Injury"],
                            "pred_group": ["Criminal/Penal Code", "Civil Dispute",
                                           "Criminal/Penal Code"]}),
    }
    return gold, split, preds


def test_pool_builds_wide_table_with_split_and_correctness():
    out = pool(*_toy())
    assert list(out.columns[:4]) == ["record_id", "split", "gold_type", "gold_group"]
    assert list(out["record_id"]) == ["a.jpg", "b.jpg", "c.jpg"]
    assert list(out["split"]) == ["dev", "dev", "holdout"]
    assert list(out["gold_group"]) == ["Criminal/Penal Code", "Civil Dispute",
                                       "Criminal/Penal Code"]
    assert list(out["m1_correct_type"]) == [True, False, False]   # unparsed reply is wrong
    assert list(out["m1_correct_group"]) == [True, False, False]
    assert list(out["m2_correct_type"]) == [False, True, True]    # input row order ignored
    assert list(out["m2_correct_group"]) == [True, True, True]    # wrong type, right group
    assert pd.isna(out.loc[2, "m1_predicted_type"])


def test_pool_rejects_a_model_missing_a_record():
    gold, split, preds = _toy()
    preds["m1"] = preds["m1"].iloc[:2]
    with pytest.raises(AssertionError):
        pool(gold, split, preds)


def test_wilson_matches_hand_computation():
    # 129/188, z = 1.959964: p = 0.686170, z^2/n = 0.020433, d = 1.020433,
    # centre = (p + z^2/2n)/d = 0.682442, half = z*sqrt(p(1-p)/n + z^2/4n^2)/d = 0.065772
    lo, hi = wilson(129, 188)
    assert lo == pytest.approx(0.61667, abs=1e-4)
    assert hi == pytest.approx(0.74821, abs=1e-4)
    # the reviewer's figures, to one decimal
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (61.7, 74.8)
    lo, hi = wilson(114, 188)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (53.5, 67.3)


def test_wilson_edges_stay_inside_unit_interval():
    # 0/10: centre = half = (z^2/2n)/(1 + z^2/n) = 0.138766, so hi = 0.277533
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.27753, abs=1e-4)
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12) and lo == pytest.approx(0.72247, abs=1e-4)
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_mcnemar_exact_known_discordants():
    # b=1, c=9: two-sided p = 2 * (C(10,0) + C(10,1)) / 2^10 = 22/1024
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(9, 1) == pytest.approx(22 / 1024)
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_mcnemar_continuity_corrected_known_discordants():
    # (|35 - 20| - 1)^2 / 55 = 196/55; thesis Table J3 dev row: (|33 - 15| - 1)^2 / 48 = 6.02
    chi, p = mcnemar_cc(35, 20)
    assert chi == pytest.approx(196 / 55)
    assert p == pytest.approx(0.0591, abs=1e-4)
    assert mcnemar_cc(33, 15)[0] == pytest.approx(6.02, abs=5e-3)
    assert mcnemar_cc(0, 0) == (0.0, 1.0)


def test_holm_known_vector():
    # sorted 0.01, 0.02, 0.03, 0.04 -> 4*0.01, 3*0.02, max(2*0.03, 0.06), max(0.04, 0.06)
    assert holm([0.03, 0.01, 0.04, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_is_seeded_and_brackets_the_mean():
    x = np.array([1] * 30 + [0] * 10)
    assert bootstrap_ci(x, n_boot=2000, seed=1) == bootstrap_ci(x, n_boot=2000, seed=1)
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=1)
    assert lo < 0.75 < hi
    assert bootstrap_ci(np.ones(20), n_boot=500) == (1.0, 1.0)


def test_paired_bootstrap_diff_of_identical_models_is_zero():
    a = np.array([1, 0, 1, 1, 0, 1])
    assert paired_bootstrap_diff(a, a, n_boot=500) == (0.0, 0.0, 0.0)
    assert paired_bootstrap_diff(np.ones(10), np.zeros(10), n_boot=500) == (1.0, 1.0, 1.0)
