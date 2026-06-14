import numpy as np

from empty_scaffold.metrics import classification_summary, norm_pr, rel_mae


def test_norm_pr_and_average_precision_ties():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(norm_pr(y, scores)) < 1e-12

    y = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.8, 0.8, 0.4, 0.4, 0.2, 0.2])
    summary = classification_summary(y, scores)
    assert abs(summary["average_precision"] - 0.5) < 1e-12
    assert abs(summary["normPR"]) < 1e-12


def test_relmae_invariant_to_rescaling():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.1, 1.8, 3.2])
    assert abs(rel_mae(y, pred) - rel_mae(10 * y, 10 * pred)) < 1e-12
