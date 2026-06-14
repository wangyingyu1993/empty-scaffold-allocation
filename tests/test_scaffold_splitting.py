import pandas as pd

from empty_scaffold.scaffold_splitting import (
    assign_scaffold_folds,
    no_scaffold_fold,
    scaffold_kfolds_for_n,
)


def test_scaffold_kfold_rule():
    assert scaffold_kfolds_for_n(1000) == 5
    assert scaffold_kfolds_for_n(1001) == 7
    assert scaffold_kfolds_for_n(3000) == 7
    assert scaffold_kfolds_for_n(3001) == 10


def test_no_scaffold_bucket_stays_in_one_fold():
    df = pd.DataFrame({
        "molecules": [
            "CCO", "CCN", "CCC", "CCCl",
            "c1ccccc1", "c1ccncc1", "O=C1CCCC1", "c1ccoc1",
        ],
        "y": list(range(8)),
    })
    folded = assign_scaffold_folds(df, n_folds=3, seed=10086)
    ns = folded.loc[folded["no_scaffold"]]
    assert ns["fold"].nunique() == 1
    assert no_scaffold_fold(folded) == int(ns["fold"].iloc[0])


def test_scaffold_folds_are_reproducible_with_seed():
    df = pd.DataFrame({
        "molecules": [
            "CCO", "CCN", "CCC", "CCCl",
            "c1ccccc1", "c1ccncc1", "O=C1CCCC1", "c1ccoc1",
        ],
        "y": list(range(8)),
    })
    a = assign_scaffold_folds(df, n_folds=3, seed=10086)
    b = assign_scaffold_folds(df, n_folds=3, seed=10086)
    assert a["fold"].tolist() == b["fold"].tolist()


def test_invalid_smiles_removed_before_splitting():
    df = pd.DataFrame({
        "molecules": ["CCO", "CCN", "c1ccccc1", "c1ccncc1", "O=C1CCCC1", "not_a_smiles"],
        "y": list(range(6)),
    })
    folded = assign_scaffold_folds(df, n_folds=3, seed=10086)
    assert "not_a_smiles" not in set(folded["molecules"])
