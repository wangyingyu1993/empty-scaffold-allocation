import pandas as pd

from empty_scaffold.chem_utils import get_scaffold, is_invalid_scaffold, is_no_scaffold
from empty_scaffold.data_utils import annotate_scaffolds


def test_no_scaffold_and_scaffold_detection():
    assert is_no_scaffold(get_scaffold("CCO"))
    assert not is_no_scaffold(get_scaffold("c1ccccc1"))


def test_invalid_smiles_is_not_counted_as_no_scaffold():
    scaffold = get_scaffold("not_a_smiles")
    assert is_invalid_scaffold(scaffold)
    assert not is_no_scaffold(scaffold)


def test_annotate_scaffolds_filters_invalid_and_keeps_no_scaffold():
    df = pd.DataFrame({"molecules": ["CCO", "c1ccccc1", "not_a_smiles"], "y": [1, 2, 3]})
    annotated = annotate_scaffolds(df, smiles_col="molecules")
    assert len(annotated) == 2
    assert annotated["no_scaffold"].sum() == 1
    assert "not_a_smiles" not in set(annotated["molecules"])
