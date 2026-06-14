def test_core_imports():
    from empty_scaffold.chem_utils import get_scaffold, is_no_scaffold
    from empty_scaffold.data_utils import annotate_scaffolds, summarize_scaffolds
    from empty_scaffold.scaffold_splitting import assign_scaffold_folds, scaffold_kfolds_for_n
    from empty_scaffold.leakage import load_split_csv, summarize_split_leakage
    from empty_scaffold.esa_repair import repair_archived_datasail_split, write_repaired_split_tables

    assert callable(get_scaffold)
    assert callable(is_no_scaffold)
    assert callable(annotate_scaffolds)
    assert callable(summarize_scaffolds)
    assert callable(assign_scaffold_folds)
    assert scaffold_kfolds_for_n(1000) == 5
    assert callable(load_split_csv)
    assert callable(summarize_split_leakage)
    assert callable(repair_archived_datasail_split)
    assert callable(write_repaired_split_tables)
