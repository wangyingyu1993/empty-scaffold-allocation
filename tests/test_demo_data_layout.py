from pathlib import Path

import pytest


def test_current_demo_data_layout():
    root = Path(__file__).resolve().parents[1]
    demo_root = root / "data" / "demo"
    if not demo_root.exists():
        pytest.skip("demo data directory is not bundled in this test-only checkout")

    required = [
        "scaffold_from_scratch/bp.csv",
        "scaffold_from_scratch/pkb.csv",
        "scaffold_from_scratch/eye_corrosion.csv",
        "scaffold_from_scratch/micronucleus_tox.csv",
        "datasail_esa_cases/raw_datasail_splits/ames/train.csv",
        "datasail_esa_cases/raw_datasail_splits/ames/valid.csv",
        "datasail_esa_cases/raw_datasail_splits/ames/test.csv",
        "datasail_esa_cases/raw_datasail_splits/pkb/train.csv",
        "datasail_esa_cases/raw_datasail_splits/pkb/valid.csv",
        "datasail_esa_cases/raw_datasail_splits/pkb/test.csv",
    ]
    missing = [rel for rel in required if not (demo_root / rel).exists()]
    assert not missing, "Missing current demo files: " + ", ".join(missing)
