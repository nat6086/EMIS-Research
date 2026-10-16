# tests/test_splits.py  — s'executa amb `python tests/test_splits.py` (també compatible amb pytest)
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emis_research.make_splits import OUT_DIR, VERSION, validate  # noqa: E402

PATH = OUT_DIR / f"splits_{VERSION}.csv"


def load():
    return pd.read_csv(PATH, dtype={"reference_id": str})


def test_no_group_overlap_and_balance():
    validate(load())


def test_reference_ids_keep_leading_zeros():
    assert load()["reference_id"].str.len().eq(4).all()


def test_s2_sizes_match_notebook():
    sizes = load()["fold_S2"].value_counts().sort_index().tolist()
    assert sizes == [136, 156, 115, 125, 126, 121, 112, 122, 109, 126]


if __name__ == "__main__":
    for fn in (test_no_group_overlap_and_balance,
               test_reference_ids_keep_leading_zeros, test_s2_sizes_match_notebook):
        fn()
        print("OK ", fn.__name__)
    print("ALL CHECKS PASSED")