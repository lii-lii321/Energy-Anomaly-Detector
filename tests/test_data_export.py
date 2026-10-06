from pathlib import Path

import pandas as pd
from conftest import REPO_ROOT

from train import FEATURE_COLUMNS, build_labeled_samples

DATA_PATH = Path(REPO_ROOT) / "data" / "sensor_data.csv"
LABEL_COLUMN = "label"
DOCS_TEST_PATH = Path(REPO_ROOT) / "tests" / "test_docs_consistency.py"
INJECTED_ANOMALY = [1.7, 18.0]


def _export_to(tmp_path):
    features, labels = build_labeled_samples()
    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df[LABEL_COLUMN] = labels
    output = tmp_path / "sensor_data.csv"
    df.to_csv(output, index=False, lineterminator="\n")
    return output


def test_committed_csv_matches_in_memory_source(tmp_path):
    assert DATA_PATH.is_file(), f"missing committed data artifact: {DATA_PATH}"
    regenerated = _export_to(tmp_path)
    committed_text = DATA_PATH.read_text(encoding="utf-8")
    regenerated_text = regenerated.read_text(encoding="utf-8")
    assert committed_text == regenerated_text, (
        "data/sensor_data.csv has drifted from build_labeled_samples; "
        "regenerate it via scripts/generate_data.py"
    )


def test_committed_csv_structure():
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 201
    assert list(df.columns) == [*FEATURE_COLUMNS, LABEL_COLUMN]
    assert set(df[LABEL_COLUMN].unique()) == {0, 1}
    anomalies = df[df[LABEL_COLUMN] == 1]
    assert len(anomalies) == 1
    assert anomalies[FEATURE_COLUMNS].values.tolist() == [INJECTED_ANOMALY]


def test_docs_gate_reversed_to_existence_check():
    source = DOCS_TEST_PATH.read_text(encoding="utf-8")
    assert "test_nonexistent_data_dir_is_not_mentioned" not in source
    assert "def test_data_dir_references_exist" in source
