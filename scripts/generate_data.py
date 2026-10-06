import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train import FEATURE_COLUMNS, build_labeled_samples  # noqa: E402

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "sensor_data.csv")
LABEL_COLUMN = "label"


def build_export_frame():
    features, labels = build_labeled_samples()
    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df[LABEL_COLUMN] = labels
    return df


def export_labeled_samples(output_path=DEFAULT_OUTPUT_PATH):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_export_frame().to_csv(output, index=False, lineterminator="\n")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="从 build_labeled_samples 确定性导出合成数据工件 data/sensor_data.csv"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    output = export_labeled_samples(args.output)
    print(f"✅ 合成数据已保存至: {output}")


if __name__ == "__main__":
    main()
