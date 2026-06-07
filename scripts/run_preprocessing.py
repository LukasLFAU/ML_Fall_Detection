"""
run_preprocessing.py

This script runs the preprocessing pipeline for all Sensor Logger recordings.

Run from the repository root:

    python scripts/run_preprocessing.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import preprocess_all_recordings


def main():
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    output_root = PROJECT_ROOT / "data" / "processed"
    metadata_output_dir = PROJECT_ROOT / "data" / "metadata"

    metadata_df = preprocess_all_recordings(
        raw_data_dir=raw_data_dir,
        output_root=output_root,
        metadata_output_dir=metadata_output_dir
    )

    print()
    print("Metadata preview:")
    print(metadata_df.head())


if __name__ == "__main__":
    main()