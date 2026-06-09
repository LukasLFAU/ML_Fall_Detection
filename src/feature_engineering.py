"""
feature_engineering.py

This script builds a simple feature dataset from the preprocessed Sensor Logger data.

Input:
    data/processed/combined/all_recordings_combined_trimmed.csv

Output:
    data/processed/features/features_dataset.csv

Why this step is needed:
The preprocessed dataset still contains time series data, meaning that each
recording has many sensor rows. A Random Forest model cannot directly use the
raw time series in this simple baseline setup. Therefore, we summarize each
recording into one row of statistical features.

Each output row represents one recording.

Important adjustment:
We intentionally do not calculate "energy" features here. Energy is based on the
sum of squared values and can therefore depend strongly on the number of rows in
a recording. Since our fall and non-fall recordings may have different trimmed
durations, energy could accidentally leak information about the preprocessing
instead of representing real movement patterns.

We also exclude gravity_mag from the feature signals because gravity magnitude
is almost constant around 9.81. Instead, we keep gravity_x, gravity_y and
gravity_z, because the axes can still describe phone orientation changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/combined/all_recordings_combined_trimmed.csv")
OUTPUT_DIR = Path("data/processed/features")
OUTPUT_FILE = OUTPUT_DIR / "features_dataset.csv"

"""
These are the sensor columns we use for the baseline model.
Accelerometer captures movement intensity and impact.
Gyroscope captures rotation.
Gravity axes capture orientation changes.

We exclude gravity_mag because it is almost constant and not very informative.
"""
SIGNAL_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "acc_mag",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "gyro_mag",
    "gravity_x",
    "gravity_y",
    "gravity_z",
]


def calculate_rms(values: pd.Series) -> float:
    """
    Calculate root mean square.

    RMS is useful for sensor data because it describes the overall signal strength
    without simply summing over all rows like an energy feature would do.
    """
    values = values.dropna()

    if values.empty:
        return np.nan

    return float(np.sqrt(np.mean(values ** 2)))


def extract_features_for_recording(recording_df: pd.DataFrame) -> dict:
    """
    Extract statistical features for one recording.

    One recording can be a fall or a non-fall activity. We keep metadata such as
    label, subtype and person, and then calculate statistics for each sensor signal.
    """
    first_row = recording_df.iloc[0]

    features = {
        "recording_id": first_row["recording_id"],
        "label": first_row["label"],
        "subtype": first_row["subtype"],
        "person": first_row["person"],
        "trim_method": first_row["trim_method"],
    }

    for signal in SIGNAL_COLUMNS:
        if signal not in recording_df.columns:
            continue

        values = pd.to_numeric(recording_df[signal], errors="coerce").dropna()

        if values.empty:
            continue

        features[f"{signal}_mean"] = values.mean()
        features[f"{signal}_std"] = values.std()
        features[f"{signal}_min"] = values.min()
        features[f"{signal}_max"] = values.max()
        features[f"{signal}_range"] = values.max() - values.min()
        features[f"{signal}_median"] = values.median()
        features[f"{signal}_rms"] = calculate_rms(values)
        features[f"{signal}_peak"] = values.max()

    return features


def build_feature_dataset(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """
    Build the complete feature dataset from the combined preprocessed CSV.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}. "
            "Please run preprocessing first: python scripts/run_preprocessing.py"
        )

    df = pd.read_csv(input_file)

    required_columns = ["recording_id", "label", "subtype", "person", "trim_method"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"The input file is missing required columns: {missing_columns}"
        )

    feature_rows = []

    for recording_id, recording_df in df.groupby("recording_id"):
        features = extract_features_for_recording(recording_df)
        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    return features_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_df = build_feature_dataset(INPUT_FILE)

    features_df.to_csv(OUTPUT_FILE, index=False)

    print("Feature engineering finished.")
    print(f"Saved features to: {OUTPUT_FILE}")
    print(f"Rows: {len(features_df)}")
    print(f"Columns: {len(features_df.columns)}")

    print()
    print("Label distribution:")
    print(features_df["label"].value_counts())

    print()
    print("Subtype distribution:")
    print(features_df.groupby(["label", "subtype"]).size())

    print()
    print("Preview:")
    print(features_df.head())


if __name__ == "__main__":
    main()