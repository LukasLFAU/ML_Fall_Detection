"""
prediction.py

Reusable prediction logic for the fall detection project.

This module supports two prediction modes:

1. Prediction from an already preprocessed combined CSV / DataFrame
2. Prediction from a raw Sensor Logger ZIP file

The raw ZIP workflow is label-independent:
    Raw Sensor Logger ZIP
    -> read Accelerometer, Gyroscope and Gravity data
    -> calculate acceleration magnitude
    -> select the strongest movement window automatically
    -> combine sensors
    -> extract features
    -> apply trained Random Forest model

Important:
    Labels, subtypes, persons and recording IDs are not used as model features.
"""

from pathlib import Path
from typing import Any
import zipfile

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import extract_features_for_recording


MODEL_FILE = Path("models/fall_detection_model.pkl")
FEATURE_COLUMNS_FILE = Path("models/feature_columns.pkl")

ALL_RECORDINGS_FILE = Path(
    "data/processed/combined/all_recordings_combined_trimmed.csv"
)

NON_FEATURE_COLUMNS = [
    "recording_id",
    "label",
    "subtype",
    "person",
    "trim_method",
]


def load_prediction_assets(
    model_path: Path = MODEL_FILE,
    feature_columns_path: Path = FEATURE_COLUMNS_FILE,
):
    """
    Load the trained model and the feature columns used during training.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run training first: python src/train_model.py"
        )

    if not feature_columns_path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {feature_columns_path}. "
            "Please run training first: python src/train_model.py"
        )

    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_columns_path)

    return model, feature_columns



# Raw Sensor Logger ZIP handling


def find_file_in_zip(zip_file: zipfile.ZipFile, expected_file_name: str) -> str:
    """
    Find a file inside a Sensor Logger ZIP by file name.

    Example:
        expected_file_name = "Accelerometer.csv"

    The file may be located inside a subfolder in the ZIP.
    """
    expected_file_name = expected_file_name.lower()

    for member_name in zip_file.namelist():
        if Path(member_name).name.lower() == expected_file_name:
            return member_name

    available_files = "\n".join(zip_file.namelist())

    raise FileNotFoundError(
        f"Could not find {expected_file_name} in ZIP file.\n"
        f"Available files:\n{available_files}"
    )


def read_sensor_csv_from_zip(
    zip_path: Path | str,
    expected_file_name: str,
) -> pd.DataFrame:
    """
    Read one sensor CSV from a raw Sensor Logger ZIP file.
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        member_name = find_file_in_zip(
            zip_file=zip_file,
            expected_file_name=expected_file_name,
        )

        with zip_file.open(member_name) as sensor_file:
            sensor_df = pd.read_csv(sensor_file)

    return sensor_df


def standardize_sensor_dataframe(
    sensor_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Keep and rename the required sensor columns.

    Expected Sensor Logger columns:
        seconds_elapsed, x, y, z

    Output example for prefix="acc":
        seconds_elapsed, acc_x, acc_y, acc_z
    """
    required_columns = ["seconds_elapsed", "x", "y", "z"]
    missing_columns = [
        column for column in required_columns
        if column not in sensor_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Sensor data is missing required columns: {missing_columns}"
        )

    df = sensor_df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns)
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    df = df.rename(
        columns={
            "x": f"{prefix}_x",
            "y": f"{prefix}_y",
            "z": f"{prefix}_z",
        }
    )

    return df


def calculate_magnitude(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Calculate vector magnitude from x, y and z columns.

    Example:
        acc_mag = sqrt(acc_x^2 + acc_y^2 + acc_z^2)
    """
    df = df.copy()

    df[output_col] = np.sqrt(
        df[x_col] ** 2
        + df[y_col] ** 2
        + df[z_col] ** 2
    )

    return df


def select_peak_window(
    acc_df: pd.DataFrame,
    setup_ignore_seconds: float = 5.0,
    pre_peak_seconds: float = 3.0,
    post_peak_seconds: float = 5.0,
) -> dict:
    """
    Select an analysis window around the strongest acceleration peak.

    This is label-independent:
        It does not use the file name, label, subtype or person.

    Logic:
        1. Ignore the first few seconds because the phone may still be handled.
        2. Find the strongest acc_mag peak.
        3. Select a window around this peak.
    """
    if "seconds_elapsed" not in acc_df.columns:
        raise ValueError("Column 'seconds_elapsed' is missing.")

    if "acc_mag" not in acc_df.columns:
        raise ValueError("Column 'acc_mag' is missing.")

    first_time = float(acc_df["seconds_elapsed"].min())
    last_time = float(acc_df["seconds_elapsed"].max())

    search_start = first_time + setup_ignore_seconds

    search_df = acc_df[acc_df["seconds_elapsed"] >= search_start].copy()

    # Fallback for very short recordings
    if search_df.empty:
        search_df = acc_df.copy()

    peak_index = search_df["acc_mag"].idxmax()
    peak_time = float(acc_df.loc[peak_index, "seconds_elapsed"])
    peak_acc_mag = float(acc_df.loc[peak_index, "acc_mag"])

    window_start = max(first_time, peak_time - pre_peak_seconds)
    window_end = min(last_time, peak_time + post_peak_seconds)

    if window_end <= window_start:
        raise ValueError(
            "Could not create a valid analysis window around the peak."
        )

    return {
        "window_start_s": window_start,
        "window_end_s": window_end,
        "peak_time_s": peak_time,
        "peak_acc_mag": peak_acc_mag,
        "peak_acc_g": peak_acc_mag / 9.80665,
        "setup_ignore_seconds": setup_ignore_seconds,
        "pre_peak_seconds": pre_peak_seconds,
        "post_peak_seconds": post_peak_seconds,
    }


def trim_sensor_to_window(
    sensor_df: pd.DataFrame,
    window_start: float,
    window_end: float,
) -> pd.DataFrame:
    """
    Trim one sensor DataFrame to the selected analysis window.
    """
    trimmed_df = sensor_df[
        (sensor_df["seconds_elapsed"] >= window_start)
        & (sensor_df["seconds_elapsed"] <= window_end)
    ].copy()

    if trimmed_df.empty:
        raise ValueError(
            "Selected analysis window produced an empty sensor DataFrame."
        )

    return trimmed_df.sort_values("seconds_elapsed").reset_index(drop=True)


def combine_sensor_data(
    acc_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    gravity_df: pd.DataFrame,
    window_start: float,
) -> pd.DataFrame:
    """
    Combine accelerometer, gyroscope and gravity data into one DataFrame.

    The accelerometer timeline is used as the base timeline.
    Gyroscope and gravity values are matched by nearest timestamp.
    """
    acc_df = acc_df.sort_values("seconds_elapsed").reset_index(drop=True)
    gyro_df = gyro_df.sort_values("seconds_elapsed").reset_index(drop=True)
    gravity_df = gravity_df.sort_values("seconds_elapsed").reset_index(drop=True)

    combined_df = pd.merge_asof(
        acc_df,
        gyro_df,
        on="seconds_elapsed",
        direction="nearest",
    )

    combined_df = pd.merge_asof(
        combined_df,
        gravity_df,
        on="seconds_elapsed",
        direction="nearest",
    )

    combined_df = combined_df.sort_values("seconds_elapsed").reset_index(drop=True)

    # Fill possible small gaps after timestamp matching.
    combined_df = combined_df.ffill().bfill()

    combined_df["seconds_trimmed"] = (
        combined_df["seconds_elapsed"] - window_start
    )

    return combined_df


def build_combined_dataframe_from_raw_zip(
    zip_path: Path | str,
    recording_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build a combined processed DataFrame from a raw Sensor Logger ZIP file.

    Returns:
        combined_df:
            Sensor data in the same general structure as the processed pipeline.
        raw_acc_df:
            Accelerometer data with acc_mag for plotting.
        window_info:
            Information about the automatically selected analysis window.
    """
    zip_path = Path(zip_path)

    if recording_id is None:
        recording_id = zip_path.stem

    raw_acc_df = read_sensor_csv_from_zip(zip_path, "Accelerometer.csv")
    raw_gyro_df = read_sensor_csv_from_zip(zip_path, "Gyroscope.csv")
    raw_gravity_df = read_sensor_csv_from_zip(zip_path, "Gravity.csv")

    acc_df = standardize_sensor_dataframe(raw_acc_df, prefix="acc")
    gyro_df = standardize_sensor_dataframe(raw_gyro_df, prefix="gyro")
    gravity_df = standardize_sensor_dataframe(raw_gravity_df, prefix="gravity")

    acc_df = calculate_magnitude(
        df=acc_df,
        x_col="acc_x",
        y_col="acc_y",
        z_col="acc_z",
        output_col="acc_mag",
    )

    gyro_df = calculate_magnitude(
        df=gyro_df,
        x_col="gyro_x",
        y_col="gyro_y",
        z_col="gyro_z",
        output_col="gyro_mag",
    )

    window_info = select_peak_window(acc_df)

    window_start = window_info["window_start_s"]
    window_end = window_info["window_end_s"]

    acc_window_df = trim_sensor_to_window(acc_df, window_start, window_end)
    gyro_window_df = trim_sensor_to_window(gyro_df, window_start, window_end)
    gravity_window_df = trim_sensor_to_window(gravity_df, window_start, window_end)

    combined_df = combine_sensor_data(
        acc_df=acc_window_df,
        gyro_df=gyro_window_df,
        gravity_df=gravity_window_df,
        window_start=window_start,
    )

    # Add metadata columns required by the feature extraction function.
    # These values are removed before model prediction.
    combined_df["recording_id"] = recording_id
    combined_df["label"] = "unknown"
    combined_df["subtype"] = "unknown"
    combined_df["person"] = "unknown"
    combined_df["trim_method"] = "automatic_peak_window"

    raw_plot_acc_df = acc_df.copy()

    return combined_df, raw_plot_acc_df, window_info


# Feature alignment and prediction

def ensure_required_metadata_columns(
    recording_df: pd.DataFrame,
    default_recording_id: str = "uploaded_recording",
) -> pd.DataFrame:
    """
    Ensure that metadata columns required by feature_engineering.py exist.

    These metadata columns are removed before model prediction.
    """
    df = recording_df.copy()

    if "recording_id" not in df.columns:
        df["recording_id"] = default_recording_id

    if "label" not in df.columns:
        df["label"] = "unknown"

    if "subtype" not in df.columns:
        df["subtype"] = "unknown"

    if "person" not in df.columns:
        df["person"] = "unknown"

    if "trim_method" not in df.columns:
        df["trim_method"] = "unknown"

    return df


def validate_single_recording(recording_df: pd.DataFrame) -> None:
    """
    Check whether the input DataFrame contains exactly one recording.
    """
    if "recording_id" not in recording_df.columns:
        return

    unique_recordings = recording_df["recording_id"].dropna().unique()

    if len(unique_recordings) > 1:
        raise ValueError(
            "The provided DataFrame contains multiple recording_id values. "
            "Please filter to one recording before predicting."
        )


def build_model_input_from_recording(
    recording_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Extract features from one recording and align them with training columns.
    """
    validate_single_recording(recording_df)

    df = ensure_required_metadata_columns(recording_df)

    feature_dict = extract_features_for_recording(df)
    feature_df = pd.DataFrame([feature_dict])

    columns_to_drop = [
        column for column in NON_FEATURE_COLUMNS
        if column in feature_df.columns
    ]

    X = feature_df.drop(columns=columns_to_drop)

    X = X.apply(pd.to_numeric, errors="coerce")

    # Add missing columns if necessary.
    for column in feature_columns:
        if column not in X.columns:
            X[column] = 0

    # Remove extra columns and ensure the same order as during training.
    X = X[feature_columns]

    X = X.fillna(0)

    return X


def get_human_readable_interpretation(prediction: str) -> str:
    """
    Convert technical model labels into user-friendly text.
    """
    if prediction == "fall":
        return "Potential fall-like movement detected"

    if prediction == "non_fall":
        return "No fall-like movement detected"

    return "Unknown movement pattern"


def predict_from_recording_dataframe(
    recording_df: pd.DataFrame,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """
    Predict fall / non_fall for one already combined recording DataFrame.
    """
    if model is None or feature_columns is None:
        model, feature_columns = load_prediction_assets()

    X = build_model_input_from_recording(
        recording_df=recording_df,
        feature_columns=feature_columns,
    )

    prediction = model.predict(X)[0]

    result = {
        "prediction": prediction,
        "interpretation": get_human_readable_interpretation(prediction),
        "confidence": None,
        "fall_probability": None,
        "non_fall_probability": None,
        "probabilities": {},
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        class_labels = list(model.classes_)

        probability_dict = {
            str(label): float(probability)
            for label, probability in zip(class_labels, probabilities)
        }

        result["probabilities"] = probability_dict
        result["confidence"] = float(max(probabilities))
        result["fall_probability"] = probability_dict.get("fall")
        result["non_fall_probability"] = probability_dict.get("non_fall")

    return result


def predict_from_raw_sensor_zip(
    zip_path: Path | str,
    recording_id: str | None = None,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """
    Predict from a raw Sensor Logger ZIP file.

    Returns:
        result:
            Prediction result dictionary.
        combined_df:
            Automatically processed analysis window.
        raw_acc_df:
            Full accelerometer signal with acc_mag for plotting.
        window_info:
            Information about selected peak window.
    """
    if model is None or feature_columns is None:
        model, feature_columns = load_prediction_assets()

    combined_df, raw_acc_df, window_info = build_combined_dataframe_from_raw_zip(
        zip_path=zip_path,
        recording_id=recording_id,
    )

    result = predict_from_recording_dataframe(
        recording_df=combined_df,
        model=model,
        feature_columns=feature_columns,
    )

    result["input_type"] = "raw_sensor_logger_zip"
    result["selected_window_start_s"] = window_info["window_start_s"]
    result["selected_window_end_s"] = window_info["window_end_s"]
    result["peak_time_s"] = window_info["peak_time_s"]
    result["peak_acc_mag"] = window_info["peak_acc_mag"]
    result["peak_acc_g"] = window_info["peak_acc_g"]

    return result, combined_df, raw_acc_df, window_info


# Processed CSV / processed dataset helpers

def predict_from_combined_csv(
    csv_path: Path | str,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """
    Load one preprocessed combined CSV file and predict fall / non_fall.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    recording_df = pd.read_csv(csv_path)

    return predict_from_recording_dataframe(
        recording_df=recording_df,
        model=model,
        feature_columns=feature_columns,
    )


def load_all_recordings_combined(
    path: Path = ALL_RECORDINGS_FILE,
) -> pd.DataFrame:
    """
    Load the combined processed dataset containing all recordings.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Combined recordings file not found: {path}. "
            "Please run preprocessing first."
        )

    return pd.read_csv(path)


def get_available_recording_ids(
    all_recordings_df: pd.DataFrame,
) -> list[str]:
    """
    Return all available recording IDs from the combined dataset.
    """
    if "recording_id" not in all_recordings_df.columns:
        raise ValueError("Column 'recording_id' is missing.")

    return sorted(all_recordings_df["recording_id"].dropna().unique().tolist())


def get_recording_by_id(
    all_recordings_df: pd.DataFrame,
    recording_id: str,
) -> pd.DataFrame:
    """
    Extract one recording from the combined processed dataset.
    """
    if "recording_id" not in all_recordings_df.columns:
        raise ValueError("Column 'recording_id' is missing.")

    recording_df = all_recordings_df[
        all_recordings_df["recording_id"] == recording_id
    ].copy()

    if recording_df.empty:
        raise ValueError(f"No recording found for recording_id: {recording_id}")

    return recording_df


def predict_by_recording_id(
    recording_id: str,
    all_recordings_df: pd.DataFrame | None = None,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """
    Predict fall / non_fall for one recording selected from all recordings.
    """
    if all_recordings_df is None:
        all_recordings_df = load_all_recordings_combined()

    recording_df = get_recording_by_id(
        all_recordings_df=all_recordings_df,
        recording_id=recording_id,
    )

    return predict_from_recording_dataframe(
        recording_df=recording_df,
        model=model,
        feature_columns=feature_columns,
    )


def get_recording_metadata(recording_df: pd.DataFrame) -> dict:
    """
    Return basic metadata for optional debugging.

    These metadata fields are not used as model features.
    """
    metadata = {}

    for column in ["recording_id", "label", "subtype", "person", "trim_method"]:
        if column in recording_df.columns:
            metadata[column] = str(recording_df[column].iloc[0])

    return metadata