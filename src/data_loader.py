"""

Utility functions for loading raw Sensor Logger recordings.

The raw data is expected in a nested folder structure such as:

data/raw/
    fall_backward/lukas/Fall_Backward_Lukas1.zip
    fall_forward/polina/Fall_Forward_Polina1.zip
    non_fall/nfa_walking/lukas/NFA_Walking_Lukas1.zip

Each Sensor Logger ZIP file should contain:
- Accelerometer.csv
- Gyroscope.csv
- Gravity.csv
"""

from pathlib import Path
import zipfile
import re
from typing import Dict, List

import pandas as pd


REQUIRED_SENSOR_FILES = {
    "accelerometer": "Accelerometer.csv",
    "gyroscope": "Gyroscope.csv",
    "gravity": "Gravity.csv",
}


def list_raw_recordings(raw_data_dir: str | Path = "data/raw") -> List[Path]:
    """
    Recursively finds all ZIP files in the raw data folder.

    Example:
        data/raw/fall_backward/lukas/Fall_Backward_Lukas1.zip

    Returns
    -------
    list[Path]
        Sorted list of all ZIP recording paths.
    """
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    zip_files = sorted(raw_data_dir.rglob("*.zip"))

    return zip_files


def find_file_in_zip(zip_file: zipfile.ZipFile, target_filename: str) -> str:
    """
    Finds a specific CSV file inside a Sensor Logger ZIP file.

    This is written flexibly because some ZIP files may contain files directly:
        Accelerometer.csv

    while others may contain them inside a folder:
        SomeFolder/Accelerometer.csv
    """
    for file_name in zip_file.namelist():
        if Path(file_name).name.lower() == target_filename.lower():
            return file_name

    raise FileNotFoundError(
        f"Could not find {target_filename} inside ZIP file. "
        f"Available files: {zip_file.namelist()}"
    )


def validate_sensor_dataframe(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """
    Checks whether the required Sensor Logger columns exist and converts them to numeric values.

    Required columns:
    - seconds_elapsed
    - x
    - y
    - z
    """
    required_columns = ["seconds_elapsed", "x", "y", "z"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{sensor_name} data is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns)
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)

    return df


def load_sensor_recording(zip_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Loads Accelerometer, Gyroscope and Gravity data from one Sensor Logger ZIP file.

    Parameters
    ----------
    zip_path:
        Path to one raw Sensor Logger ZIP file.

    Returns
    -------
    dict
        Dictionary with three pandas DataFrames:
        {
            "accelerometer": df,
            "gyroscope": df,
            "gravity": df
        }
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file does not exist: {zip_path}")

    sensor_data = {}

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for sensor_key, sensor_filename in REQUIRED_SENSOR_FILES.items():
            internal_file = find_file_in_zip(zip_file, sensor_filename)

            with zip_file.open(internal_file) as file:
                df = pd.read_csv(file)

            df = validate_sensor_dataframe(df, sensor_key)
            sensor_data[sensor_key] = df

    return sensor_data


def infer_metadata_from_path(zip_path: str | Path, raw_data_dir: str | Path = "data/raw") -> dict:
    """
    Infers metadata from the folder path and file name.
    We do this because our folder structure already contains useful labels.
    
    Example path:
        data/raw/fall_backward/lukas/Fall_Backward_Lukas1.zip

    Inferred metadata:
        label = fall
        subtype = backward
        person = lukas

    Example path:
        data/raw/non_fall/nfa_walking/polina/NFA_Walking_Polina1.zip

    Inferred metadata:
        label = non_fall
        subtype = walking
        person = polina

    This metadata is later important for preprocessing, feature engineering,
    train/test splits and evaluation.
    """
    zip_path = Path(zip_path)
    raw_data_dir = Path(raw_data_dir)

    recording_id = zip_path.stem
    source_zip = zip_path.name

    try:
        relative_parts = zip_path.relative_to(raw_data_dir).parts
    except ValueError:
        relative_parts = zip_path.parts

    label = "unknown"
    subtype = "unknown"
    person = "unknown"

    if len(relative_parts) >= 3:
        first_level = relative_parts[0].lower()

        if first_level.startswith("fall_"):
            label = "fall"
            subtype = first_level.replace("fall_", "")
            person = relative_parts[1].lower()

        elif first_level == "non_fall":
            label = "non_fall"

            if len(relative_parts) >= 4:
                subtype = relative_parts[1].lower().replace("nfa_", "")
                person = relative_parts[2].lower()

    recording_number = extract_recording_number(recording_id)

    return {
        "recording_id": recording_id,
        "source_zip": source_zip,
        "path": str(zip_path),
        "label": label,
        "subtype": subtype,
        "person": person,
        "recording_number": recording_number,
    }


def extract_recording_number(recording_id: str) -> int | None:
    """
    Extracts the final number from a recording ID.

    Examples:
        Fall_Backward_Lukas1 -> 1
        NFA_Walking_Polina12 -> 12

    This is not essential for the model, but it helps us keep recordings ordered
    and check whether files are missing.
    """
    match = re.search(r"(\d+)$", recording_id)

    if match:
        return int(match.group(1))

    return None


def build_recording_index(raw_data_dir: str | Path = "data/raw") -> pd.DataFrame:
    
    """
    Creates a metadata overview for all raw ZIP recordings.

    Important:
    This function does not load all sensor values. It only scans the folder
    structure and creates one row per recording. This makes it fast and useful
    for checking whether our raw dataset is complete and correctly organized.
    """

    zip_files = list_raw_recordings(raw_data_dir)

    rows = [
        infer_metadata_from_path(zip_path, raw_data_dir=raw_data_dir)
        for zip_path in zip_files
    ]

    return pd.DataFrame(rows)