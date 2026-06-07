"""
preprocessing.py

This file contains the preprocessing logic for our Sensor Logger recordings.

Main idea:
The raw Sensor Logger ZIP files contain separate CSV files for Accelerometer,
Gyroscope and Gravity. Before we can train a model, we need to cut away irrelevant
setup movements and keep the meaningful part of each recording.

For fall recordings:
- calculate acceleration magnitude
- search for the strongest acceleration peak after the initial setup phase
- keep 3 seconds before and 5 seconds after this peak

For non-fall recordings:
- do not use peak-based trimming
- remove the setup phase at the beginning
- remove the possible stop/handling phase at the end
- keep the middle part of the recording

Important:
Gyroscope and Gravity are not used to determine the trimming point because they
have different units and describe different aspects of movement. However, they
are trimmed with the same time window and kept in the combined dataset so they
can later be used for feature engineering and model training.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import (
    list_raw_recordings,
    load_sensor_recording,
    infer_metadata_from_path,
)


# For fall recordings, we ignore the first seconds when searching for the
# acceleration peak. This is because the phone is started by hand and then placed
# into the pocket, which can create strong setup movements.
SEARCH_START_S = 5.0

# For fall recordings, we keep a window around the detected impact peak.
PRE_PEAK_S = 3.0
POST_PEAK_S = 5.0

# For non-fall activities, we do not search for a peak. Instead, we remove the
# setup phase at the beginning and the possible stop/phone handling phase at the end.
NFA_SETUP_CUT_S = 5.0
NFA_END_CUT_S = 3.0


def calculate_magnitude(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Calculate the vector magnitude from x, y and z.

    Example for accelerometer:
        acc_mag = sqrt(x² + y² + z²)

    We use this because one combined magnitude value is easier to analyze than
    looking at three axes separately.
    """
    df = df.copy()

    df[f"{prefix}_mag"] = np.sqrt(
        df["x"] ** 2 +
        df["y"] ** 2 +
        df["z"] ** 2
    )

    return df


def find_strongest_acc_peak(
    acc_df: pd.DataFrame,
    search_start_s: float = SEARCH_START_S
) -> tuple[float, float]:
    """
    Find the strongest acceleration peak after the setup phase.

    We ignore the first few seconds because the phone is placed into the pocket
    after starting the recording. This movement can create strong acceleration
    peaks but is not part of the actual fall or activity.

    We intentionally do not exclude the last seconds here. If a few fall recordings
    are still trimmed incorrectly, we handle them later via manual quality control
    instead of changing the global fall logic for all recordings.
    """
    acc_with_mag = calculate_magnitude(acc_df, "acc")

    search_area = acc_with_mag[
        acc_with_mag["seconds_elapsed"] >= search_start_s
    ]

    if search_area.empty:
        raise ValueError(
            "No accelerometer data available after the setup phase. "
            "The recording may be too short."
        )

    peak_idx = search_area["acc_mag"].idxmax()

    peak_time_s = float(acc_with_mag.loc[peak_idx, "seconds_elapsed"])
    peak_acc_mag = float(acc_with_mag.loc[peak_idx, "acc_mag"])

    return peak_time_s, peak_acc_mag


def get_peak_centered_window(
    acc_df: pd.DataFrame,
    search_start_s: float = SEARCH_START_S,
    pre_peak_s: float = PRE_PEAK_S,
    post_peak_s: float = POST_PEAK_S
) -> tuple[float, float, float, float]:
    """
    Create a trimming window around the strongest acceleration peak.

    This logic is used for fall recordings because the fall impact is usually
    visible as a strong peak in the accelerometer magnitude.

    Returns:
        trim_start_s, trim_end_s, peak_time_s, peak_acc_mag
    """
    peak_time_s, peak_acc_mag = find_strongest_acc_peak(
        acc_df,
        search_start_s=search_start_s
    )

    original_start_s = float(acc_df["seconds_elapsed"].min())
    original_end_s = float(acc_df["seconds_elapsed"].max())

    trim_start_s = max(original_start_s, peak_time_s - pre_peak_s)
    trim_end_s = min(original_end_s, peak_time_s + post_peak_s)

    return trim_start_s, trim_end_s, peak_time_s, peak_acc_mag


def get_middle_window_for_non_fall_activity(
    acc_df: pd.DataFrame,
    setup_cut_s: float = NFA_SETUP_CUT_S,
    end_cut_s: float = NFA_END_CUT_S
) -> tuple[float, float]:
    """
    Create a trimming window for non-fall activities.

    For non-fall activities, the strongest accelerometer peak is not always the
    actual relevant movement. It can also be caused by stopping the recording,
    taking the phone out, or other handling movements.

    Therefore, we remove the setup phase at the beginning and the ending phase
    at the end and keep the middle part of the recording.
    """
    original_start_s = float(acc_df["seconds_elapsed"].min())
    original_end_s = float(acc_df["seconds_elapsed"].max())

    trim_start_s = original_start_s + setup_cut_s
    trim_end_s = original_end_s - end_cut_s

    if trim_end_s <= trim_start_s:
        # Fallback for very short recordings:
        # keep everything after the initial setup phase.
        trim_start_s = min(original_start_s + SEARCH_START_S, original_end_s)
        trim_end_s = original_end_s

    return trim_start_s, trim_end_s


def choose_trimming_window(
    acc_df: pd.DataFrame,
    metadata: dict
) -> dict:
    """
    Decide how a recording should be trimmed.

    Fall recordings:
        We use peak-centered trimming based on accelerometer magnitude, because
        the fall impact is usually visible as a strong acceleration peak.

    Non-fall recordings:
        We do not use peak-centered trimming, because the strongest peak is often
        not the actual activity but a handling movement. Instead, we remove the
        setup phase and ending phase and keep the middle part.

    Gyroscope and Gravity:
        These sensors are not used to define the trimming point because they have
        different units and describe different aspects of movement. However, they
        are trimmed with the same time window and kept for later feature
        engineering and model training.
    """
    label = metadata["label"]

    if label == "fall":
        trim_start_s, trim_end_s, peak_time_s, peak_acc_mag = get_peak_centered_window(acc_df)

        return {
            "trim_method": "peak_centered_fall",
            "trim_start_s": trim_start_s,
            "trim_end_s": trim_end_s,
            "peak_time_s": peak_time_s,
            "peak_acc_mag": peak_acc_mag,
        }

    if label == "non_fall":
        trim_start_s, trim_end_s = get_middle_window_for_non_fall_activity(acc_df)

        return {
            "trim_method": "middle_cut_non_fall",
            "trim_start_s": trim_start_s,
            "trim_end_s": trim_end_s,
            "peak_time_s": None,
            "peak_acc_mag": None,
        }

    # Fallback for unexpected labels.
    # This should normally not happen if the folder structure is correct.
    trim_start_s, trim_end_s, peak_time_s, peak_acc_mag = get_peak_centered_window(acc_df)

    return {
        "trim_method": "peak_centered_fallback",
        "trim_start_s": trim_start_s,
        "trim_end_s": trim_end_s,
        "peak_time_s": peak_time_s,
        "peak_acc_mag": peak_acc_mag,
    }


def trim_sensor_dataframe(
    df: pd.DataFrame,
    trim_start_s: float,
    trim_end_s: float,
    prefix: str
) -> pd.DataFrame:
    """
    Apply the same time window to one sensor dataframe.

    We use the same trimming window for Accelerometer, Gyroscope and Gravity so
    that all sensors describe the same part of the recording.
    """
    trimmed = df[
        (df["seconds_elapsed"] >= trim_start_s) &
        (df["seconds_elapsed"] <= trim_end_s)
    ].copy()

    trimmed["seconds_trimmed"] = trimmed["seconds_elapsed"] - trim_start_s
    trimmed = calculate_magnitude(trimmed, prefix)

    return trimmed.reset_index(drop=True)


def combine_trimmed_sensors(
    acc_trimmed: pd.DataFrame,
    gyro_trimmed: pd.DataFrame,
    gravity_trimmed: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine Accelerometer, Gyroscope and Gravity into one table.

    The sensor timestamps are usually very close, but not always exactly equal.
    Therefore, we use merge_asof to match each accelerometer row with the nearest
    gyroscope and gravity row.
    """
    acc = acc_trimmed[
        ["seconds_elapsed", "seconds_trimmed", "x", "y", "z", "acc_mag"]
    ].rename(
        columns={
            "x": "acc_x",
            "y": "acc_y",
            "z": "acc_z",
        }
    ).sort_values("seconds_elapsed")

    gyro = gyro_trimmed[
        ["seconds_elapsed", "x", "y", "z", "gyro_mag"]
    ].rename(
        columns={
            "x": "gyro_x",
            "y": "gyro_y",
            "z": "gyro_z",
        }
    ).sort_values("seconds_elapsed")

    gravity = gravity_trimmed[
        ["seconds_elapsed", "x", "y", "z", "gravity_mag"]
    ].rename(
        columns={
            "x": "gravity_x",
            "y": "gravity_y",
            "z": "gravity_z",
        }
    ).sort_values("seconds_elapsed")

    combined = pd.merge_asof(
        acc,
        gyro,
        on="seconds_elapsed",
        direction="nearest",
        tolerance=0.03,
    )

    combined = pd.merge_asof(
        combined,
        gravity,
        on="seconds_elapsed",
        direction="nearest",
        tolerance=0.03,
    )

    return combined


def save_quality_control_plot(
    acc_df: pd.DataFrame,
    recording_id: str,
    trim_start_s: float,
    trim_end_s: float,
    output_dir: str | Path,
    peak_time_s: Optional[float] = None
) -> None:
    """
    Save a plot for manual quality control.

    The QC plot focuses on the accelerometer magnitude because this is the signal
    used for fall peak detection and trimming. Gyroscope and Gravity are still
    included in the processed data, but not used to define the trimming point.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_with_mag = calculate_magnitude(acc_df, "acc")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        acc_with_mag["seconds_elapsed"],
        acc_with_mag["acc_mag"],
        label="acc_mag"
    )

    ax.axvline(trim_start_s, linestyle="--", label="trim_start")
    ax.axvline(trim_end_s, linestyle="--", label="trim_end")

    if peak_time_s is not None:
        ax.axvline(peak_time_s, linestyle="-.", label="detected_peak")

    ax.set_title(f"{recording_id} - Acceleration Magnitude")
    ax.set_xlabel("Seconds elapsed")
    ax.set_ylabel("Acceleration magnitude")
    ax.legend()

    fig.tight_layout()

    output_path = output_dir / f"{recording_id}_qc_acc_mag.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def preprocess_one_recording(
    zip_path: str | Path,
    raw_data_dir: str | Path,
    output_root: str | Path,
    metadata_output_dir: str | Path
) -> tuple[pd.DataFrame, dict]:
    """
    Preprocess one Sensor Logger ZIP recording.

    Output for each recording:
    - trimmed sensor CSVs
    - one combined CSV
    - one quality-control plot
    - one metadata row
    """
    zip_path = Path(zip_path)
    output_root = Path(output_root)
    metadata_output_dir = Path(metadata_output_dir)

    metadata = infer_metadata_from_path(zip_path, raw_data_dir=raw_data_dir)
    recording_id = metadata["recording_id"]

    sensor_data = load_sensor_recording(zip_path)

    acc_df = sensor_data["accelerometer"]
    gyro_df = sensor_data["gyroscope"]
    gravity_df = sensor_data["gravity"]

    trimming_info = choose_trimming_window(acc_df, metadata)

    trim_start_s = trimming_info["trim_start_s"]
    trim_end_s = trimming_info["trim_end_s"]
    peak_time_s = trimming_info["peak_time_s"]

    acc_trimmed = trim_sensor_dataframe(
        acc_df,
        trim_start_s,
        trim_end_s,
        prefix="acc"
    )

    gyro_trimmed = trim_sensor_dataframe(
        gyro_df,
        trim_start_s,
        trim_end_s,
        prefix="gyro"
    )

    gravity_trimmed = trim_sensor_dataframe(
        gravity_df,
        trim_start_s,
        trim_end_s,
        prefix="gravity"
    )

    sensor_output_dir = output_root / "trimmed_by_sensor" / recording_id
    sensor_output_dir.mkdir(parents=True, exist_ok=True)

    acc_trimmed.to_csv(sensor_output_dir / "Accelerometer_trimmed.csv", index=False)
    gyro_trimmed.to_csv(sensor_output_dir / "Gyroscope_trimmed.csv", index=False)
    gravity_trimmed.to_csv(sensor_output_dir / "Gravity_trimmed.csv", index=False)

    combined = combine_trimmed_sensors(
        acc_trimmed,
        gyro_trimmed,
        gravity_trimmed
    )

    for key, value in metadata.items():
        combined[key] = value

    # These columns describe how the recording was trimmed and help us later
    # understand which preprocessing logic was applied. Peak-related values are
    # stored in metadata_preprocessing.csv only, because they are preprocessing
    # information and should not be repeated in every row of the sensor dataset.
    combined["trim_method"] = trimming_info["trim_method"]
    combined["trim_start_s"] = trim_start_s
    combined["trim_end_s"] = trim_end_s

    combined_output_dir = output_root / "combined"
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        combined_output_dir / f"{recording_id}_combined_trimmed.csv",
        index=False
    )

    save_quality_control_plot(
        acc_df=acc_df,
        recording_id=recording_id,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
        peak_time_s=peak_time_s,
        output_dir=output_root / "qc_plots"
    )

    original_start_s = float(acc_df["seconds_elapsed"].min())
    original_end_s = float(acc_df["seconds_elapsed"].max())

    metadata_row = {
        **metadata,
        "usable": "yes",
        "trim_method": trimming_info["trim_method"],
        "search_start_s": SEARCH_START_S,
        "trim_start_s": round(trim_start_s, 4),
        "trim_end_s": round(trim_end_s, 4),
        "trimmed_duration_s": round(trim_end_s - trim_start_s, 4),
        "peak_time_s": None if trimming_info["peak_time_s"] is None else round(trimming_info["peak_time_s"], 4),
        "peak_acc_mag": None if trimming_info["peak_acc_mag"] is None else round(trimming_info["peak_acc_mag"], 6),
        "original_start_s": round(original_start_s, 4),
        "original_end_s": round(original_end_s, 4),
        "original_duration_s": round(original_end_s - original_start_s, 4),
        "original_acc_rows": len(acc_df),
        "trimmed_acc_rows": len(acc_trimmed),
        "notes": "Automatically preprocessed from Sensor Logger export.",
    }

    return combined, metadata_row


def preprocess_all_recordings(
    raw_data_dir: str | Path = "data/raw",
    output_root: str | Path = "data/processed",
    metadata_output_dir: str | Path = "data/metadata"
) -> pd.DataFrame:
    """
    Preprocess all ZIP recordings found in data/raw.

    The function continues even if one recording fails. Failed recordings are
    written into the metadata file with usable = no.
    """
    raw_data_dir = Path(raw_data_dir)
    output_root = Path(output_root)
    metadata_output_dir = Path(metadata_output_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = list_raw_recordings(raw_data_dir)

    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP recordings found in {raw_data_dir}. "
            "Please add raw Sensor Logger exports first."
        )

    all_combined = []
    metadata_rows = []

    for zip_path in zip_files:
        print(f"Processing {zip_path.name}...")

        try:
            combined, metadata_row = preprocess_one_recording(
                zip_path=zip_path,
                raw_data_dir=raw_data_dir,
                output_root=output_root,
                metadata_output_dir=metadata_output_dir
            )

            all_combined.append(combined)
            metadata_rows.append(metadata_row)

        except Exception as error:
            metadata = infer_metadata_from_path(zip_path, raw_data_dir=raw_data_dir)

            metadata_rows.append({
                **metadata,
                "usable": "no",
                "trim_method": None,
                "search_start_s": SEARCH_START_S,
                "trim_start_s": None,
                "trim_end_s": None,
                "trimmed_duration_s": None,
                "peak_time_s": None,
                "peak_acc_mag": None,
                "original_start_s": None,
                "original_end_s": None,
                "original_duration_s": None,
                "original_acc_rows": None,
                "trimmed_acc_rows": None,
                "notes": f"Preprocessing failed: {error}",
            })

            print(f"Could not process {zip_path.name}: {error}")

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_df.to_csv(
        metadata_output_dir / "metadata_preprocessing.csv",
        index=False
    )

    if all_combined:
        # Some generated DataFrames may contain columns that are empty for a
        # specific recording type. Removing completely empty columns before
        # concatenation avoids pandas warnings and keeps the combined output cleaner.
        all_combined_clean = [
            df.dropna(axis=1, how="all")
            for df in all_combined
            if not df.empty
        ]

        all_combined_df = pd.concat(all_combined_clean, ignore_index=True)

        combined_output_dir = output_root / "combined"
        combined_output_dir.mkdir(parents=True, exist_ok=True)

        all_combined_df.to_csv(
            combined_output_dir / "all_recordings_combined_trimmed.csv",
            index=False
        )

    print()
    print("Preprocessing finished.")
    print(f"Recordings found: {len(zip_files)}")
    print(f"Successfully processed: {len(all_combined)}")
    print(f"Failed: {len(zip_files) - len(all_combined)}")
    print(f"Metadata saved to: {metadata_output_dir / 'metadata_preprocessing.csv'}")

    return metadata_df