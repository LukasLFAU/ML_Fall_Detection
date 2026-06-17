"""
streamlit_app.py

Simple Streamlit demo for the fall detection project.

The app supports:
1. Raw Sensor Logger ZIP upload
2. Upload of a preprocessed combined CSV file
3. Selection from the already preprocessed project dataset

How to run:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.prediction import (  # noqa: E402
    load_prediction_assets,
    load_all_recordings_combined,
    get_available_recording_ids,
    get_recording_by_id,
    predict_from_recording_dataframe,
    predict_from_raw_sensor_zip,
)


st.set_page_config(
    page_title="Fall Detection Demo",
    page_icon="📱",
    layout="wide",
)


def inject_custom_css() -> None:
    """
    Add custom styles that work reasonably well in light and dark mode.
    """
    st.markdown(
        """
        <style>
        .tag {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            line-height: 1.3;
        }

        .tag-fall {
            color: #ff4b4b;
            background: rgba(255, 75, 75, 0.12);
            border: 1px solid rgba(255, 75, 75, 0.25);
        }

        .tag-nonfall {
            color: #22c55e;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.25);
        }

        .prob-card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.95rem;
            padding: 1.1rem 1.2rem;
            min-height: 155px;
        }

        .prob-title {
            font-size: 0.9rem;
            font-weight: 600;
            opacity: 0.78;
            margin-bottom: 0.35rem;
        }

        .prob-value {
            font-size: 2rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }

        .prob-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: rgba(128, 128, 128, 0.15);
            margin-bottom: 0.7rem;
        }

        .progress-bg {
            width: 100%;
            height: 0.6rem;
            border-radius: 999px;
            background: rgba(128, 128, 128, 0.22);
            overflow: hidden;
            margin-top: 0.65rem;
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
        }

        .window-panel {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 1rem;
            padding: 1rem;
            margin-top: 0.25rem;
        }

        .window-intro {
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }

        .window-card {
            border-radius: 0.95rem;
            padding: 0.95rem 1rem;
            min-height: 105px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            margin-bottom: 0.75rem;
        }

        .window-card-start {
            background: rgba(59, 130, 246, 0.13);
            border-left: 6px solid #3b82f6;
        }

        .window-card-peak {
            background: rgba(249, 115, 22, 0.15);
            border-left: 6px solid #f97316;
        }

        .window-card-end {
            background: rgba(34, 197, 94, 0.14);
            border-left: 6px solid #22c55e;
        }

        .window-card-wide {
            background: rgba(168, 85, 247, 0.14);
            border-left: 6px solid #a855f7;
            border-radius: 0.95rem;
            padding: 0.95rem 1rem;
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
        }

        .window-label {
            font-size: 0.8rem;
            font-weight: 650;
            opacity: 0.78;
            margin-bottom: 0.3rem;
        }

        .window-value {
            font-size: 1.8rem;
            font-weight: 760;
        }

        .window-caption {
            font-size: 0.88rem;
            opacity: 0.78;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file_to_temp(uploaded_file, suffix: str) -> Path:
    """
    Save an uploaded Streamlit file temporarily and return the local path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    return temp_path


def delete_temp_file(temp_path: Path) -> None:
    """
    Delete a temporary file if it exists.
    """
    try:
        if temp_path.exists():
            os.remove(temp_path)
    except OSError:
        pass


def render_probability_card(
    title: str,
    probability: float | None,
    selected: bool,
    accent_color: str,
) -> None:
    """
    Render a probability card with a custom progress bar.
    """
    if probability is None:
        value_text = "n/a"
        width_percent = 0
    else:
        value_text = f"{probability:.2%}"
        width_percent = max(0, min(100, round(probability * 100)))

    badge_text = "Selected result" if selected else "Alternative class"
    border_style = f"border-left: 6px solid {accent_color};" if selected else ""

    st.markdown(
        f"""
        <div class="prob-card" style="{border_style}">
            <div class="prob-title">{title}</div>
            <div class="prob-value">{value_text}</div>
            <div class="prob-badge">{badge_text}</div>
            <div class="progress-bg">
                <div class="progress-fill"
                     style="width: {width_percent}%; background: {accent_color};">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_window_card(label: str, value: str, card_class: str) -> None:
    """
    Render one colored window information card.
    """
    st.markdown(
        f"""
        <div class="window-card {card_class}">
            <div class="window-label">{label}</div>
            <div class="window-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_prediction_result(result: dict) -> None:
    """
    Display the model result in a modern and non-technical way.
    """
    prediction = result.get("prediction", "unknown")
    interpretation = result.get("interpretation", "Unknown movement pattern")

    fall_probability = result.get("fall_probability")
    non_fall_probability = result.get("non_fall_probability")

    st.subheader("Model Result")

    with st.container(border=True):
        if prediction == "fall":
            st.error(f"⚠️ {interpretation}")
        elif prediction == "non_fall":
            st.success(f"✅ {interpretation}")
        else:
            st.info(f"ℹ️ {interpretation}")

        col1, col2 = st.columns(2)

        with col1:
            render_probability_card(
                title="Fall-like movement probability",
                probability=fall_probability,
                selected=prediction == "fall",
                accent_color="#ff4b4b",
            )

        with col2:
            render_probability_card(
                title="Non-fall-like movement probability",
                probability=non_fall_probability,
                selected=prediction == "non_fall",
                accent_color="#22c55e",
            )

        st.caption(
            "The class with the higher probability is selected as the model "
            "result. Labels, subtypes, persons and file names are not used as "
            "model input."
        )


def get_time_column_for_plot(recording_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    Determine the best available time column for plotting.
    """
    plot_df = recording_df.copy()

    if "seconds_elapsed" in plot_df.columns:
        return plot_df, "seconds_elapsed", "Seconds elapsed"

    if "seconds_trimmed" in plot_df.columns:
        return plot_df, "seconds_trimmed", "Seconds within movement window"

    plot_df = plot_df.reset_index().rename(columns={"index": "sample"})
    return plot_df, "sample", "Sample"


def build_window_info_from_preprocessed_recording(recording_df: pd.DataFrame) -> dict:
    """
    Build window information for an already preprocessed recording.

    For preprocessed data, the displayed time range corresponds to the available
    preprocessed movement window.
    """
    if "acc_mag" not in recording_df.columns:
        raise ValueError("Column 'acc_mag' is required for window visualization.")

    plot_df, time_column, _x_label = get_time_column_for_plot(recording_df)

    signal_df = plot_df[[time_column, "acc_mag"]].dropna().copy()

    if signal_df.empty:
        raise ValueError("No valid acceleration magnitude values available.")

    peak_index = signal_df["acc_mag"].idxmax()
    peak_acc_mag = float(signal_df.loc[peak_index, "acc_mag"])

    return {
        "window_start_s": float(signal_df[time_column].min()),
        "window_end_s": float(signal_df[time_column].max()),
        "peak_time_s": float(signal_df.loc[peak_index, time_column]),
        "peak_acc_mag": peak_acc_mag,
        "peak_acc_g": peak_acc_mag / 9.80665,
    }


def show_acceleration_overview(
    signal_df: pd.DataFrame,
    window_info: dict,
    mode: str,
) -> None:
    """
    Plot acceleration magnitude and highlight the selected or preprocessed window.
    """
    st.subheader("Detected Analysis Window")

    if signal_df.empty:
        st.info("No acceleration data available.")
        return

    if "acc_mag" not in signal_df.columns:
        st.info("Column 'acc_mag' is not available.")
        return

    plot_df, time_column, x_label = get_time_column_for_plot(signal_df)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        plot_df[time_column],
        plot_df["acc_mag"],
        label="Acceleration magnitude",
    )

    ax.axvspan(
        window_info["window_start_s"],
        window_info["window_end_s"],
        alpha=0.2,
        label="selected movement window",
    )

    ax.axvline(
        window_info["peak_time_s"],
        linestyle="--",
        label="strongest acceleration peak",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Acceleration magnitude (m/s²)")
    ax.set_title("Acceleration magnitude with selected movement window")
    ax.legend(loc="upper left")

    st.pyplot(fig)
    plt.close(fig)

    if mode == "raw":
        st.caption(
            "The plot shows the acceleration magnitude from the raw Sensor "
            "Logger recording. The highlighted area is the automatically "
            "selected analysis window. The final model prediction also uses "
            "gyroscope and gravity features extracted from the same time segment."
        )
    else:
        st.caption(
            "The plot shows the acceleration magnitude of the already "
            "preprocessed movement window. The final model prediction also uses "
            "gyroscope and gravity features extracted from the same time segment."
        )


def show_window_information(window_info: dict, mode: str) -> None:
    """
    Display details about the selected movement window.
    """
    st.subheader("Automatic Window Selection")

    if mode == "raw":
        intro_text = (
            "The app selects the movement section around the strongest "
            "acceleration peak and extracts model features from this window."
        )
    else:
        intro_text = (
            "This recording is already preprocessed. The app summarizes the "
            "available movement window and highlights its strongest acceleration peak."
        )

    st.markdown(
        f"""
        <div class="window-panel">
            <div class="window-intro">{intro_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_window_card(
            label="Window start",
            value=f"{window_info['window_start_s']:.2f} s",
            card_class="window-card-start",
        )

    with col2:
        render_window_card(
            label="Peak time",
            value=f"{window_info['peak_time_s']:.2f} s",
            card_class="window-card-peak",
        )

    with col3:
        render_window_card(
            label="Window end",
            value=f"{window_info['window_end_s']:.2f} s",
            card_class="window-card-end",
        )

    st.markdown(
        f"""
        <div class="window-card-wide">
            <div class="window-label">Peak acceleration</div>
            <div class="window-value">
                {window_info['peak_acc_mag']:.2f} m/s² = {window_info['peak_acc_g']:.2f} g
            </div>
            <div class="window-caption">
                The peak acceleration helps identify the most relevant movement
                section. The final prediction is based on statistical features
                from accelerometer, gyroscope and gravity signals within this window.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_and_display_preprocessed_recording(
    recording_df: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> None:
    """
    Run prediction for preprocessed recording data and display results.
    """
    result = predict_from_recording_dataframe(
        recording_df=recording_df,
        model=model,
        feature_columns=feature_columns,
    )

    window_info = build_window_info_from_preprocessed_recording(recording_df)

    show_prediction_result(result)

    st.divider()

    show_acceleration_overview(
        signal_df=recording_df,
        window_info=window_info,
        mode="preprocessed",
    )

    show_window_information(
        window_info=window_info,
        mode="preprocessed",
    )


def main() -> None:
    inject_custom_css()

    st.title("📱 Fall Detection Demo")

    st.markdown(
        """
        <div style="font-size: 1.05rem; margin-bottom: 0.65rem;">
            This prototype classifies smartphone sensor recordings as
            <span class="tag tag-fall">fall-like movement</span>
            or
            <span class="tag tag-nonfall">non-fall-like movement</span>
            using a trained Random Forest model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Prototype note: This demo is not a real-world safety system. "
        "It demonstrates model inference on collected Sensor Logger data. "
        "The model prediction is based on extracted sensor features, not on "
        "labels, subtypes, persons or file names."
    )

    try:
        model, feature_columns = load_prediction_assets()
    except Exception as error:
        st.error("Could not load model files.")
        st.exception(error)
        st.stop()

    tab_raw_zip, tab_upload_preprocessed, tab_existing = st.tabs(
        [
            "Upload raw Sensor Logger ZIP",
            "Upload preprocessed CSV",
            "Select preprocessed recording",
        ]
    )

    with tab_raw_zip:
        st.header("Upload Raw Sensor Logger ZIP")

        st.write(
            "Upload an original Sensor Logger ZIP file containing "
            "`Accelerometer.csv`, `Gyroscope.csv` and `Gravity.csv`. "
            "The app automatically extracts the sensor data, selects a movement "
            "window and applies the trained model."
        )

        uploaded_zip = st.file_uploader(
            "Upload raw Sensor Logger ZIP",
            type=["zip"],
            key="raw_zip_uploader",
        )

        if uploaded_zip is not None:
            if st.button("Analyze raw ZIP recording"):
                temp_zip_path = save_uploaded_file_to_temp(
                    uploaded_file=uploaded_zip,
                    suffix=".zip",
                )

                try:
                    (
                        result,
                        _combined_df,
                        raw_acc_df,
                        window_info,
                    ) = predict_from_raw_sensor_zip(
                        zip_path=temp_zip_path,
                        recording_id=uploaded_zip.name,
                        model=model,
                        feature_columns=feature_columns,
                    )

                    show_prediction_result(result)

                    st.divider()

                    show_acceleration_overview(
                        signal_df=raw_acc_df,
                        window_info=window_info,
                        mode="raw",
                    )

                    show_window_information(
                        window_info=window_info,
                        mode="raw",
                    )

                except Exception as error:
                    st.error("Raw ZIP prediction failed.")
                    st.exception(error)

                finally:
                    delete_temp_file(temp_zip_path)

    with tab_upload_preprocessed:
        st.header("Upload Preprocessed Recording CSV")

        st.write(
            "Upload a preprocessed combined CSV file. The app uses the same "
            "prediction workflow as for the project recordings: feature "
            "extraction, feature alignment and model inference."
        )

        uploaded_csv = st.file_uploader(
            "Upload preprocessed combined CSV",
            type=["csv"],
            key="preprocessed_csv_uploader",
        )

        if uploaded_csv is not None:
            uploaded_df = pd.read_csv(uploaded_csv)

            if "recording_id" in uploaded_df.columns:
                unique_recording_ids = (
                    uploaded_df["recording_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                if len(unique_recording_ids) > 1:
                    selected_uploaded_id = st.selectbox(
                        "The uploaded file contains multiple recordings. "
                        "Choose one recording:",
                        sorted(unique_recording_ids),
                    )

                    uploaded_df = uploaded_df[
                        uploaded_df["recording_id"] == selected_uploaded_id
                    ].copy()

            if st.button("Analyze uploaded preprocessed CSV"):
                try:
                    predict_and_display_preprocessed_recording(
                        recording_df=uploaded_df,
                        model=model,
                        feature_columns=feature_columns,
                    )
                except Exception as error:
                    st.error("Preprocessed CSV prediction failed.")
                    st.exception(error)

    with tab_existing:
        st.header("Select Preprocessed Project Recording")

        st.write(
            "Select one of the recordings collected for this project after "
            "preprocessing. This mode is mainly useful for testing the app with "
            "known project data and for demonstrating how the trained model "
            "behaves on the collected recordings."
        )

        try:
            all_recordings_df = load_all_recordings_combined()
            recording_ids = get_available_recording_ids(all_recordings_df)
        except Exception as error:
            st.error("Could not load the combined preprocessed dataset.")
            st.exception(error)
            recording_ids = []

        if recording_ids:
            selected_recording_id = st.selectbox(
                "Choose a preprocessed project recording",
                recording_ids,
            )

            selected_recording_df = get_recording_by_id(
                all_recordings_df=all_recordings_df,
                recording_id=selected_recording_id,
            )

            if st.button("Analyze selected preprocessed recording"):
                predict_and_display_preprocessed_recording(
                    recording_df=selected_recording_df,
                    model=model,
                    feature_columns=feature_columns,
                )


if __name__ == "__main__":
    main()