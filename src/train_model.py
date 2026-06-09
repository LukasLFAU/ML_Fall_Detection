"""
train_model.py

This script trains and evaluates a baseline model for fall detection.

Input:
    data/processed/features/features_dataset.csv

Outputs:
    models/fall_detection_model.pkl
    models/feature_columns.pkl

    data/processed/evaluation/evaluation_report.txt
    data/processed/evaluation/evaluation_metrics.csv
    data/processed/evaluation/confusion_matrix_*.csv
    data/processed/evaluation/confusion_matrix_*.png
    data/processed/evaluation/feature_importances_*.csv

Why we use multiple evaluations:
A normal random train/test split can look overly good, because recordings from
the same person and similar recording conditions may appear in both training and
test data.

Therefore, this script reports:
1. Random recording-level split
2. Person holdout: train on Lukas, test on Polina
3. Person holdout: train on Polina, test on Lukas

The final model for the Streamlit demo is then trained on the full dataset.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


FEATURE_DATASET = Path("data/processed/features/features_dataset.csv")

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "fall_detection_model.pkl"
FEATURE_COLUMNS_FILE = MODEL_DIR / "feature_columns.pkl"

EVALUATION_DIR = Path("data/processed/evaluation")
EVALUATION_REPORT_FILE = EVALUATION_DIR / "evaluation_report.txt"
EVALUATION_METRICS_FILE = EVALUATION_DIR / "evaluation_metrics.csv"


# These columns describe the recording, but they are not sensor features.
# They must not be used for model training, because they would leak metadata.
NON_FEATURE_COLUMNS = [
    "recording_id",
    "label",
    "subtype",
    "person",
    "trim_method",
]

CLASS_LABELS = ["fall", "non_fall"]


def log(message: str = "", report_lines: list[str] | None = None) -> None:
    """
    Print a message and optionally store it for the evaluation report.
    """
    print(message)

    if report_lines is not None:
        report_lines.append(str(message))


def load_feature_dataset(path: Path = FEATURE_DATASET) -> pd.DataFrame:
    """
    Load the feature dataset created by feature_engineering.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. "
            "Please run feature engineering first: python src/feature_engineering.py"
        )

    df = pd.read_csv(path)

    required_columns = ["recording_id", "label", "person"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Feature dataset is missing required columns: {missing_columns}"
        )

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Prepare X and y.

    X contains only numeric sensor-derived features.
    y contains the class label: fall / non_fall.
    """
    y = df["label"]

    columns_to_drop = [
        column for column in NON_FEATURE_COLUMNS
        if column in df.columns
    ]

    X = df.drop(columns=columns_to_drop)

    # Make sure all remaining columns are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Random Forest cannot handle missing values directly.
    # For this first baseline, we fill missing values with 0.
    X = X.fillna(0)

    return X, y


def create_random_forest() -> RandomForestClassifier:
    """
    Create a Random Forest baseline model.

    The parameters are intentionally a bit conservative to reduce overfitting.
    """
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=6,
        min_samples_leaf=3,
    )


def save_confusion_matrix_outputs(
    cm,
    evaluation_key: str,
    evaluation_title: str
) -> None:
    """
    Save confusion matrix as CSV and PNG.

    The label order is:
        fall, non_fall

    For the PNG visualization, the colors do not depend on the count values.

    Correct classifications are always shown in green:
        - actual fall, predicted fall
        - actual non_fall, predicted non_fall

    Wrong classifications are always shown in red:
        - actual fall, predicted non_fall
        - actual non_fall, predicted fall

    This makes the plot easier to interpret visually, because green always means
    correct and red always means incorrect.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in CLASS_LABELS],
        columns=[f"predicted_{label}" for label in CLASS_LABELS],
    )

    csv_path = EVALUATION_DIR / f"confusion_matrix_{evaluation_key}.csv"
    png_path = EVALUATION_DIR / f"confusion_matrix_{evaluation_key}.png"

    cm_df.to_csv(csv_path)

    # Fixed color logic:
    # 1 = correct classification -> green
    # 0 = wrong classification   -> red
    color_grid = [
        [1, 0],
        [0, 1],
    ]

    cmap = ListedColormap([
        "#f4a6a6",  # red for incorrect cells
        "#9bd79b",  # green for correct cells
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=1)

    ax.set_title(evaluation_title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)

    # Add values and interpretation into each cell.
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            is_correct = i == j
            cell_label = "correct" if is_correct else "wrong"

            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cell_label}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

    # Add thin grid lines between cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(CLASS_LABELS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(CLASS_LABELS))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def save_feature_importances(
    model: RandomForestClassifier,
    feature_columns: list[str],
    evaluation_key: str
) -> pd.Series:
    """
    Save feature importances as CSV and return the sorted Series.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    importances = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    output_path = EVALUATION_DIR / f"feature_importances_{evaluation_key}.csv"
    importances.to_csv(output_path, header=["importance"])

    return importances


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    evaluation_key: str,
    evaluation_title: str,
    report_lines: list[str]
) -> dict:
    """
    Evaluate one trained model and save relevant outputs.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_fall = precision_score(
        y_test,
        y_pred,
        pos_label="fall",
        zero_division=0
    )
    recall_fall = recall_score(
        y_test,
        y_pred,
        pos_label="fall",
        zero_division=0
    )
    f1_fall = f1_score(
        y_test,
        y_pred,
        pos_label="fall",
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    report = classification_report(y_test, y_pred, zero_division=0)

    save_confusion_matrix_outputs(
        cm=cm,
        evaluation_key=evaluation_key,
        evaluation_title=evaluation_title
    )

    importances = save_feature_importances(
        model=model,
        feature_columns=X_test.columns.tolist(),
        evaluation_key=evaluation_key
    )

    log("", report_lines)
    log("=" * 60, report_lines)
    log(evaluation_title, report_lines)
    log("=" * 60, report_lines)

    log("Accuracy:", report_lines)
    log(str(accuracy), report_lines)

    log("", report_lines)
    log("Precision (fall):", report_lines)
    log(str(precision_fall), report_lines)

    log("", report_lines)
    log("Recall (fall):", report_lines)
    log(str(recall_fall), report_lines)

    log("", report_lines)
    log("F1-score (fall):", report_lines)
    log(str(f1_fall), report_lines)

    log("", report_lines)
    log("Confusion Matrix [labels: fall, non_fall]:", report_lines)
    log(str(cm), report_lines)

    log("", report_lines)
    log("Classification Report:", report_lines)
    log(report, report_lines)

    log("", report_lines)
    log("Top 20 feature importances:", report_lines)
    log(str(importances.head(20)), report_lines)

    return {
        "evaluation_key": evaluation_key,
        "evaluation_title": evaluation_title,
        "accuracy": accuracy,
        "precision_fall": precision_fall,
        "recall_fall": recall_fall,
        "f1_fall": f1_fall,
        "test_rows": len(y_test),
    }


def evaluate_random_split(
    X: pd.DataFrame,
    y: pd.Series,
    report_lines: list[str]
) -> dict:
    """
    Evaluate the model using a normal random recording-level split.

    This is useful as a baseline, but it can be too optimistic because both
    persons and similar recordings can appear in training and test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = create_random_forest()
    model.fit(X_train, y_train)

    log("", report_lines)
    log("Random split details:", report_lines)
    log(f"Training rows: {len(X_train)}", report_lines)
    log(f"Test rows: {len(X_test)}", report_lines)

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        evaluation_key="random_split",
        evaluation_title="Evaluation 1: Random recording-level split",
        report_lines=report_lines
    )

    metrics["train_rows"] = len(X_train)
    metrics["train_person"] = "mixed"
    metrics["test_person"] = "mixed"

    return metrics


def evaluate_person_holdout(
    df: pd.DataFrame,
    train_person: str,
    test_person: str,
    evaluation_number: int,
    report_lines: list[str]
) -> dict:
    """
    Evaluate generalization to another person.

    Example:
        train_person = "lukas"
        test_person = "polina"

    This is stricter than a random split because the model is tested on a person
    it has not seen during training.
    """
    train_df = df[df["person"].str.lower() == train_person.lower()].copy()
    test_df = df[df["person"].str.lower() == test_person.lower()].copy()

    if train_df.empty:
        raise ValueError(f"No training data found for person: {train_person}")

    if test_df.empty:
        raise ValueError(f"No test data found for person: {test_person}")

    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)

    # Ensure the same feature order in train and test.
    X_test = X_test[X_train.columns]

    model = create_random_forest()
    model.fit(X_train, y_train)

    log("", report_lines)
    log("Person holdout details:", report_lines)
    log(f"Train person: {train_person}", report_lines)
    log(f"Test person: {test_person}", report_lines)
    log(f"Training rows: {len(X_train)}", report_lines)
    log(f"Test rows: {len(X_test)}", report_lines)

    log("", report_lines)
    log("Training label distribution:", report_lines)
    log(str(y_train.value_counts()), report_lines)

    log("", report_lines)
    log("Test label distribution:", report_lines)
    log(str(y_test.value_counts()), report_lines)

    evaluation_key = f"{train_person}_to_{test_person}"
    evaluation_title = (
        f"Evaluation {evaluation_number}: "
        f"Train on {train_person}, test on {test_person}"
    )

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        evaluation_key=evaluation_key,
        evaluation_title=evaluation_title,
        report_lines=report_lines
    )

    metrics["train_rows"] = len(X_train)
    metrics["train_person"] = train_person
    metrics["test_person"] = test_person

    return metrics


def train_final_model_on_all_data(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    report_lines: list[str]
) -> RandomForestClassifier:
    """
    Train the final model on all available data.

    This model is saved for the Streamlit demo. The evaluation above should be
    used to discuss model performance, not this final training step.
    """
    model = create_random_forest()
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_columns, FEATURE_COLUMNS_FILE)

    log("", report_lines)
    log("=" * 60, report_lines)
    log("Final model", report_lines)
    log("=" * 60, report_lines)
    log("Final model trained on all available recordings.", report_lines)
    log(f"Model saved to: {MODEL_FILE}", report_lines)
    log(f"Feature columns saved to: {FEATURE_COLUMNS_FILE}", report_lines)

    return model


def save_evaluation_report(report_lines: list[str]) -> None:
    """
    Save the complete evaluation text report.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    EVALUATION_REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print()
    print(f"Evaluation report saved to: {EVALUATION_REPORT_FILE}")


def save_evaluation_metrics(metrics_rows: list[dict]) -> None:
    """
    Save evaluation metrics as a structured CSV file.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(EVALUATION_METRICS_FILE, index=False)

    print(f"Evaluation metrics saved to: {EVALUATION_METRICS_FILE}")


def main() -> None:
    report_lines = []
    metrics_rows = []

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_dataset(FEATURE_DATASET)

    log("Loaded feature dataset.", report_lines)
    log(f"Rows: {len(df)}", report_lines)
    log(f"Columns: {len(df.columns)}", report_lines)

    log("", report_lines)
    log("Label distribution:", report_lines)
    log(str(df["label"].value_counts()), report_lines)

    log("", report_lines)
    log("Person distribution:", report_lines)
    log(str(df["person"].value_counts()), report_lines)

    log("", report_lines)
    log("Label distribution by person:", report_lines)
    log(str(df.groupby(["person", "label"]).size()), report_lines)

    X, y = prepare_features_and_target(df)
    feature_columns = X.columns.tolist()

    # 1. Baseline evaluation with random split
    metrics_rows.append(
        evaluate_random_split(
            X=X,
            y=y,
            report_lines=report_lines
        )
    )

    # 2. Stricter person-independent evaluations
    metrics_rows.append(
        evaluate_person_holdout(
            df=df,
            train_person="lukas",
            test_person="polina",
            evaluation_number=2,
            report_lines=report_lines
        )
    )

    metrics_rows.append(
        evaluate_person_holdout(
            df=df,
            train_person="polina",
            test_person="lukas",
            evaluation_number=3,
            report_lines=report_lines
        )
    )

    # 3. Train final model for the Streamlit demo
    train_final_model_on_all_data(
        X=X,
        y=y,
        feature_columns=feature_columns,
        report_lines=report_lines
    )

    save_evaluation_report(report_lines)
    save_evaluation_metrics(metrics_rows)


if __name__ == "__main__":
    main()