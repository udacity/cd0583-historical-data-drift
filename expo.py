from pathlib import Path
import io
import warnings
import zipfile

import mlflow
import pandas as pd
import plotly.graph_objects as go
import requests

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00275/Bike-Sharing-Dataset.zip"
)

EXPERIMENT_NAME = "Dataset Drift Analysis with Evidently"

NUMERICAL_FEATURES = ["weathersit", "temp", "atemp", "hum", "windspeed"]

REFERENCE_START = "2011-01-01 00:00:00"
REFERENCE_END = "2011-01-28 23:00:00"

EXPERIMENT_BATCHES = [
    ("2011-02-01 00:00:00", "2011-02-28 23:00:00"),
    ("2011-03-01 00:00:00", "2011-03-31 23:00:00"),
    ("2011-04-01 00:00:00", "2011-04-30 23:00:00"),
    ("2011-05-01 00:00:00", "2011-05-31 23:00:00"),
    ("2011-06-01 00:00:00", "2011-06-30 23:00:00"),
    ("2011-07-01 00:00:00", "2011-07-31 23:00:00"),
]


def setup_mlflow() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_data() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        df = pd.read_csv(
            archive.open("day.csv"),
            header=0,
            sep=",",
            parse_dates=["dteday"],
            index_col="dteday",
        )

    return df


def make_dataset(df: pd.DataFrame) -> Dataset:
    definition = DataDefinition(numerical_columns=NUMERICAL_FEATURES)
    return Dataset.from_pandas(
        df[NUMERICAL_FEATURES].copy(),
        data_definition=definition,
    )


def first_match(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = first_match(value, keys)
            if found is not None:
                return found

    if isinstance(obj, list):
        for item in obj:
            found = first_match(item, keys)
            if found is not None:
                return found

    return None


def run_report(metric, current_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict:
    current_dataset = make_dataset(current_df)
    reference_dataset = make_dataset(reference_df)

    report = Report([metric])
    result = report.run(current_dataset, reference_dataset)
    return result.dict()


def detect_dataset_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    return_ratio: bool = False,
):
    report_dict = run_report(
        DriftedColumnsCount(),
        current_df=production_df,
        reference_df=reference_df,
    )

    drift_count = first_match(report_dict, ["count", "drifted_columns_count"])
    drift_share = first_match(report_dict, ["share", "drift_share"])

    if drift_share is None and drift_count is not None:
        drift_share = drift_count / len(NUMERICAL_FEATURES)

    if drift_share is None:
        raise ValueError("Could not extract dataset drift results from Evidently output.")

    if return_ratio:
        return float(drift_share)

    return bool(drift_share >= 0.5)


def detect_features_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    return_scores: bool = False,
):
    results = []

    for feature in NUMERICAL_FEATURES:
        report_dict = run_report(
            ValueDrift(column=feature),
            current_df=production_df,
            reference_df=reference_df,
        )

        drift_detected = first_match(report_dict, ["drift_detected", "detected"])
        drift_score = first_match(report_dict, ["drift_score", "score", "value"])

        if drift_detected is None and drift_score is None:
            raise ValueError(
                f"Could not extract drift results for feature '{feature}'."
            )

        if return_scores:
            results.append((feature, float(drift_score)))
        else:
            results.append((feature, bool(drift_detected)))

    return results


def log_batch_to_mlflow(
    batch_label: str,
    dataset_drift: bool,
    dataset_drift_ratio: float,
    feature_flags: list[tuple[str, bool]],
    feature_scores: list[tuple[str, float]],
) -> None:
    with mlflow.start_run(run_name=batch_label):
        mlflow.log_param("batch", batch_label)
        mlflow.log_param("experiment_type", "historical_data_drift")

        mlflow.log_metric("dataset_drift", int(dataset_drift))
        mlflow.log_metric("dataset_drift_ratio", float(dataset_drift_ratio))

        for feature, flag in feature_flags:
            mlflow.log_metric(f"{feature}_drift_flag", int(flag))

        for feature, score in feature_scores:
            mlflow.log_metric(f"{feature}_drift_score", float(score))


def save_figure(fig: go.Figure, stem: str) -> None:
    html_path = OUTPUT_DIR / f"{stem}.html"
    fig.write_html(str(html_path))

    png_path = OUTPUT_DIR / f"{stem}.png"
    try:
        fig.write_image(str(png_path))
        print(f"Saved: {html_path.name}, {png_path.name}")
    except Exception as exc:
        print(f"Saved: {html_path.name}")
        print(f"PNG export skipped for {stem}: {exc}")


def make_heatmap(
    z,
    x_labels,
    y_labels,
    title_x: str,
    title_y: str,
    colorscale: str,
    zmin=None,
    zmax=None,
    showscale=True,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            hoverongaps=False,
            xgap=1,
            ygap=1,
            zmin=zmin,
            zmax=zmax,
            showscale=showscale,
            colorscale=colorscale,
        )
    )

    fig.update_xaxes(side="top")
    fig.update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        template="plotly_white",
    )
    return fig


def main() -> None:
    setup_mlflow()

    raw_data = load_data()
    reference_df = raw_data.loc[REFERENCE_START:REFERENCE_END].copy()
    x_labels = [end for _, end in EXPERIMENT_BATCHES]

    feature_drift_flags_matrix = []
    feature_drift_scores_matrix = []
    dataset_drift_flags = []
    dataset_drift_ratios = []

    for start, end in EXPERIMENT_BATCHES:
        production_df = raw_data.loc[start:end].copy()
        batch_label = f"{start} to {end}"

        feature_flags = detect_features_drift(reference_df, production_df)
        feature_scores = detect_features_drift(
            reference_df,
            production_df,
            return_scores=True,
        )
        dataset_drift = detect_dataset_drift(reference_df, production_df)
        dataset_drift_ratio = detect_dataset_drift(
            reference_df,
            production_df,
            return_ratio=True,
        )

        feature_drift_flags_matrix.append([int(flag) for _, flag in feature_flags])
        feature_drift_scores_matrix.append([score for _, score in feature_scores])
        dataset_drift_flags.append(int(dataset_drift))
        dataset_drift_ratios.append(dataset_drift_ratio)

        log_batch_to_mlflow(
            batch_label=batch_label,
            dataset_drift=dataset_drift,
            dataset_drift_ratio=dataset_drift_ratio,
            feature_flags=feature_flags,
            feature_scores=feature_scores,
        )

    feature_drift_flags_df = pd.DataFrame(
        feature_drift_flags_matrix,
        columns=NUMERICAL_FEATURES,
    )

    fig1 = make_heatmap(
        z=feature_drift_flags_df.transpose().values,
        x_labels=x_labels,
        y_labels=NUMERICAL_FEATURES,
        title_x="Timestamp",
        title_y="Feature Drift",
        colorscale="Bluered",
        zmin=0,
        zmax=1,
        showscale=False,
    )
    save_figure(fig1, "fig1")

    feature_drift_scores_df = pd.DataFrame(
        feature_drift_scores_matrix,
        columns=NUMERICAL_FEATURES,
    )

    fig2 = make_heatmap(
        z=feature_drift_scores_df.transpose().values,
        x_labels=x_labels,
        y_labels=NUMERICAL_FEATURES,
        title_x="Timestamp",
        title_y="Drift Score",
        colorscale="Reds",
        zmin=0,
        zmax=1,
        showscale=True,
    )
    save_figure(fig2, "fig2")

    fig3 = make_heatmap(
        z=[dataset_drift_flags],
        x_labels=x_labels,
        y_labels=[""],
        title_x="Timestamp",
        title_y="Dataset Drift",
        colorscale="Bluered",
        zmin=0,
        zmax=1,
        showscale=False,
    )
    save_figure(fig3, "fig3")

    fig4 = make_heatmap(
        z=[dataset_drift_ratios],
        x_labels=x_labels,
        y_labels=[""],
        title_x="Timestamp",
        title_y="Drift Ratio",
        colorscale="Reds",
        zmin=0,
        zmax=1,
        showscale=True,
    )
    save_figure(fig4, "fig4")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
