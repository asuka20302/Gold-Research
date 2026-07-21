"""Shared helpers for PCA, loss, and weighted-ensemble experiments."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import gold_model_comparison as gm
from compare_bagged_huber import load_frames


PROJECT = Path(__file__).resolve().parent
SELECTED_MODELS_PATH = (
    PROJECT / "outputs" / "horizon_comparison" / "h1" / "selected_models.csv"
)

MODEL_CLASSES = {
    "OLS": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "Huber": HuberRegressor,
}


def selected_model_table() -> pd.DataFrame:
    """Load the frozen horizon-one model definitions."""
    return pd.read_csv(SELECTED_MODELS_PATH)


def parse_parameter_text(model_name: str, text: str) -> dict[str, object]:
    """Recover estimator parameters and ignore the optional bagging suffix."""
    raw = ast.literal_eval(text.split(";")[0])
    estimator_class = MODEL_CLASSES[model_name]
    allowed = estimator_class().get_params().keys()
    return {key: value for key, value in raw.items() if key in allowed}


def selected_templates() -> tuple[dict[str, Pipeline], dict[str, list[str]]]:
    """Construct unfitted copies of the five selected base estimators."""
    templates: dict[str, Pipeline] = {}
    features: dict[str, list[str]] = {}
    for _, row in selected_model_table().iterrows():
        name = str(row["model"])
        parameters = parse_parameter_text(name, str(row["selected_parameters"]))
        estimator = MODEL_CLASSES[name](**parameters)
        templates[name] = make_pipeline(estimator)
        features[name] = str(row["active_features"]).split(",")
    return templates, features


def make_pipeline(estimator: object, n_components: int | None = None) -> Pipeline:
    """Impute, scale, optionally apply PCA, then fit an estimator."""
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, svd_solver="full")))
    steps.append(("model", estimator))
    return Pipeline(steps)


def amount_metrics(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    entry_open: pd.Series | np.ndarray | None = None,
    exit_close: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Score continuous growth/drop and price-change errors.

    Returns are stored as log returns by the model.  The ``growth_*`` fields
    convert them to ordinary percentage growth, so +2.0 means a 2% increase
    and -2.0 means a 2% decrease.  No directional-accuracy classification is
    used.
    """
    actual_array = np.asarray(actual, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    zero_mse = mean_squared_error(actual_array, np.zeros(len(actual_array)))
    mse = mean_squared_error(actual_array, predicted)
    actual_std = float(np.std(actual_array))
    actual_growth = np.expm1(actual_array) * 100.0
    predicted_growth = np.expm1(predicted) * 100.0
    growth_error = predicted_growth - actual_growth
    absolute_growth_error = np.abs(growth_error)
    magnitude_error = np.abs(predicted_growth) - np.abs(actual_growth)
    metrics = {
        "mse": float(mse),
        "mae": float(mean_absolute_error(actual_array, predicted)),
        "relative_mse_zero": float(mse / zero_mse),
        "r2": float(r2_score(actual_array, predicted)),
        "correlation": gm.safe_correlation(actual_array, predicted),
        "magnitude_mae": float(
            mean_absolute_error(np.abs(actual_array), np.abs(predicted))
        ),
        "magnitude_correlation": gm.safe_correlation(
            np.abs(actual_array), np.abs(predicted)
        ),
        "amplitude_ratio": (
            float(np.std(predicted)) / actual_std if actual_std > 0 else np.nan
        ),
        "return_bias": float(np.mean(predicted - actual_array)),
        "growth_mae_pp": float(np.mean(absolute_growth_error)),
        "growth_rmse_pp": float(np.sqrt(np.mean(growth_error**2))),
        "growth_bias_pp": float(np.mean(growth_error)),
        "growth_median_ae_pp": float(np.median(absolute_growth_error)),
        "growth_p90_ae_pp": float(np.quantile(absolute_growth_error, 0.90)),
        "growth_magnitude_mae_pp": float(np.mean(np.abs(magnitude_error))),
        "growth_magnitude_bias_pp": float(np.mean(magnitude_error)),
    }
    if entry_open is not None and exit_close is not None:
        entry = np.asarray(entry_open, dtype=float)
        actual_exit = np.asarray(exit_close, dtype=float)
        predicted_exit = entry * np.exp(predicted)
        price_change_error = (predicted_exit - entry) - (actual_exit - entry)
        metrics.update(
            {
                "price_change_mae_cny_per_gram": float(
                    np.mean(np.abs(price_change_error))
                ),
                "price_change_rmse_cny_per_gram": float(
                    np.sqrt(np.mean(price_change_error**2))
                ),
                "price_change_bias_cny_per_gram": float(
                    np.mean(price_change_error)
                ),
            }
        )
    return metrics


def fit_and_predict(
    template: Pipeline,
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    features: list[str],
) -> tuple[Pipeline, np.ndarray]:
    model = clone(template)
    model.fit(fit_frame[features], fit_frame["target"])
    return model, np.asarray(model.predict(predict_frame[features]), dtype=float)


def experiment_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the same frozen frames used by the current horizon-one work."""
    return load_frames()
