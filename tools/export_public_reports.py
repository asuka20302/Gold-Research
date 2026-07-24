"""Export compact public reports without machine-specific absolute paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "outputs"
REPORTS = PROJECT / "reports"

CSV_REPORTS = {
    OUTPUTS / "horizon_comparison" / "eligible_horizon_models.csv": (
        "eligible_horizon_models.csv"
    ),
    OUTPUTS / "huber_10fold_blocked" / "huber_10fold_fold_metrics.csv": (
        "huber_10fold_fold_metrics.csv"
    ),
    OUTPUTS / "huber_10fold_blocked" / "huber_10fold_overall_metrics.csv": (
        "huber_10fold_overall_metrics.csv"
    ),
    OUTPUTS
    / "huber_10fold_blocked"
    / "huber_10fold_coefficient_summary.csv": (
        "huber_10fold_coefficient_summary.csv"
    ),
    OUTPUTS / "huber_loss_minimization" / "loss_minimization_summary.csv": (
        "huber_loss_minimization_summary.csv"
    ),
    OUTPUTS
    / "huber_loss_minimization"
    / "loss_minimizing_huber_coefficients.csv": (
        "huber_loss_minimizing_coefficients.csv"
    ),
    OUTPUTS / "huber_rolling_19block" / "rolling_overall_metrics.csv": (
        "huber_rolling_19block_overall_metrics.csv"
    ),
    OUTPUTS / "huber_rolling_19block" / "rolling_fold_metrics.csv": (
        "huber_rolling_19block_fold_metrics.csv"
    ),
    OUTPUTS
    / "huber_rolling_19block"
    / "rolling_coefficient_stability.csv": (
        "huber_rolling_19block_coefficient_stability.csv"
    ),
    OUTPUTS / "locked_block19_ensemble" / "locked_block19_metrics.csv": (
        "locked_block19_metrics.csv"
    ),
    OUTPUTS / "locked_block19_ensemble" / "frozen_voting_weights.csv": (
        "locked_block19_weights.csv"
    ),
    OUTPUTS / "locked_block19_pnl" / "locked_block19_strategy_summary.csv": (
        "locked_block19_pnl_summary.csv"
    ),
    OUTPUTS / "locked_block19_pnl" / "locked_block19_cost_sensitivity.csv": (
        "locked_block19_cost_sensitivity.csv"
    ),
    OUTPUTS / "locked_block19_pnl" / "development_primary_block_stability.csv": (
        "locked_block19_development_stability.csv"
    ),
}

JSON_REPORTS = {
    OUTPUTS / "horizon_comparison" / "selected_horizon.json": (
        "selected_horizon.json"
    ),
    OUTPUTS / "huber_rolling_19block" / "rolling_metadata.json": (
        "huber_rolling_19block_metadata.json"
    ),
    OUTPUTS / "locked_block19_ensemble" / "locked_block19_metadata.json": (
        "locked_block19_metadata.json"
    ),
    OUTPUTS / "locked_block19_pnl" / "selected_trading_rules.json": (
        "locked_block19_trading_rules.json"
    ),
}

BINARY_REPORTS = {
    OUTPUTS / "huber_rolling_19block" / "rolling_prediction_comparison.png": (
        "huber_rolling_19block_predictions.png"
    ),
    OUTPUTS / "huber_rolling_19block" / "rolling_fold_error_comparison.png": (
        "huber_rolling_19block_fold_errors.png"
    ),
    OUTPUTS / "locked_block19_ensemble" / "locked_block19_comparison.png": (
        "locked_block19_comparison.png"
    ),
    OUTPUTS / "locked_block19_pnl" / "locked_block19_equity_comparison.png": (
        "locked_block19_pnl_equity.png"
    ),
}


def portable_string(value: str) -> str:
    """Replace the local repository prefix in a path-like string."""
    normalized = value.replace("\\", "/")
    project = PROJECT.as_posix()
    position = normalized.lower().find(project.lower())
    if position < 0:
        return value
    relative = normalized[position + len(project) :].lstrip("/")
    return relative


def portable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: portable_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_json(item) for item in value]
    if isinstance(value, str):
        return portable_string(value)
    return value


def export_csv(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    for column in frame.columns:
        dtype = frame[column].dtype
        if not (is_object_dtype(dtype) or is_string_dtype(dtype)):
            continue
        frame[column] = frame[column].map(
            lambda value: portable_string(value) if isinstance(value, str) else value
        )
    frame.to_csv(destination, index=False)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    for source, filename in CSV_REPORTS.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing report source: {source}")
        export_csv(source, REPORTS / filename)

    for source, filename in JSON_REPORTS.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing report source: {source}")
        with source.open("r", encoding="utf-8") as file:
            content = portable_json(json.load(file))
        with (REPORTS / filename).open("w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)

    for source, filename in BINARY_REPORTS.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing report source: {source}")
        shutil.copy2(source, REPORTS / filename)

    print(f"Exported {len(CSV_REPORTS) + len(JSON_REPORTS) + len(BINARY_REPORTS)} reports.")


if __name__ == "__main__":
    main()
