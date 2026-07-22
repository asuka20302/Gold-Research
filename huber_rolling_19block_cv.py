"""Leakage-free fixed-size rolling validation for the selected Huber model.

The available horizon-one modelling rows are divided into 19 equal consecutive
blocks.  Ten models are then fitted:

    model 1: train blocks 1-9,   test block 10
    model 2: train blocks 2-10,  test block 11
    ...
    model 10: train blocks 10-18, test block 19

Every model therefore receives the same number of training and test rows, and
every training date precedes its test dates.  If the row count is not divisible
by 19, the oldest remainder rows are excluded from this fairness diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from advanced_experiment_utils import (
    PROJECT,
    amount_metrics,
    experiment_frames,
    selected_templates,
)
from huber_10fold_blocked_cv import prediction_ledger


N_BLOCKS = 19
TRAIN_BLOCKS = 9
TEST_MODELS = 10
OUTPUT = PROJECT / "outputs" / "huber_rolling_19block"
AMOUNT_COLUMNS = [
    "growth_mae_pp",
    "growth_rmse_pp",
    "growth_bias_pp",
    "growth_median_ae_pp",
    "growth_p90_ae_pp",
    "growth_magnitude_mae_pp",
    "growth_magnitude_bias_pp",
    "price_change_mae_cny_per_gram",
    "price_change_rmse_cny_per_gram",
    "price_change_bias_cny_per_gram",
]


def equal_chronological_blocks(
    frame: pd.DataFrame,
    n_blocks: int = N_BLOCKS,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Return equally sized chronological blocks and the oldest remainder."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least two.")
    ordered = frame.sort_index()
    if ordered.index.has_duplicates:
        raise ValueError("The modelling index must not contain duplicate dates.")
    block_size = len(ordered) // n_blocks
    if block_size == 0:
        raise ValueError("There are fewer observations than requested blocks.")

    used_rows = block_size * n_blocks
    remainder_rows = len(ordered) - used_rows
    excluded = ordered.iloc[:remainder_rows].copy()
    used = ordered.iloc[remainder_rows:].copy()
    blocks = [
        used.iloc[number * block_size : (number + 1) * block_size].copy()
        for number in range(n_blocks)
    ]
    if len({len(block) for block in blocks}) != 1:
        raise RuntimeError("The chronological blocks are not equal in size.")
    for earlier, later in zip(blocks, blocks[1:]):
        if earlier.index.max() >= later.index.min():
            raise RuntimeError("Chronological blocks overlap or are out of order.")
    return excluded, blocks


def rolling_windows(
    blocks: list[pd.DataFrame],
    train_blocks: int = TRAIN_BLOCKS,
) -> list[tuple[int, list[pd.DataFrame], pd.DataFrame]]:
    """Create one fixed-size past-only training window per later test block."""
    model_count = len(blocks) - train_blocks
    if model_count < 1:
        raise ValueError("At least one test block is required.")
    windows: list[tuple[int, list[pd.DataFrame], pd.DataFrame]] = []
    for model_number in range(1, model_count + 1):
        start = model_number - 1
        training = blocks[start : start + train_blocks]
        test = blocks[start + train_blocks]
        if training[-1].index.max() >= test.index.min():
            raise RuntimeError("A rolling training window reaches into its test block.")
        windows.append((model_number, training, test))
    return windows


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Summarize coefficient dispersion and sign agreement across models."""
    rows: list[dict[str, object]] = []
    for feature, group in coefficients.groupby("feature", sort=True):
        values = group["standardized_coefficient"].to_numpy(float)
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        zero = int(np.sum(values == 0))
        rows.append(
            {
                "feature": feature,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "positive_models": positive,
                "negative_models": negative,
                "zero_models": zero,
                "dominant_sign_share": float(max(positive, negative, zero) / len(values)),
            }
        )
    return pd.DataFrame(rows)


def save_charts(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    """Save amount-focused prediction and per-window error comparisons."""
    dated = predictions.copy()
    dated["trade_date"] = pd.to_datetime(dated["trade_date"])
    tail = dated.tail(300)

    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(
        tail["trade_date"],
        tail["actual_growth_pct"],
        label="actual growth",
        linewidth=1.2,
    )
    axes[0].plot(
        tail["trade_date"],
        tail["predicted_growth_pct"],
        label="predicted growth",
        linewidth=1.2,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Leakage-free rolling Huber growth: latest 300 test rows")
    axes[0].set_ylabel("ordinary growth (%)")
    axes[0].legend()

    actual = dated["actual_growth_pct"].to_numpy(float)
    predicted = dated["predicted_growth_pct"].to_numpy(float)
    limit = float(
        np.quantile(np.abs(np.concatenate([actual, predicted])), 0.99)
    )
    axes[1].scatter(actual, predicted, alpha=0.35, s=14)
    axes[1].plot([-limit, limit], [-limit, limit], linestyle="--", color="black")
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_title("Actual versus predicted growth (99% display range)")
    axes[1].set_xlabel("actual growth (%)")
    axes[1].set_ylabel("predicted growth (%)")
    figure.savefig(OUTPUT / "rolling_prediction_comparison.png", dpi=160)
    plt.close(figure)

    positions = np.arange(len(fold_metrics))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        fold_metrics["growth_mae_pp"],
        width,
        label="growth MAE",
    )
    axis.bar(
        positions + width / 2,
        fold_metrics["growth_rmse_pp"],
        width,
        label="growth RMSE",
    )
    axis.set_xticks(positions, fold_metrics["model_number"].astype(int))
    axis.set_xlabel("rolling model / test window")
    axis.set_ylabel("error (percentage points)")
    axis.set_title("Error by equal-size leakage-free test window")
    axis.legend()
    figure.savefig(OUTPUT / "rolling_fold_error_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _, _, pretest, test = experiment_frames()
    all_data = pd.concat([pretest, test]).sort_index()
    if all_data.index.has_duplicates:
        duplicates = all_data.index[all_data.index.duplicated()].unique()
        raise RuntimeError(f"Duplicate model dates found: {list(duplicates[:5])}")

    excluded, blocks = equal_chronological_blocks(all_data)
    windows = rolling_windows(blocks)
    if len(windows) != TEST_MODELS:
        raise RuntimeError(
            f"Expected {TEST_MODELS} rolling models but obtained {len(windows)}."
        )

    templates, feature_map = selected_templates()
    template = templates["Huber"]
    features = feature_map["Huber"]
    expected_train_rows = TRAIN_BLOCKS * len(blocks[0])
    expected_test_rows = len(blocks[0])

    fold_rows: list[dict[str, object]] = []
    ledgers: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []

    for model_number, training_blocks, test_block in windows:
        training_frame = pd.concat(training_blocks).sort_index()
        if len(training_frame) != expected_train_rows:
            raise RuntimeError("A model received an unequal training sample.")
        if len(test_block) != expected_test_rows:
            raise RuntimeError("A model received an unequal test sample.")
        if training_frame.index.max() >= test_block.index.min():
            raise RuntimeError("Training dates are not strictly before test dates.")
        if pd.to_datetime(training_frame["exit_date"]).max() > test_block.index.min():
            raise RuntimeError(
                "A training target was not observable by the first test signal date."
            )

        training_block_start = model_number
        training_block_end = model_number + TRAIN_BLOCKS - 1
        test_block_number = model_number + TRAIN_BLOCKS
        print(
            f"Model {model_number:02d}: train blocks "
            f"{training_block_start:02d}-{training_block_end:02d} "
            f"({len(training_frame)} rows), test block {test_block_number:02d} "
            f"({len(test_block)} rows), dates "
            f"{test_block.index.min().date()} to {test_block.index.max().date()}"
        )

        model = clone(template)
        model.fit(training_frame[features], training_frame["target"])
        prediction = np.asarray(model.predict(test_block[features]), dtype=float)
        model_path = models_dir / f"huber_rolling_model_{model_number:02d}.joblib"
        joblib.dump(model, model_path)

        metrics = amount_metrics(
            test_block["target"],
            prediction,
            test_block["entry_open"],
            test_block["exit_close"],
        )
        fold_rows.append(
            {
                "model_number": model_number,
                "training_block_start": training_block_start,
                "training_block_end": training_block_end,
                "test_block": test_block_number,
                "train_rows": len(training_frame),
                "test_rows": len(test_block),
                "train_start": training_frame.index.min(),
                "train_end": training_frame.index.max(),
                "latest_training_exit_date": pd.to_datetime(
                    training_frame["exit_date"]
                ).max(),
                "test_start": test_block.index.min(),
                "test_end": test_block.index.max(),
                "training_dates_strictly_before_test": True,
                "training_labels_available_by_test_start": True,
                "model_file": model_path.relative_to(PROJECT).as_posix(),
                **{column: metrics[column] for column in AMOUNT_COLUMNS},
            }
        )

        ledger = prediction_ledger(model_number, test_block, prediction).rename(
            columns={"fold": "model_number"}
        )
        ledger.insert(1, "test_block", test_block_number)
        ledgers.append(ledger)

        model_coefficients = np.asarray(
            model.named_steps["model"].coef_, dtype=float
        ).ravel()
        for feature, coefficient in zip(features, model_coefficients):
            coefficient_rows.append(
                {
                    "model_number": model_number,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                }
            )
        coefficient_rows.append(
            {
                "model_number": model_number,
                "feature": "intercept",
                "standardized_coefficient": float(
                    model.named_steps["model"].intercept_
                ),
            }
        )

    predictions = pd.concat(ledgers, ignore_index=True).sort_values("trade_date")
    if len(predictions) != TEST_MODELS * expected_test_rows:
        raise RuntimeError("The rolling prediction ledger has an unexpected size.")
    if predictions["trade_date"].duplicated().any():
        raise RuntimeError("A test date received more than one rolling prediction.")

    overall_metrics = amount_metrics(
        predictions["actual_log_return"],
        predictions["predicted_log_return"],
        predictions["entry_open_cny_per_gram"],
        predictions["actual_exit_close_cny_per_gram"],
    )
    overall_row = {
        "models_trained": TEST_MODELS,
        "chronological_blocks": N_BLOCKS,
        "training_blocks_per_model": TRAIN_BLOCKS,
        "block_rows": expected_test_rows,
        "train_rows_per_model": expected_train_rows,
        "test_rows_per_model": expected_test_rows,
        "total_oos_rows": len(predictions),
        "oldest_remainder_rows_excluded": len(excluded),
        **{column: overall_metrics[column] for column in AMOUNT_COLUMNS},
    }

    assignments: list[dict[str, object]] = []
    for block_number, block in enumerate(blocks, start=1):
        test_model = block_number - TRAIN_BLOCKS if block_number > TRAIN_BLOCKS else None
        for trade_date in block.index:
            assignments.append(
                {
                    "trade_date": trade_date,
                    "block": block_number,
                    "test_model_number": test_model,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    stability = coefficient_stability(coefficients)
    fold_metrics.to_csv(OUTPUT / "rolling_fold_metrics.csv", index=False)
    pd.DataFrame([overall_row]).to_csv(
        OUTPUT / "rolling_overall_metrics.csv", index=False
    )
    predictions.to_csv(OUTPUT / "rolling_oos_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT / "rolling_coefficients.csv", index=False)
    stability.to_csv(OUTPUT / "rolling_coefficient_stability.csv", index=False)
    pd.DataFrame(assignments).to_csv(
        OUTPUT / "rolling_block_assignments.csv", index=False
    )
    excluded.reset_index(names="trade_date").assign(
        exclusion_reason="oldest remainder removed to make 19 equal blocks"
    ).to_csv(OUTPUT / "rolling_excluded_remainder.csv", index=False)
    save_charts(predictions, fold_metrics)

    metadata = {
        "procedure": "fixed-size leakage-free 19-block rolling validation",
        "models_trained": TEST_MODELS,
        "chronological_blocks": N_BLOCKS,
        "training_blocks_per_model": TRAIN_BLOCKS,
        "block_rows": expected_test_rows,
        "training_rows_per_model": expected_train_rows,
        "test_rows_per_model": expected_test_rows,
        "available_rows": len(all_data),
        "used_rows": sum(len(block) for block in blocks),
        "excluded_oldest_remainder_rows": len(excluded),
        "out_of_sample_prediction_rows": len(predictions),
        "features": features,
        "model_parameters": template.named_steps["model"].get_params(),
        "preprocessing": ["median imputer", "standard scaler"],
        "hyperparameters_fixed_across_models": True,
        "future_data_used_for_training": False,
        "event_exclusions_applied": True,
        "interpretation": (
            "Each model uses exactly nine immediately preceding equal-sized "
            "blocks and predicts the next block. Results are comparable across "
            "windows and contain no later-date training observations."
        ),
    }
    with (OUTPUT / "rolling_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)

    print("\nOverall leakage-free amount metrics:")
    print(pd.DataFrame([overall_row]).to_string(index=False))
    print(f"\nSaved 10 rolling models under: {models_dir.resolve()}")
    print(f"All outputs: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
