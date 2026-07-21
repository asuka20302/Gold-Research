"""Train ten Huber models with consecutive blocked 10-fold rotation.

This implements the requested diagnostic exactly: split all model rows into ten
consecutive pieces, train on nine pieces, test on the remaining piece, and
rotate the held-out piece.  For early held-out blocks, the nine training blocks
contain later dates, so these results are cross-validation diagnostics and not
a leakage-free historical trading simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from advanced_experiment_utils import (
    PROJECT,
    amount_metrics,
    experiment_frames,
    selected_templates,
)


N_SPLITS = 10
OUTPUT = PROJECT / "outputs" / "huber_10fold_blocked"
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


def prediction_ledger(
    fold: int,
    frame: pd.DataFrame,
    predicted_log_return: np.ndarray,
) -> pd.DataFrame:
    actual_log_return = frame["target"].to_numpy(float)
    actual_growth = np.expm1(actual_log_return) * 100.0
    predicted_growth = np.expm1(predicted_log_return) * 100.0
    growth_error = predicted_growth - actual_growth
    magnitude_error = np.abs(predicted_growth) - np.abs(actual_growth)
    entry_open = frame["entry_open"].to_numpy(float)
    actual_exit = frame["exit_close"].to_numpy(float)
    predicted_exit = entry_open * np.exp(predicted_log_return)
    price_error = predicted_exit - actual_exit
    amount_assessment = np.where(
        magnitude_error > 1e-12,
        "overestimated move size",
        np.where(
            magnitude_error < -1e-12,
            "underestimated move size",
            "matched move size",
        ),
    )
    return pd.DataFrame(
        {
            "fold": fold,
            "trade_date": frame.index,
            "entry_date": frame["entry_date"].to_numpy(),
            "exit_date": frame["exit_date"].to_numpy(),
            "entry_open_cny_per_gram": entry_open,
            "actual_exit_close_cny_per_gram": actual_exit,
            "predicted_exit_close_cny_per_gram": predicted_exit,
            "actual_log_return": actual_log_return,
            "predicted_log_return": predicted_log_return,
            "actual_growth_pct": actual_growth,
            "predicted_growth_pct": predicted_growth,
            "growth_error_percentage_points": growth_error,
            "absolute_growth_error_percentage_points": np.abs(growth_error),
            "actual_move_size_pct": np.abs(actual_growth),
            "predicted_move_size_pct": np.abs(predicted_growth),
            "move_size_error_percentage_points": magnitude_error,
            "actual_change_cny_per_gram": actual_exit - entry_open,
            "predicted_change_cny_per_gram": predicted_exit - entry_open,
            "change_error_cny_per_gram": price_error,
            "absolute_change_error_cny_per_gram": np.abs(price_error),
            "amount_assessment": amount_assessment,
        }
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train, validation, pretest, test = experiment_frames()
    # pretest already equals train + validation.  Adding test reconstructs all
    # event-filtered model rows without duplicating the training rows.
    all_data = pd.concat([pretest, test]).sort_index()
    if all_data.index.has_duplicates:
        duplicates = all_data.index[all_data.index.duplicated()].unique()
        raise RuntimeError(f"Duplicate model dates found: {list(duplicates[:5])}")

    templates, feature_map = selected_templates()
    template = templates["Huber"]
    features = feature_map["Huber"]
    row_pieces = np.array_split(np.arange(len(all_data)), N_SPLITS)

    fold_rows: list[dict[str, object]] = []
    ledgers: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    for fold, held_out_indices in enumerate(row_pieces, start=1):
        training_indices = np.concatenate(
            [piece for number, piece in enumerate(row_pieces, start=1) if number != fold]
        )
        training_frame = all_data.iloc[training_indices]
        held_out_frame = all_data.iloc[held_out_indices]

        print(
            f"Fold {fold:02d}: train={len(training_frame)}, "
            f"test={len(held_out_frame)}, "
            f"test_dates={held_out_frame.index.min().date()} to "
            f"{held_out_frame.index.max().date()}"
        )
        model = clone(template)
        model.fit(training_frame[features], training_frame["target"])
        prediction = np.asarray(
            model.predict(held_out_frame[features]), dtype=float
        )
        model_path = models_dir / f"huber_fold_{fold:02d}.joblib"
        joblib.dump(model, model_path)

        metrics = amount_metrics(
            held_out_frame["target"],
            prediction,
            held_out_frame["entry_open"],
            held_out_frame["exit_close"],
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(training_frame),
                "test_rows": len(held_out_frame),
                "test_start": held_out_frame.index.min(),
                "test_end": held_out_frame.index.max(),
                "training_uses_dates_after_test": bool(
                    training_frame.index.max() > held_out_frame.index.max()
                ),
                "model_file": str(model_path.resolve()),
                **{column: metrics[column] for column in AMOUNT_COLUMNS},
            }
        )
        ledgers.append(prediction_ledger(fold, held_out_frame, prediction))

        coefficients = np.asarray(
            model.named_steps["model"].coef_, dtype=float
        ).ravel()
        for feature, coefficient in zip(features, coefficients):
            coefficient_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                }
            )
        coefficient_rows.append(
            {
                "fold": fold,
                "feature": "intercept",
                "standardized_coefficient": float(
                    model.named_steps["model"].intercept_
                ),
            }
        )
        for trade_date in held_out_frame.index:
            assignment_rows.append(
                {"trade_date": trade_date, "held_out_fold": fold}
            )

    ledger = pd.concat(ledgers, ignore_index=True).sort_values("trade_date")
    if len(ledger) != len(all_data):
        raise RuntimeError(
            f"Expected {len(all_data)} OOF rows but obtained {len(ledger)}."
        )
    if ledger["trade_date"].duplicated().any():
        raise RuntimeError("A date received more than one held-out prediction.")

    overall = amount_metrics(
        ledger["actual_log_return"],
        ledger["predicted_log_return"],
        ledger["entry_open_cny_per_gram"],
        ledger["actual_exit_close_cny_per_gram"],
    )
    overall_row = {
        "models_trained": N_SPLITS,
        "total_oof_rows": len(ledger),
        **{column: overall[column] for column in AMOUNT_COLUMNS},
    }

    fold_metrics = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficient_summary = (
        coefficients.groupby("feature")["standardized_coefficient"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("feature")
    )
    fold_metrics.to_csv(OUTPUT / "huber_10fold_fold_metrics.csv", index=False)
    pd.DataFrame([overall_row]).to_csv(
        OUTPUT / "huber_10fold_overall_metrics.csv", index=False
    )
    ledger.to_csv(OUTPUT / "huber_10fold_oof_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT / "huber_10fold_coefficients.csv", index=False)
    coefficient_summary.to_csv(
        OUTPUT / "huber_10fold_coefficient_summary.csv", index=False
    )
    pd.DataFrame(assignment_rows).to_csv(
        OUTPUT / "huber_10fold_fold_assignments.csv", index=False
    )

    metadata = {
        "procedure": "consecutive blocked 10-fold rotation",
        "models_trained": N_SPLITS,
        "rows": len(all_data),
        "features": features,
        "model_parameters": template.named_steps["model"].get_params(),
        "preprocessing": ["median imputer", "standard scaler"],
        "event_exclusions_applied": True,
        "warning": (
            "Nine-of-ten training blocks can contain dates after an early "
            "held-out block. These are cross-validation diagnostics, not a "
            "leakage-free trading backtest."
        ),
    }
    with (OUTPUT / "huber_10fold_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)

    print("\nOverall amount-based OOF metrics:")
    print(pd.DataFrame([overall_row]).to_string(index=False))
    print(f"\nSaved 10 models under: {models_dir.resolve()}")
    print(f"All outputs: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
