"""Train five equal-footing models and evaluate once on locked Block 19.

The event-filtered horizon-one dataset is divided into 19 equal chronological
blocks.  Blocks 1-18 are the development area.  Block 19 is excluded from
training and from voting-weight estimation, then used as the common test period
for the five frozen models and their continuous weighted-average forecast.

Development predictions use the same nine-block rolling rule as the final fit:

    train blocks 1-9, score block 10
    ...
    train blocks 9-17, score block 18

The five voting weights are proportional to inverse development growth MAE.
This implements "larger historical amount error -> smaller weight" without
using Block 19 outcomes.
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

import gold_model_comparison as gm
from advanced_experiment_utils import (
    PROJECT,
    amount_metrics,
    experiment_frames,
    selected_templates,
)
from huber_10fold_blocked_cv import prediction_ledger
from huber_rolling_19block_cv import equal_chronological_blocks


OUTPUT = PROJECT / "outputs" / "locked_block19_ensemble"
MODEL_NAMES = ["OLS", "Ridge", "Lasso", "ElasticNet", "BaggedHuber"]
TRAIN_BLOCKS = 9
DEVELOPMENT_LAST_BLOCK = 18
HOLDOUT_BLOCK = 19
RANDOM_STATE = 42
METRIC_COLUMNS = [
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


def build_model(name: str, templates: dict[str, object], random_state: int) -> object:
    """Construct one of the five frozen candidate estimators."""
    if name == "BaggedHuber":
        return gm.BlockBaggedRegressor(
            base_model=templates["Huber"],
            n_estimators=50,
            block_size=20,
            random_state=random_state,
        )
    return clone(templates[name])


def inverse_mae_weights(
    predictions: pd.DataFrame,
    actual_log_return: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return growth MAE and normalized inverse-MAE weights for each model."""
    actual_growth = np.expm1(actual_log_return.to_numpy(float)) * 100.0
    model_mae = {}
    for name in MODEL_NAMES:
        predicted_growth = np.expm1(predictions[name].to_numpy(float)) * 100.0
        model_mae[name] = float(np.mean(np.abs(predicted_growth - actual_growth)))
    mae = pd.Series(model_mae, name="development_growth_mae_pp")
    inverse = 1.0 / np.maximum(mae.to_numpy(float), 1e-15)
    weights = pd.Series(inverse / inverse.sum(), index=mae.index, name="weight")
    return mae, weights


def development_windows(
    blocks: list[pd.DataFrame],
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Build nine equal-size past-only windows without touching Block 19."""
    if len(blocks) != HOLDOUT_BLOCK:
        raise ValueError(f"Expected {HOLDOUT_BLOCK} chronological blocks.")
    windows: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for score_block in range(TRAIN_BLOCKS + 1, DEVELOPMENT_LAST_BLOCK + 1):
        start = score_block - TRAIN_BLOCKS - 1
        training = pd.concat(blocks[start : start + TRAIN_BLOCKS]).sort_index()
        scoring = blocks[score_block - 1].copy()
        if training.index.max() >= scoring.index.min():
            raise RuntimeError("A development training window reaches its score block.")
        windows.append((score_block, training, scoring))
    return windows


def save_chart(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    dated = predictions.copy()
    dated["trade_date"] = pd.to_datetime(dated["trade_date"])
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(
        dated["trade_date"],
        dated["actual_growth_pct"],
        label="actual growth",
        linewidth=1.5,
        color="black",
    )
    for name in [*MODEL_NAMES, "WeightedEnsemble"]:
        axes[0].plot(
            dated["trade_date"],
            dated[f"{name}_predicted_growth_pct"],
            label=name,
            linewidth=1.0,
            alpha=0.8,
        )
    axes[0].axhline(0.0, color="grey", linewidth=0.8)
    axes[0].set_title("Locked Block 19: common actual and predicted growth")
    axes[0].set_ylabel("ordinary growth (%)")
    axes[0].legend(ncol=3)

    ordered = metrics.sort_values("growth_mae_pp")
    positions = np.arange(len(ordered))
    axes[1].bar(positions - 0.2, ordered["growth_mae_pp"], 0.4, label="MAE")
    axes[1].bar(positions + 0.2, ordered["growth_rmse_pp"], 0.4, label="RMSE")
    axes[1].set_xticks(positions, ordered["candidate"], rotation=20)
    axes[1].set_ylabel("growth error (percentage points)")
    axes[1].set_title("Common Block 19 amount-error comparison")
    axes[1].legend()
    figure.savefig(OUTPUT / "locked_block19_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_directory = OUTPUT / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    _, _, pretest, test = experiment_frames()
    all_data = pd.concat([pretest, test]).sort_index()
    excluded, blocks = equal_chronological_blocks(all_data, n_blocks=HOLDOUT_BLOCK)
    block_size = len(blocks[0])
    if any(len(block) != block_size for block in blocks):
        raise RuntimeError("Block sizes differ.")

    holdout = blocks[HOLDOUT_BLOCK - 1].copy()
    development = pd.concat(blocks[:DEVELOPMENT_LAST_BLOCK]).sort_index()
    if development.index.max() >= holdout.index.min():
        raise RuntimeError("Development observations reach into Block 19.")

    templates, feature_map = selected_templates()
    features = feature_map["Huber"]
    for model_name in ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]:
        if feature_map[model_name] != features:
            raise RuntimeError("All five candidates must use identical features.")

    oof_parts: list[pd.DataFrame] = []
    for score_block, training, scoring in development_windows(blocks):
        if len(training) != TRAIN_BLOCKS * block_size or len(scoring) != block_size:
            raise RuntimeError("Development windows are not equal in size.")
        if pd.to_datetime(training["exit_date"]).max() > scoring.index.min():
            raise RuntimeError("A development label is unavailable at score time.")
        part = pd.DataFrame(
            {
                "trade_date": scoring.index,
                "score_block": score_block,
                "actual_log_return": scoring["target"].to_numpy(float),
            }
        ).set_index("trade_date")
        for model_offset, name in enumerate(MODEL_NAMES):
            model = build_model(
                name,
                templates,
                random_state=RANDOM_STATE + score_block * 10 + model_offset,
            )
            model.fit(training[features], training["target"])
            part[name] = np.asarray(model.predict(scoring[features]), dtype=float)
        oof_parts.append(part)
        print(
            f"Development score block {score_block:02d}: "
            f"train={len(training)}, score={len(scoring)}"
        )

    development_oof = pd.concat(oof_parts).sort_index()
    if development_oof.index.min() < blocks[TRAIN_BLOCKS].index.min():
        raise RuntimeError("Unexpected development scoring date.")
    if development_oof.index.max() >= holdout.index.min():
        raise RuntimeError("Block 19 entered development predictions.")
    development_actual = development_oof["actual_log_return"]
    development_mae, weights = inverse_mae_weights(
        development_oof[MODEL_NAMES], development_actual
    )
    development_oof["WeightedEnsemble"] = (
        development_oof[MODEL_NAMES].to_numpy(float) @ weights.to_numpy(float)
    )

    final_training = pd.concat(blocks[9:18]).sort_index()
    if len(final_training) != TRAIN_BLOCKS * block_size:
        raise RuntimeError("Final models did not receive nine blocks.")
    if final_training.index.max() >= holdout.index.min():
        raise RuntimeError("Final training dates reach into Block 19.")
    if pd.to_datetime(final_training["exit_date"]).max() > holdout.index.min():
        raise RuntimeError("A final-training label is unavailable at holdout time.")

    holdout_predictions: dict[str, np.ndarray] = {}
    for model_offset, name in enumerate(MODEL_NAMES):
        model = build_model(name, templates, RANDOM_STATE + model_offset)
        model.fit(final_training[features], final_training["target"])
        holdout_predictions[name] = np.asarray(
            model.predict(holdout[features]), dtype=float
        )
        model_path = model_directory / f"{name}.joblib"
        joblib.dump(model, model_path)
    holdout_predictions["WeightedEnsemble"] = (
        np.column_stack([holdout_predictions[name] for name in MODEL_NAMES])
        @ weights.to_numpy(float)
    )

    metric_rows: list[dict[str, object]] = []
    for name, prediction in holdout_predictions.items():
        metric = amount_metrics(
            holdout["target"],
            prediction,
            holdout["entry_open"],
            holdout["exit_close"],
        )
        metric_rows.append(
            {
                "candidate": name,
                "candidate_type": (
                    "ensemble" if name == "WeightedEnsemble" else "individual"
                ),
                **{column: metric[column] for column in METRIC_COLUMNS},
            }
        )
    holdout_metrics = pd.DataFrame(metric_rows).sort_values("growth_mae_pp")
    individual_metrics = holdout_metrics[
        holdout_metrics["candidate_type"] == "individual"
    ]
    best_individual = individual_metrics.iloc[0]
    ensemble_result = holdout_metrics[
        holdout_metrics["candidate"] == "WeightedEnsemble"
    ].iloc[0]
    ensemble_beats_best_individual = bool(
        ensemble_result["growth_mae_pp"] < best_individual["growth_mae_pp"]
        and ensemble_result["growth_rmse_pp"] < best_individual["growth_rmse_pp"]
        and ensemble_result["price_change_mae_cny_per_gram"]
        < best_individual["price_change_mae_cny_per_gram"]
    )

    ledger = prediction_ledger(
        HOLDOUT_BLOCK,
        holdout,
        holdout_predictions["WeightedEnsemble"],
    ).rename(
        columns={
            "fold": "holdout_block",
            "predicted_log_return": "WeightedEnsemble_predicted_log_return",
            "predicted_growth_pct": "WeightedEnsemble_predicted_growth_pct",
            "predicted_exit_close_cny_per_gram": (
                "WeightedEnsemble_predicted_exit_close_cny_per_gram"
            ),
        }
    )
    for name in MODEL_NAMES:
        prediction = holdout_predictions[name]
        ledger[f"{name}_predicted_log_return"] = prediction
        ledger[f"{name}_predicted_growth_pct"] = np.expm1(prediction) * 100.0
        ledger[f"{name}_predicted_exit_close_cny_per_gram"] = (
            holdout["entry_open"].to_numpy(float) * np.exp(prediction)
        )

    weight_table = pd.DataFrame(
        {
            "model": MODEL_NAMES,
            "development_growth_mae_pp": development_mae.loc[MODEL_NAMES].values,
            "inverse_mae_weight": weights.loc[MODEL_NAMES].values,
        }
    )
    development_metric_rows = []
    for name in [*MODEL_NAMES, "WeightedEnsemble"]:
        metric = amount_metrics(
            development_actual,
            development_oof[name],
        )
        development_metric_rows.append(
            {
                "candidate": name,
                **{column: metric[column] for column in METRIC_COLUMNS[:7]},
            }
        )

    development_oof.reset_index().to_csv(
        OUTPUT / "development_oof_predictions.csv", index=False
    )
    pd.DataFrame(development_metric_rows).sort_values("growth_mae_pp").to_csv(
        OUTPUT / "development_oof_metrics.csv", index=False
    )
    weight_table.to_csv(OUTPUT / "frozen_voting_weights.csv", index=False)
    ledger.to_csv(OUTPUT / "locked_block19_predictions.csv", index=False)
    holdout_metrics.to_csv(OUTPUT / "locked_block19_metrics.csv", index=False)
    excluded.reset_index(names="trade_date").to_csv(
        OUTPUT / "excluded_oldest_remainder.csv", index=False
    )
    save_chart(ledger, holdout_metrics)

    metadata = {
        "design": "locked common Block 19 holdout",
        "available_rows": len(all_data),
        "equal_block_rows": block_size,
        "excluded_oldest_remainder_rows": len(excluded),
        "development_blocks": "1-18",
        "development_oof_score_blocks": "10-18",
        "development_oof_rows": len(development_oof),
        "final_training_blocks": "10-18",
        "final_training_rows_per_model": len(final_training),
        "holdout_block": HOLDOUT_BLOCK,
        "holdout_rows": len(holdout),
        "holdout_start": holdout.index.min(),
        "holdout_end": holdout.index.max(),
        "models": MODEL_NAMES,
        "features": features,
        "weight_rule": "normalized inverse development OOF growth MAE",
        "weights_frozen_before_block19_evaluation": True,
        "block19_used_for_training": False,
        "block19_used_for_weight_estimation": False,
        "best_locked_holdout_candidate": str(holdout_metrics.iloc[0]["candidate"]),
        "best_individual_candidate": str(best_individual["candidate"]),
        "weighted_ensemble_beats_best_individual": ensemble_beats_best_individual,
        "deployment_selection": (
            "WeightedEnsemble"
            if ensemble_beats_best_individual
            else str(best_individual["candidate"])
        ),
        "historical_exposure_warning": (
            "Block 19 outcomes were seen in an earlier Huber rolling diagnostic; "
            "this run enforces computational isolation but cannot restore pristine "
            "human-level blindness."
        ),
    }
    with (OUTPUT / "locked_block19_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)

    print("\nFrozen inverse-MAE voting weights:")
    print(weight_table.to_string(index=False))
    print("\nLocked Block 19 amount errors:")
    print(holdout_metrics.to_string(index=False))
    print(f"\nOutputs: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
