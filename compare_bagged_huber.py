"""Tune moving-block bagging for the selected horizon-1 Huber model."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import gold_model_comparison as gm
from interaction_regime_features import (
    fit_regime_thresholds,
    transform_interactions_and_regimes,
)
from tushare_factors import build_tushare_factor_features


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs" / "bagging_h1"
START_DATE = "20150101"
END_DATE = "20260703"
HORIZON = 1
BLOCK_SIZES = [5, 10, 20, 40]
ESTIMATOR_COUNTS = [50, 100, 200]
RANDOM_STATE = 42


def metric_row(
    name: str,
    actual: pd.Series,
    prediction: np.ndarray,
    member_std: float = 0.0,
) -> dict[str, object]:
    zero_mse = mean_squared_error(actual, np.zeros(len(actual)))
    mse = mean_squared_error(actual, prediction)
    return {
        "candidate": name,
        "validation_mse": mse,
        "relative_mse_zero": mse / zero_mse,
        "validation_correlation": gm.safe_correlation(actual, prediction),
        "validation_direction_accuracy": float(
            np.mean(np.sign(prediction) == np.sign(actual))
        ),
        "mean_member_prediction_std": member_std,
    }


def load_frames() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    raw = gm.download_gold_data(
        "Au99.99", START_DATE, END_DATE, PROJECT / "data", refresh=False
    )
    factors = build_tushare_factor_features(
        START_DATE, END_DATE, PROJECT / "data", refresh=False
    )
    data = gm.build_dataset(raw, HORIZON, factors)
    exclusions = gm.load_event_exclusions(PROJECT / "data/event_exclusions.csv")
    data, _, _ = gm.apply_event_exclusions(data, exclusions)
    train, validation, pretest, test = gm.chronological_split(data, HORIZON)
    thresholds = fit_regime_thresholds(train)
    return tuple(
        transform_interactions_and_regimes(frame, thresholds)
        for frame in [train, validation, pretest, test]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, pretest, test = load_frames()
    selected = pd.read_csv(
        PROJECT / "outputs/horizon_comparison/h1/selected_models.csv"
    )
    huber_row = selected.loc[selected["model"] == "Huber"].iloc[0]
    features = str(huber_row["active_features"]).split(",")
    stored_parameters = ast.literal_eval(str(huber_row["selected_parameters"]))
    allowed = {
        "epsilon", "alpha", "fit_intercept", "max_iter", "tol", "warm_start"
    }
    huber_parameters = {
        key: value
        for key, value in stored_parameters.items()
        if key in allowed
    }
    base_template = gm.build_pipeline(HuberRegressor(**huber_parameters))

    base_train = gm.clone(base_template)
    base_train.fit(train[features], train["target"])
    base_prediction = base_train.predict(validation[features])
    rows = [
        {
            **metric_row(
                "UnbaggedHuber", validation["target"], base_prediction
            ),
            "block_size": 0,
            "n_estimators": 1,
        }
    ]

    max_estimators = max(ESTIMATOR_COUNTS)
    for block_size in BLOCK_SIZES:
        print(
            f"Fitting block size {block_size} with "
            f"{max_estimators} estimators..."
        )
        bag = gm.BlockBaggedRegressor(
            base_model=base_template,
            n_estimators=max_estimators,
            block_size=block_size,
            random_state=RANDOM_STATE,
        )
        bag.fit(train[features], train["target"])
        member_predictions = bag.predict_members(validation[features])
        for count in ESTIMATOR_COUNTS:
            selected_members = member_predictions[:, :count]
            prediction = selected_members.mean(axis=1)
            rows.append(
                {
                    **metric_row(
                        f"BaggedHuber_b{block_size}_n{count}",
                        validation["target"],
                        prediction,
                        member_std=float(
                            selected_members.std(axis=1, ddof=1).mean()
                        ),
                    ),
                    "block_size": block_size,
                    "n_estimators": count,
                }
            )

    validation_grid = pd.DataFrame(rows)
    baseline = validation_grid.iloc[0]
    candidates = validation_grid.iloc[1:].copy()
    candidates["mse_improves"] = (
        candidates["validation_mse"] < baseline["validation_mse"]
    )
    candidates["correlation_improves"] = (
        candidates["validation_correlation"]
        > baseline["validation_correlation"]
    )
    candidates["direction_not_worse"] = (
        candidates["validation_direction_accuracy"]
        >= baseline["validation_direction_accuracy"]
    )
    candidates["passes_all_rules"] = (
        candidates["mse_improves"]
        & candidates["correlation_improves"]
        & candidates["direction_not_worse"]
    )
    validation_grid = validation_grid.merge(
        candidates[
            [
                "candidate", "mse_improves", "correlation_improves",
                "direction_not_worse", "passes_all_rules",
            ]
        ],
        on="candidate",
        how="left",
    )
    validation_grid[
        [
            "mse_improves", "correlation_improves", "direction_not_worse",
            "passes_all_rules",
        ]
    ] = validation_grid[
        [
            "mse_improves", "correlation_improves", "direction_not_worse",
            "passes_all_rules",
        ]
    ].fillna(False)

    passing = validation_grid[validation_grid["passes_all_rules"]]
    if passing.empty:
        winner = validation_grid.iloc[0]
        selected_name = "UnbaggedHuber"
    else:
        winner = passing.sort_values(
            ["validation_mse", "n_estimators", "block_size"]
        ).iloc[0]
        selected_name = "BaggedHuber"

    base_final = gm.clone(base_template)
    base_final.fit(pretest[features], pretest["target"])
    models: dict[str, object] = {"UnbaggedHuber": base_final}
    if selected_name == "BaggedHuber":
        bag_final = gm.BlockBaggedRegressor(
            base_model=base_template,
            n_estimators=int(winner["n_estimators"]),
            block_size=int(winner["block_size"]),
            random_state=RANDOM_STATE,
        )
        bag_final.fit(pretest[features], pretest["target"])
        models["BaggedHuber"] = bag_final

    feature_map = {name: features for name in models}
    predictions, evaluation, test_metrics, coefficients = gm.evaluate_models(
        models, test, HORIZON, model_features=feature_map
    )
    residuals = gm.residual_diagnostics(evaluation, models)

    validation_grid.to_csv(OUTPUT / "bagging_validation_grid.csv", index=False)
    predictions.to_csv(OUTPUT / "bagging_test_predictions.csv")
    test_metrics.to_csv(OUTPUT / "bagging_test_metrics.csv", index=False)
    coefficients.to_csv(OUTPUT / "bagging_mean_coefficients.csv")
    residuals.to_csv(OUTPUT / "bagging_residual_diagnostics.csv", index=False)
    with (OUTPUT / "selected_bagging.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": selected_name,
                "block_size": int(winner["block_size"]),
                "n_estimators": int(winner["n_estimators"]),
                "validation_mse": float(winner["validation_mse"]),
                "validation_correlation": float(
                    winner["validation_correlation"]
                ),
                "validation_direction_accuracy": float(
                    winner["validation_direction_accuracy"]
                ),
                "feature_count": len(features),
                "selection_data": "validation_only",
                "bootstrap": "moving_block_with_replacement",
                "test_warning": (
                    "Test metrics are event-filtered diagnostics and were "
                    "computed only after the bagging configuration was frozen."
                ),
            },
            f,
            indent=2,
        )

    print("\nBest validation configuration:")
    print(winner.to_string())
    print("\nTest metrics after freezing the validation decision:")
    print(test_metrics.to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
