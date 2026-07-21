"""Tune moving-block bagging for all five reduced regression families."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

import gold_model_comparison as gm
from compare_bagged_huber import load_frames


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs" / "bagging_all_models_h1"
BLOCK_SIZES = [5, 10, 20, 40]
ESTIMATOR_COUNTS = [50, 100, 200]
RANDOM_STATE = 42

MODEL_CLASSES = {
    "OLS": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "Huber": HuberRegressor,
}


def build_template(model_name: str, parameter_text: str):
    parameter_dict = ast.literal_eval(parameter_text.split(";")[0])
    estimator_class = MODEL_CLASSES[model_name]
    allowed = estimator_class().get_params().keys()
    parameters = {
        key: value for key, value in parameter_dict.items() if key in allowed
    }
    return gm.build_pipeline(estimator_class(**parameters))


def amount_metrics(
    actual: pd.Series,
    prediction: np.ndarray,
) -> dict[str, float]:
    actual_array = actual.to_numpy(float)
    predicted = np.asarray(prediction, dtype=float)
    actual_std = float(np.std(actual_array))
    mse = mean_squared_error(actual_array, predicted)
    return {
        "mse": mse,
        "mae": mean_absolute_error(actual_array, predicted),
        "relative_mse_zero": mse
        / mean_squared_error(actual_array, np.zeros(len(actual_array))),
        "correlation": gm.safe_correlation(actual_array, predicted),
        "direction_accuracy": float(
            np.mean(np.sign(actual_array) == np.sign(predicted))
        ),
        "magnitude_mae": mean_absolute_error(
            np.abs(actual_array), np.abs(predicted)
        ),
        "magnitude_correlation": gm.safe_correlation(
            np.abs(actual_array), np.abs(predicted)
        ),
        "amplitude_ratio": (
            float(np.std(predicted)) / actual_std
            if actual_std > 0
            else np.nan
        ),
        "return_bias": float(np.mean(predicted - actual_array)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, pretest, test = load_frames()
    selected = pd.read_csv(
        PROJECT / "outputs/horizon_comparison/h1/selected_models.csv"
    )

    grid_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    templates = {}
    features_by_model = {}

    for _, row in selected.iterrows():
        model_name = str(row["model"])
        template = build_template(model_name, str(row["selected_parameters"]))
        templates[model_name] = template
        features = str(row["active_features"]).split(",")
        features_by_model[model_name] = features

        unbagged = gm.clone(template)
        unbagged.fit(train[features], train["target"])
        base_prediction = unbagged.predict(validation[features])
        baseline_metrics = amount_metrics(
            validation["target"], base_prediction
        )
        grid_rows.append(
            {
                "model": model_name,
                "candidate": "unbagged",
                "block_size": 0,
                "n_estimators": 1,
                "mean_member_prediction_std": 0.0,
                **baseline_metrics,
            }
        )

        for block_size in BLOCK_SIZES:
            print(
                f"{model_name}: fitting block={block_size}, "
                f"members={max(ESTIMATOR_COUNTS)}"
            )
            bag = gm.BlockBaggedRegressor(
                base_model=template,
                n_estimators=max(ESTIMATOR_COUNTS),
                block_size=block_size,
                random_state=RANDOM_STATE,
            )
            bag.fit(train[features], train["target"])
            members = bag.predict_members(validation[features])
            for count in ESTIMATOR_COUNTS:
                chosen = members[:, :count]
                grid_rows.append(
                    {
                        "model": model_name,
                        "candidate": "bagged",
                        "block_size": block_size,
                        "n_estimators": count,
                        "mean_member_prediction_std": float(
                            chosen.std(axis=1, ddof=1).mean()
                        ),
                        **amount_metrics(
                            validation["target"], chosen.mean(axis=1)
                        ),
                    }
                )

    grid = pd.DataFrame(grid_rows)
    chosen_configurations: dict[str, dict[str, object]] = {}
    for model_name, model_grid in grid.groupby("model", sort=False):
        baseline = model_grid[model_grid["candidate"] == "unbagged"].iloc[0]
        candidate_rows = model_grid[model_grid["candidate"] == "bagged"].copy()
        candidate_rows["mse_improves"] = candidate_rows["mse"] < baseline["mse"]
        candidate_rows["mae_improves"] = candidate_rows["mae"] < baseline["mae"]
        candidate_rows["magnitude_mae_improves"] = (
            candidate_rows["magnitude_mae"] < baseline["magnitude_mae"]
        )
        candidate_rows["correlation_improves"] = (
            candidate_rows["correlation"] > baseline["correlation"]
        )
        candidate_rows["direction_not_worse"] = (
            candidate_rows["direction_accuracy"]
            >= baseline["direction_accuracy"]
        )
        candidate_rows["passes_all_rules"] = (
            candidate_rows["mse_improves"]
            & candidate_rows["mae_improves"]
            & candidate_rows["magnitude_mae_improves"]
            & candidate_rows["correlation_improves"]
            & candidate_rows["direction_not_worse"]
        )
        for column in [
            "mse_improves", "mae_improves", "magnitude_mae_improves",
            "correlation_improves", "direction_not_worse", "passes_all_rules",
        ]:
            grid.loc[candidate_rows.index, column] = candidate_rows[column]

        passing = candidate_rows[candidate_rows["passes_all_rules"]]
        if passing.empty:
            winner = baseline
            bagging_selected = False
        else:
            winner = passing.sort_values(
                ["mse", "mae", "n_estimators", "block_size"]
            ).iloc[0]
            bagging_selected = True
        decisions.append(
            {
                "model": model_name,
                "bagging_selected": bagging_selected,
                "block_size": int(winner["block_size"]),
                "n_estimators": int(winner["n_estimators"]),
                "validation_mse": winner["mse"],
                "validation_mae": winner["mae"],
                "validation_magnitude_mae": winner["magnitude_mae"],
                "validation_magnitude_correlation": winner[
                    "magnitude_correlation"
                ],
                "validation_amplitude_ratio": winner["amplitude_ratio"],
                "validation_correlation": winner["correlation"],
                "validation_direction_accuracy": winner["direction_accuracy"],
                "mse_change_pct": (winner["mse"] / baseline["mse"] - 1) * 100,
                "mae_change_pct": (winner["mae"] / baseline["mae"] - 1) * 100,
                "magnitude_mae_change_pct": (
                    winner["magnitude_mae"] / baseline["magnitude_mae"] - 1
                )
                * 100,
                "direction_change_pp": (
                    winner["direction_accuracy"]
                    - baseline["direction_accuracy"]
                )
                * 100,
            }
        )
        chosen_configurations[model_name] = {
            "bagging_selected": bagging_selected,
            "block_size": int(winner["block_size"]),
            "n_estimators": int(winner["n_estimators"]),
        }

    final_models: dict[str, object] = {}
    feature_map: dict[str, list[str]] = {}
    for model_name, configuration in chosen_configurations.items():
        features = features_by_model[model_name]
        template = templates[model_name]
        base = gm.clone(template)
        base.fit(pretest[features], pretest["target"])
        final_models[f"Unbagged{model_name}"] = base
        feature_map[f"Unbagged{model_name}"] = features
        if configuration["bagging_selected"]:
            bag = gm.BlockBaggedRegressor(
                base_model=template,
                n_estimators=configuration["n_estimators"],
                block_size=configuration["block_size"],
                random_state=RANDOM_STATE,
            )
            bag.fit(pretest[features], pretest["target"])
            final_models[f"Bagged{model_name}"] = bag
            feature_map[f"Bagged{model_name}"] = features

    predictions, evaluation, test_metrics, coefficients = gm.evaluate_models(
        final_models, test, horizon=1, model_features=feature_map
    )
    residuals = gm.residual_diagnostics(evaluation, final_models)

    grid.to_csv(OUTPUT / "all_model_bagging_validation_grid.csv", index=False)
    decision_table = pd.DataFrame(decisions)
    decision_table.to_csv(OUTPUT / "all_model_bagging_decisions.csv", index=False)
    predictions.to_csv(OUTPUT / "all_model_bagging_test_predictions.csv")
    test_metrics.to_csv(
        OUTPUT / "all_model_bagging_test_metrics.csv", index=False
    )
    coefficients.to_csv(OUTPUT / "all_model_bagging_coefficients.csv")
    residuals.to_csv(
        OUTPUT / "all_model_bagging_residual_diagnostics.csv", index=False
    )
    with (OUTPUT / "selected_bagging_by_model.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "selection_data": "validation_only",
                "selection_rule": (
                    "lower MSE, MAE, and magnitude MAE; higher correlation; "
                    "non-decreasing direction accuracy"
                ),
                "models": chosen_configurations,
                "test_warning": (
                    "Event-filtered test metrics were calculated after all "
                    "bagging choices were frozen."
                ),
            },
            file,
            indent=2,
        )

    print("\nValidation decisions:")
    print(decision_table.to_string(index=False))
    print("\nTest amount and direction metrics:")
    print(test_metrics.to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
