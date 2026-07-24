"""Time-series out-of-fold weighted averaging for gold-return regressions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit

import gold_model_comparison as gm
from advanced_experiment_utils import (
    PROJECT,
    amount_metrics,
    experiment_frames,
    make_pipeline,
    selected_templates,
)


OUTPUT = PROJECT / "outputs" / "weighted_ensemble_h1"
N_SPLITS = 5
GAP = 1
RANDOM_STATE = 42

CURRENT_FIVE = ["OLS", "Ridge", "Lasso", "ElasticNet", "BaggedHuber"]
LOSS_ENHANCED_FIVE = [
    "OLS", "Ridge", "Lasso", "ElasticNet", "AbsolutePCA11"
]
ROBUST_PAIR = ["BaggedHuber", "AbsolutePCA11"]
ALL_SIX = [
    "OLS",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "BaggedHuber",
    "AbsolutePCA11",
]
ENSEMBLE_SETS = {
    "current_five": CURRENT_FIVE,
    "loss_enhanced_five": LOSS_ENHANCED_FIVE,
    "robust_pair": ROBUST_PAIR,
    "all_six_research": ALL_SIX,
}


def fit_candidate_predictions(
    name: str,
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    features: list[str],
    templates: dict[str, object],
    random_state: int,
) -> np.ndarray:
    if name == "BaggedHuber":
        model = gm.BlockBaggedRegressor(
            base_model=templates["Huber"],
            n_estimators=50,
            block_size=20,
            random_state=random_state,
        )
    elif name == "AbsolutePCA11":
        model = make_pipeline(
            QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"),
            n_components=11,
        )
    else:
        model = clone(templates[name])
    model.fit(fit_frame[features], fit_frame["target"])
    return np.asarray(model.predict(predict_frame[features]), dtype=float)


def normalized_inverse(values: np.ndarray) -> np.ndarray:
    inverse = 1.0 / np.maximum(np.asarray(values, dtype=float), 1e-15)
    return inverse / inverse.sum()


def optimized_weights(
    predictions: np.ndarray,
    actual: np.ndarray,
    ridge_strength: float = 0.0,
) -> np.ndarray:
    n_models = predictions.shape[1]
    initial = np.repeat(1.0 / n_models, n_models)
    actual_growth = np.expm1(actual) * 100.0
    zero_growth_mse = float(np.mean(actual_growth**2))

    def objective(weights: np.ndarray) -> float:
        predicted_log_return = predictions @ weights
        predicted_growth = np.expm1(predicted_log_return) * 100.0
        residual = actual_growth - predicted_growth
        mse = float(np.mean(residual**2))
        penalty = (
            ridge_strength
            * zero_growth_mse
            * float(np.sum((weights - initial) ** 2))
        )
        return mse + penalty

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def calculate_weight_methods(
    predictions: pd.DataFrame,
    actual: pd.Series,
    members: list[str],
) -> dict[str, np.ndarray]:
    matrix = predictions[members].to_numpy(float)
    actual_array = actual.to_numpy(float)
    actual_growth = np.expm1(actual_array) * 100.0
    predicted_growth = np.expm1(matrix) * 100.0
    individual_mse = np.mean(
        (predicted_growth - actual_growth[:, None]) ** 2, axis=0
    )
    individual_mae = np.mean(
        np.abs(predicted_growth - actual_growth[:, None]), axis=0
    )
    return {
        "equal": np.repeat(1.0 / len(members), len(members)),
        "inverse_growth_mse": normalized_inverse(individual_mse),
        "inverse_growth_mae": normalized_inverse(individual_mae),
        "optimized_growth_mse": optimized_weights(matrix, actual_array),
        "optimized_growth_mse_stable": optimized_weights(
            matrix, actual_array, ridge_strength=0.01
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, pretest, test = experiment_frames()
    templates, features_by_model = selected_templates()
    features = features_by_model["Huber"]
    if any(feature_list != features for feature_list in features_by_model.values()):
        raise RuntimeError("Weighted comparison requires identical feature columns.")

    candidate_names = ALL_SIX
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)
    oof = pd.DataFrame(index=train.index, columns=candidate_names, dtype=float)
    fold_id = pd.Series(index=train.index, dtype="Int64")
    for fold, (fit_index, score_index) in enumerate(splitter.split(train), start=1):
        fit_frame = train.iloc[fit_index]
        score_frame = train.iloc[score_index]
        print(
            f"OOF fold {fold}: fit={len(fit_frame)}, score={len(score_frame)}"
        )
        fold_id.loc[score_frame.index] = fold
        for name in candidate_names:
            oof.loc[score_frame.index, name] = fit_candidate_predictions(
                name,
                fit_frame,
                score_frame,
                features,
                templates,
                random_state=RANDOM_STATE + fold,
            )

    usable = oof.dropna().index
    oof_actual = train.loc[usable, "target"]
    oof = oof.loc[usable]
    weight_rows: list[dict[str, object]] = []
    weights_by_candidate: dict[str, tuple[list[str], np.ndarray]] = {}
    for set_name, members in ENSEMBLE_SETS.items():
        methods = calculate_weight_methods(oof, oof_actual, members)
        for method, weights in methods.items():
            ensemble_name = f"{set_name}__{method}"
            weights_by_candidate[ensemble_name] = (members, weights)
            for member, weight in zip(members, weights):
                weight_rows.append(
                    {
                        "ensemble": ensemble_name,
                        "set": set_name,
                        "method": method,
                        "model": member,
                        "weight": float(weight),
                    }
                )

    base_oof_rows = [
        {
            "candidate": name,
            **amount_metrics(
                oof_actual,
                oof[name],
                train.loc[usable, "entry_open"],
                train.loc[usable, "exit_close"],
            ),
        }
        for name in candidate_names
    ]

    # Validation is not used to calculate numerical weights; it selects a
    # weighting method after the OOF-derived weights are frozen.
    validation_predictions = pd.DataFrame(
        {"actual_return": validation["target"].to_numpy()},
        index=validation.index,
    )
    for name in candidate_names:
        validation_predictions[name] = fit_candidate_predictions(
            name,
            train,
            validation,
            features,
            templates,
            random_state=RANDOM_STATE,
        )

    validation_rows: list[dict[str, object]] = []
    for name in candidate_names:
        validation_rows.append(
            {
                "candidate": name,
                "candidate_type": "individual",
                "set": "individual",
                "method": "individual",
                **amount_metrics(
                    validation["target"],
                    validation_predictions[name],
                    validation["entry_open"],
                    validation["exit_close"],
                ),
            }
        )
    for ensemble_name, (members, weights) in weights_by_candidate.items():
        prediction = validation_predictions[members].to_numpy(float) @ weights
        validation_predictions[ensemble_name] = prediction
        set_name, method = ensemble_name.split("__", maxsplit=1)
        validation_rows.append(
            {
                "candidate": ensemble_name,
                "candidate_type": "ensemble",
                "set": set_name,
                "method": method,
                **amount_metrics(
                    validation["target"],
                    prediction,
                    validation["entry_open"],
                    validation["exit_close"],
                ),
            }
        )

    validation_metrics = pd.DataFrame(validation_rows)
    baseline = validation_metrics.loc[
        validation_metrics["candidate"] == "BaggedHuber"
    ].iloc[0]
    ensemble_mask = validation_metrics["candidate_type"] == "ensemble"
    validation_metrics["growth_rmse_improves_vs_bagged_huber"] = (
        validation_metrics["growth_rmse_pp"] < baseline["growth_rmse_pp"]
    )
    validation_metrics["growth_mae_improves_vs_bagged_huber"] = (
        validation_metrics["growth_mae_pp"] < baseline["growth_mae_pp"]
    )
    validation_metrics["growth_magnitude_mae_improves_vs_bagged_huber"] = (
        validation_metrics["growth_magnitude_mae_pp"]
        < baseline["growth_magnitude_mae_pp"]
    )
    validation_metrics["price_change_mae_improves_vs_bagged_huber"] = (
        validation_metrics["price_change_mae_cny_per_gram"]
        < baseline["price_change_mae_cny_per_gram"]
    )
    validation_metrics["passes_rule"] = (
        ensemble_mask
        & validation_metrics["growth_rmse_improves_vs_bagged_huber"]
        & validation_metrics["growth_mae_improves_vs_bagged_huber"]
        & validation_metrics[
            "growth_magnitude_mae_improves_vs_bagged_huber"
        ]
        & validation_metrics["price_change_mae_improves_vs_bagged_huber"]
    )
    passing = validation_metrics[validation_metrics["passes_rule"]]
    if passing.empty:
        selected_name = "BaggedHuber"
        ensemble_selected = False
    else:
        selected_name = str(
            passing.sort_values(
                ["growth_rmse_pp", "growth_mae_pp", "candidate"]
            ).iloc[0]["candidate"]
        )
        ensemble_selected = True

    # Refit all six research candidates on train + validation and score the
    # diagnostic test using the already-frozen OOF weights.
    test_predictions = pd.DataFrame(
        {"actual_return": test["target"].to_numpy()}, index=test.index
    )
    for name in candidate_names:
        test_predictions[name] = fit_candidate_predictions(
            name,
            pretest,
            test,
            features,
            templates,
            random_state=RANDOM_STATE,
        )
    test_rows: list[dict[str, object]] = []
    for name in candidate_names:
        test_rows.append(
            {
                "candidate": name,
                "candidate_type": "individual",
                **amount_metrics(
                    test["target"],
                    test_predictions[name],
                    test["entry_open"],
                    test["exit_close"],
                ),
            }
        )
    for ensemble_name, (members, weights) in weights_by_candidate.items():
        prediction = test_predictions[members].to_numpy(float) @ weights
        test_predictions[ensemble_name] = prediction
        test_rows.append(
            {
                "candidate": ensemble_name,
                "candidate_type": "ensemble",
                **amount_metrics(
                    test["target"],
                    prediction,
                    test["entry_open"],
                    test["exit_close"],
                ),
            }
        )

    oof_export = oof.copy()
    oof_export.insert(0, "target", oof_actual)
    oof_export.insert(0, "fold", fold_id.loc[usable].astype(int))
    oof_export.to_csv(OUTPUT / "weighted_oof_predictions.csv")
    pd.DataFrame(base_oof_rows).to_csv(
        OUTPUT / "weighted_base_oof_metrics.csv", index=False
    )
    pd.DataFrame(weight_rows).to_csv(
        OUTPUT / "weighted_ensemble_weights.csv", index=False
    )
    validation_predictions.to_csv(OUTPUT / "weighted_validation_predictions.csv")
    validation_metrics.sort_values("growth_rmse_pp").to_csv(
        OUTPUT / "weighted_validation_metrics.csv", index=False
    )
    test_predictions.to_csv(OUTPUT / "weighted_test_predictions.csv")
    pd.DataFrame(test_rows).sort_values("growth_rmse_pp").to_csv(
        OUTPUT / "weighted_test_metrics.csv", index=False
    )
    with (OUTPUT / "selected_weighted_ensemble.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "weight_training_data": "five-fold expanding-window OOF on train",
                "weight_training_gap": GAP,
                "method_selection_data": "validation",
                "selected_candidate": selected_name,
                "ensemble_selected": ensemble_selected,
                "selection_rule": (
                    "lower growth RMSE, growth MAE, growth-magnitude MAE, "
                    "and CNY/gram price-change MAE than BaggedHuber"
                ),
                "selected_weights": (
                    {
                        member: float(weight)
                        for member, weight in zip(
                            *weights_by_candidate[selected_name]
                        )
                    }
                    if ensemble_selected
                    else {"BaggedHuber": 1.0}
                ),
                "test_warning": (
                    "Existing event-filtered test metrics are diagnostic rather "
                    "than pristine out-of-sample evidence."
                ),
            },
            file,
            indent=2,
        )

    print("Top validation candidates:")
    print(
        validation_metrics.sort_values("growth_rmse_pp")
        .head(12)[
            [
                "candidate",
                "growth_rmse_pp",
                "growth_mae_pp",
                "growth_magnitude_mae_pp",
                "growth_bias_pp",
                "price_change_mae_cny_per_gram",
                "passes_rule",
            ]
        ]
        .to_string(index=False)
    )
    print(f"\nSelected: {selected_name}")
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
