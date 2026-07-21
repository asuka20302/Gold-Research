"""Validation-first PCA comparison for the five horizon-one regressions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from advanced_experiment_utils import (
    MODEL_CLASSES,
    PROJECT,
    amount_metrics,
    experiment_frames,
    make_pipeline,
    parse_parameter_text,
    selected_model_table,
)


OUTPUT = PROJECT / "outputs" / "pca_h1"
COMPONENT_GRID = [3, 5, 7, 9, 11, 13, 14]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, pretest, test = experiment_frames()
    selected = selected_model_table()

    rows: list[dict[str, object]] = []
    templates: dict[tuple[str, int], object] = {}
    features_by_model: dict[str, list[str]] = {}

    for _, stored in selected.iterrows():
        model_name = str(stored["model"])
        features = str(stored["active_features"]).split(",")
        features_by_model[model_name] = features
        parameters = parse_parameter_text(
            model_name, str(stored["selected_parameters"])
        )
        estimator_class = MODEL_CLASSES[model_name]

        candidates = [0] + [
            count for count in COMPONENT_GRID if count <= len(features)
        ]
        for component_count in candidates:
            template = make_pipeline(
                estimator_class(**parameters),
                n_components=(component_count or None),
            )
            model = clone(template)
            model.fit(train[features], train["target"])
            prediction = model.predict(validation[features])
            explained = 1.0
            if component_count:
                explained = float(
                    model.named_steps["pca"].explained_variance_ratio_.sum()
                )
            rows.append(
                {
                    "model": model_name,
                    "representation": "raw" if component_count == 0 else "pca",
                    "n_components": component_count,
                    "explained_variance_ratio": explained,
                    **amount_metrics(
                        validation["target"],
                        prediction,
                        validation["entry_open"],
                        validation["exit_close"],
                    ),
                }
            )
            templates[(model_name, component_count)] = template

    grid = pd.DataFrame(rows)
    decisions: list[dict[str, object]] = []
    chosen: dict[str, int] = {}
    for model_name, group in grid.groupby("model", sort=False):
        baseline = group[group["n_components"] == 0].iloc[0]
        pca = group[group["n_components"] > 0].copy()
        pca["growth_rmse_improves"] = (
            pca["growth_rmse_pp"] < baseline["growth_rmse_pp"]
        )
        pca["growth_mae_improves"] = (
            pca["growth_mae_pp"] < baseline["growth_mae_pp"]
        )
        pca["growth_magnitude_mae_improves"] = (
            pca["growth_magnitude_mae_pp"]
            < baseline["growth_magnitude_mae_pp"]
        )
        pca["price_change_mae_improves"] = (
            pca["price_change_mae_cny_per_gram"]
            < baseline["price_change_mae_cny_per_gram"]
        )
        pca["passes_rule"] = (
            pca["growth_rmse_improves"]
            & pca["growth_mae_improves"]
            & pca["growth_magnitude_mae_improves"]
            & pca["price_change_mae_improves"]
        )
        for column in [
            "growth_rmse_improves",
            "growth_mae_improves",
            "growth_magnitude_mae_improves",
            "price_change_mae_improves",
            "passes_rule",
        ]:
            grid.loc[pca.index, column] = pca[column]

        passing = pca[pca["passes_rule"]]
        if passing.empty:
            winner = baseline
            pca_selected = False
        else:
            winner = passing.sort_values(
                ["growth_rmse_pp", "growth_mae_pp", "n_components"]
            ).iloc[0]
            pca_selected = True
        component_count = int(winner["n_components"])
        chosen[model_name] = component_count
        decisions.append(
            {
                "model": model_name,
                "pca_selected": pca_selected,
                "n_components": component_count,
                "explained_variance_ratio": winner["explained_variance_ratio"],
                "validation_growth_rmse_pp": winner["growth_rmse_pp"],
                "validation_growth_mae_pp": winner["growth_mae_pp"],
                "validation_growth_magnitude_mae_pp": winner[
                    "growth_magnitude_mae_pp"
                ],
                "validation_growth_bias_pp": winner["growth_bias_pp"],
                "validation_price_change_mae_cny_per_gram": winner[
                    "price_change_mae_cny_per_gram"
                ],
                "growth_rmse_change_pct": (
                    winner["growth_rmse_pp"] / baseline["growth_rmse_pp"] - 1
                )
                * 100,
                "growth_mae_change_pct": (
                    winner["growth_mae_pp"] / baseline["growth_mae_pp"] - 1
                )
                * 100,
                "growth_magnitude_mae_change_pct": (
                    winner["growth_magnitude_mae_pp"]
                    / baseline["growth_magnitude_mae_pp"]
                    - 1
                )
                * 100,
            }
        )

    # Open the diagnostic test only after the per-model validation decisions.
    test_rows: list[dict[str, object]] = []
    prediction_table = pd.DataFrame(
        {"actual_return": test["target"].to_numpy()}, index=test.index
    )
    for model_name, component_count in chosen.items():
        features = features_by_model[model_name]
        template = templates[(model_name, component_count)]
        model = clone(template)
        model.fit(pretest[features], pretest["target"])
        prediction = model.predict(test[features])
        label = (
            f"PCA{component_count}_{model_name}"
            if component_count
            else f"Raw_{model_name}"
        )
        prediction_table[label] = prediction
        test_rows.append(
            {
                "candidate": label,
                "model": model_name,
                "n_components": component_count,
                **amount_metrics(
                    test["target"],
                    prediction,
                    test["entry_open"],
                    test["exit_close"],
                ),
            }
        )

    decision_table = pd.DataFrame(decisions)
    grid.to_csv(OUTPUT / "pca_validation_grid.csv", index=False)
    decision_table.to_csv(OUTPUT / "pca_validation_decisions.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUTPUT / "pca_test_metrics.csv", index=False)
    prediction_table.to_csv(OUTPUT / "pca_test_predictions.csv")
    with (OUTPUT / "selected_pca_by_model.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_data": "validation_only",
                "component_grid": COMPONENT_GRID,
                "selection_rule": (
                    "lower growth RMSE, growth MAE, growth-magnitude MAE, "
                    "and CNY/gram price-change MAE"
                ),
                "models": chosen,
                "test_warning": (
                    "The existing event-filtered test is diagnostic because "
                    "event windows were defined after earlier residual review."
                ),
            },
            file,
            indent=2,
        )

    print("PCA validation decisions:")
    print(decision_table.to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
