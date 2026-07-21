"""Compare squared, Huber, and absolute-error regression losses."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

from advanced_experiment_utils import (
    PROJECT,
    amount_metrics,
    experiment_frames,
    make_pipeline,
)


OUTPUT = PROJECT / "outputs" / "loss_function_h1"
HUBER_EPSILONS = [1.01, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00]
HUBER_ALPHAS = [0.00001, 0.0001, 0.001, 0.01]
ABSOLUTE_ALPHAS = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01]


def candidate_templates(pca_components: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    representations = [("raw", None)]
    if pca_components > 0:
        representations.append((f"pca_{pca_components}", pca_components))

    for representation, components in representations:
        candidates.append(
            {
                "loss_family": "squared_error",
                "candidate": f"Squared_{representation}",
                "representation": representation,
                "n_components": components or 0,
                "epsilon": np.nan,
                "alpha": 0.0,
                "template": make_pipeline(
                    LinearRegression(), n_components=components
                ),
            }
        )
        for epsilon in HUBER_EPSILONS:
            for alpha in HUBER_ALPHAS:
                candidates.append(
                    {
                        "loss_family": "huber",
                        "candidate": (
                            f"Huber_e{epsilon:g}_a{alpha:g}_{representation}"
                        ),
                        "representation": representation,
                        "n_components": components or 0,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "template": make_pipeline(
                            HuberRegressor(
                                epsilon=epsilon,
                                alpha=alpha,
                                max_iter=5000,
                                tol=1e-5,
                            ),
                            n_components=components,
                        ),
                    }
                )
        for alpha in ABSOLUTE_ALPHAS:
            candidates.append(
                {
                    "loss_family": "absolute_error",
                    "candidate": f"Absolute_a{alpha:g}_{representation}",
                    "representation": representation,
                    "n_components": components or 0,
                    "epsilon": np.nan,
                    "alpha": alpha,
                    "template": make_pipeline(
                        QuantileRegressor(
                            quantile=0.5,
                            alpha=alpha,
                            solver="highs",
                        ),
                        n_components=components,
                    ),
                }
            )
    return candidates


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, pretest, test = experiment_frames()
    pca_decisions = pd.read_csv(
        PROJECT / "outputs" / "pca_h1" / "pca_validation_decisions.csv"
    )
    huber_pca = pca_decisions.loc[pca_decisions["model"] == "Huber"].iloc[0]
    pca_components = (
        int(huber_pca["n_components"])
        if bool(huber_pca["pca_selected"])
        else 0
    )
    features = str(
        pd.read_csv(
            PROJECT
            / "outputs"
            / "horizon_comparison"
            / "h1"
            / "selected_models.csv"
        ).iloc[0]["active_features"]
    ).split(",")

    candidates = candidate_templates(pca_components)
    validation_rows: list[dict[str, object]] = []
    templates: dict[str, object] = {}
    for item in candidates:
        model = clone(item["template"])
        model.fit(train[features], train["target"])
        prediction = model.predict(validation[features])
        validation_rows.append(
            {
                key: value
                for key, value in item.items()
                if key != "template"
            }
            | amount_metrics(
                validation["target"],
                prediction,
                validation["entry_open"],
                validation["exit_close"],
            )
        )
        templates[str(item["candidate"])] = item["template"]

    grid = pd.DataFrame(validation_rows)
    baseline = grid.loc[
        (grid["loss_family"] == "huber")
        & (grid["representation"] == "raw")
        & np.isclose(grid["epsilon"], 1.10)
        & np.isclose(grid["alpha"], 0.001)
    ].iloc[0]
    grid["growth_rmse_improves_vs_huber_baseline"] = (
        grid["growth_rmse_pp"] < baseline["growth_rmse_pp"]
    )
    grid["growth_mae_improves_vs_huber_baseline"] = (
        grid["growth_mae_pp"] < baseline["growth_mae_pp"]
    )
    grid["growth_magnitude_mae_improves_vs_huber_baseline"] = (
        grid["growth_magnitude_mae_pp"]
        < baseline["growth_magnitude_mae_pp"]
    )
    grid["price_change_mae_improves_vs_huber_baseline"] = (
        grid["price_change_mae_cny_per_gram"]
        < baseline["price_change_mae_cny_per_gram"]
    )
    grid["passes_rule"] = (
        grid["growth_rmse_improves_vs_huber_baseline"]
        & grid["growth_mae_improves_vs_huber_baseline"]
        & grid["growth_magnitude_mae_improves_vs_huber_baseline"]
        & grid["price_change_mae_improves_vs_huber_baseline"]
    )

    family_winners = (
        grid.sort_values(["growth_rmse_pp", "growth_mae_pp", "candidate"])
        .groupby("loss_family", sort=False)
        .head(1)
        .sort_values("mse")
    )
    passing = grid[grid["passes_rule"]]
    if passing.empty:
        selected = baseline
        challenger_selected = False
    else:
        selected = passing.sort_values(
            ["growth_rmse_pp", "growth_mae_pp", "candidate"]
        ).iloc[0]
        challenger_selected = str(selected["candidate"]) != str(
            baseline["candidate"]
        )

    # Refit only frozen validation winners before diagnostic test scoring.
    test_candidates = list(family_winners["candidate"])
    baseline_name = str(baseline["candidate"])
    selected_name = str(selected["candidate"])
    test_candidates.extend([baseline_name, selected_name])
    test_candidates = list(dict.fromkeys(test_candidates))
    test_rows: list[dict[str, object]] = []
    predictions = pd.DataFrame(
        {"actual_return": test["target"].to_numpy()}, index=test.index
    )
    for name in test_candidates:
        model = clone(templates[name])
        model.fit(pretest[features], pretest["target"])
        prediction = model.predict(test[features])
        predictions[name] = prediction
        metadata = grid.loc[grid["candidate"] == name].iloc[0]
        test_rows.append(
            {
                "candidate": name,
                "loss_family": metadata["loss_family"],
                "representation": metadata["representation"],
                "n_components": metadata["n_components"],
                "epsilon": metadata["epsilon"],
                "alpha": metadata["alpha"],
                **amount_metrics(
                    test["target"],
                    prediction,
                    test["entry_open"],
                    test["exit_close"],
                ),
            }
        )

    grid.to_csv(OUTPUT / "loss_validation_grid.csv", index=False)
    family_winners.to_csv(
        OUTPUT / "loss_family_validation_winners.csv", index=False
    )
    pd.DataFrame(test_rows).sort_values("mse").to_csv(
        OUTPUT / "loss_test_metrics.csv", index=False
    )
    predictions.to_csv(OUTPUT / "loss_test_predictions.csv")
    selection_record = {
        "selection_data": "validation_only",
        "baseline": baseline_name,
        "selected_candidate": selected_name,
        "challenger_selected": challenger_selected,
        "selected_validation_metrics": {
            key: float(selected[key])
            for key in [
                "growth_rmse_pp",
                "growth_mae_pp",
                "growth_magnitude_mae_pp",
                "growth_bias_pp",
                "price_change_mae_cny_per_gram",
            ]
        },
        "rule": (
            "growth RMSE, growth MAE, growth-magnitude MAE, and CNY/gram "
            "price-change MAE below raw Huber"
        ),
        "test_warning": (
            "Existing event-filtered test metrics are diagnostic rather than "
            "pristine out-of-sample evidence."
        ),
    }
    with (OUTPUT / "selected_loss.json").open("w", encoding="utf-8") as file:
        json.dump(selection_record, file, indent=2)

    print("Loss-family validation winners:")
    print(
        family_winners[
            [
                "candidate",
                "growth_rmse_pp",
                "growth_mae_pp",
                "growth_magnitude_mae_pp",
                "growth_bias_pp",
                "price_change_mae_cny_per_gram",
                "passes_rule",
            ]
        ].to_string(index=False)
    )
    print(f"\nSelected candidate: {selected_name}")
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
