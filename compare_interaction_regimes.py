"""Validation-first comparison of interaction and regime feature blocks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import gold_model_comparison as gm
from interaction_regime_features import (
    enforce_feature_hierarchy,
    feature_blocks,
    fit_regime_thresholds,
    market_state_counts,
    threshold_record,
    transform_interactions_and_regimes,
)
from statistical_feature_selection import build_feature_subsets
from tushare_factors import build_tushare_factor_features


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs" / "interaction_regime_h1"
START_DATE = "20150101"
END_DATE = "20260703"
HORIZON = 1


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = gm.download_gold_data(
        "Au99.99", START_DATE, END_DATE, PROJECT / "data", refresh=False
    )
    external = build_tushare_factor_features(
        START_DATE, END_DATE, PROJECT / "data", refresh=False
    )
    data = gm.build_dataset(raw, HORIZON, external)
    exclusions = gm.load_event_exclusions(PROJECT / "data/event_exclusions.csv")
    data, _, _ = gm.apply_event_exclusions(data, exclusions)
    train, validation, pretest, test = gm.chronological_split(data, HORIZON)
    baseline_pool, _ = gm.audit_model_features(train)

    thresholds = fit_regime_thresholds(train)
    transformed = {
        "train": transform_interactions_and_regimes(train, thresholds),
        "validation": transform_interactions_and_regimes(
            validation, thresholds
        ),
        "pretest": transform_interactions_and_regimes(pretest, thresholds),
        "test": transform_interactions_and_regimes(test, thresholds),
    }

    comparisons: list[pd.DataFrame] = []
    retained_rows: list[dict[str, object]] = []
    elimination_rows: list[pd.DataFrame] = []
    fitted_by_block: dict[str, dict[str, object]] = {}
    features_by_block: dict[str, list[str]] = {}

    for block, pool in feature_blocks(baseline_pool).items():
        subsets, _, audit = build_feature_subsets(
            transformed["train"], pool, alpha=0.05
        )
        selected_features = enforce_feature_hierarchy(subsets["aic"])
        models, selections = gm.select_regression_models(
            transformed["train"],
            transformed["validation"],
            transformed["pretest"],
            horizon=HORIZON,
            feature_columns=selected_features,
        )
        selections.insert(0, "candidate_block", block)
        selections["selected_feature_count"] = len(selected_features)
        selections["selected_features"] = ",".join(selected_features)
        comparisons.append(selections)
        fitted_by_block[block] = models
        features_by_block[block] = selected_features
        for order, feature in enumerate(selected_features, start=1):
            retained_rows.append(
                {
                    "candidate_block": block,
                    "feature_order": order,
                    "feature": feature,
                }
            )
        if not audit.empty:
            audit = audit.copy()
            audit.insert(0, "candidate_block", block)
            elimination_rows.append(audit)

    comparison = pd.concat(comparisons, ignore_index=True)
    best_by_block = (
        comparison.sort_values("validation_mse")
        .groupby("candidate_block", as_index=False)
        .first()
        .set_index("candidate_block")
    )
    baseline = best_by_block.loc["baseline"]
    decision_rows = []
    passing_blocks = []
    for block, row in best_by_block.iterrows():
        mse_improves = row["validation_mse"] < baseline["validation_mse"]
        correlation_improves = (
            row["validation_correlation"]
            > baseline["validation_correlation"]
        )
        direction_not_worse = (
            row["validation_direction_accuracy"]
            >= baseline["validation_direction_accuracy"]
        )
        passes = bool(
            block != "baseline"
            and mse_improves
            and correlation_improves
            and direction_not_worse
        )
        if passes:
            passing_blocks.append(block)
        decision_rows.append(
            {
                "candidate_block": block,
                "best_model": row["model"],
                "validation_mse": row["validation_mse"],
                "validation_correlation": row["validation_correlation"],
                "validation_direction_accuracy": row[
                    "validation_direction_accuracy"
                ],
                "mse_change_vs_baseline_pct": (
                    row["validation_mse"] / baseline["validation_mse"] - 1
                )
                * 100,
                "correlation_change_vs_baseline": (
                    row["validation_correlation"]
                    - baseline["validation_correlation"]
                ),
                "direction_change_vs_baseline_pp": (
                    row["validation_direction_accuracy"]
                    - baseline["validation_direction_accuracy"]
                )
                * 100,
                "passes_all_rules": passes,
            }
        )

    selected_block = (
        min(
            passing_blocks,
            key=lambda name: best_by_block.loc[name, "validation_mse"],
        )
        if passing_blocks
        else "baseline"
    )
    comparison["selected_block"] = (
        comparison["candidate_block"] == selected_block
    )

    selected_models = fitted_by_block[selected_block]
    selected_features = features_by_block[selected_block]
    feature_map = {name: selected_features for name in selected_models}
    predictions, evaluation, test_metrics, coefficients = gm.evaluate_models(
        selected_models,
        transformed["test"],
        horizon=HORIZON,
        model_features=feature_map,
    )
    diagnostics = gm.residual_diagnostics(evaluation, selected_models)

    comparison.to_csv(
        OUTPUT / "interaction_regime_validation_comparison.csv", index=False
    )
    pd.DataFrame(decision_rows).to_csv(
        OUTPUT / "interaction_regime_block_decision.csv", index=False
    )
    pd.DataFrame(retained_rows).to_csv(
        OUTPUT / "interaction_regime_retained_features.csv", index=False
    )
    if elimination_rows:
        pd.concat(elimination_rows, ignore_index=True).to_csv(
            OUTPUT / "interaction_regime_aic_audit.csv", index=False
        )
    market_state_counts(transformed).to_csv(
        OUTPUT / "market_state_counts.csv", index=False
    )
    predictions.to_csv(OUTPUT / "selected_block_predictions.csv")
    test_metrics.to_csv(OUTPUT / "selected_block_test_metrics.csv", index=False)
    coefficients.to_csv(OUTPUT / "selected_block_coefficients.csv")
    diagnostics.to_csv(
        OUTPUT / "selected_block_residual_diagnostics.csv", index=False
    )
    with (OUTPUT / "regime_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(threshold_record(thresholds), f, indent=2)
    with (OUTPUT / "selected_interaction_regime_block.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "selected_block": selected_block,
                "selected_features": selected_features,
                "selection_data": "validation_only",
                "rule": (
                    "lower MSE, higher correlation, and non-decreasing "
                    "direction accuracy versus the reduced AIC baseline"
                ),
                "test_warning": (
                    "Test metrics are event-filtered diagnostics and were "
                    "calculated only after the block was frozen."
                ),
            },
            f,
            indent=2,
        )

    print(pd.DataFrame(decision_rows).to_string(index=False))
    print(f"Selected block: {selected_block}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
