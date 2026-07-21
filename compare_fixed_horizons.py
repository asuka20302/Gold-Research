"""Aggregate horizons 1-7 and select a model using validation data only."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "horizon_comparison"
HORIZONS = range(1, 8)
NEAR_BEST_TOLERANCE = 0.01


def load_horizon_outputs() -> dict[str, pd.DataFrame]:
    validation_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    logistic_frames: list[pd.DataFrame] = []
    strategy_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    date_checks: list[dict[str, object]] = []

    for horizon in HORIZONS:
        folder = OUTPUT_DIR / f"h{horizon}"
        required_files = [
            "selected_models.csv",
            "gold_model_comparison.csv",
            "gold_strategy_backtest.csv",
            "residual_diagnostics.csv",
            "gold_common_date_predictions.csv",
        ]
        missing = [
            filename
            for filename in required_files
            if not (folder / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Horizon {horizon} is missing: {', '.join(missing)}"
            )

        validation = pd.read_csv(folder / "selected_models.csv")
        validation["horizon"] = horizon
        regression_validation = validation[
            validation["relative_mse_zero"].notna()
        ].copy()
        validation_frames.append(regression_validation)

        test = pd.read_csv(folder / "gold_model_comparison.csv")
        test.insert(0, "horizon", horizon)
        test_frames.append(test)

        logistic_path = folder / "gold_logistic_comparison.csv"
        if logistic_path.exists():
            logistic_test = pd.read_csv(logistic_path)
            logistic_row = validation.loc[
                validation["model"] == "Logistic"
            ]
            validation_auc = np.nan
            selected_c = np.nan
            if not logistic_row.empty:
                parameters = str(
                    logistic_row.iloc[0]["selected_parameters"]
                )
                auc_match = re.search(
                    r"validation_roc_auc=([0-9.eE+-]+)", parameters
                )
                c_match = re.search(r"C=([0-9.eE+-]+)", parameters)
                if auc_match:
                    validation_auc = float(auc_match.group(1))
                if c_match:
                    selected_c = float(c_match.group(1))
            logistic_test.insert(0, "horizon", horizon)
            logistic_test["selected_c"] = selected_c
            logistic_test["validation_roc_auc"] = validation_auc
            logistic_frames.append(logistic_test)

        strategy = pd.read_csv(folder / "gold_strategy_backtest.csv")
        strategy.insert(0, "horizon", horizon)
        strategy_frames.append(strategy)

        residual = pd.read_csv(folder / "residual_diagnostics.csv")
        residual.insert(0, "horizon", horizon)
        residual_frames.append(residual)

        predictions = pd.read_csv(
            folder / "gold_common_date_predictions.csv",
            parse_dates=["trade_date", "entry_date", "exit_date"],
        )
        date_checks.append(
            {
                "horizon": horizon,
                "rows": len(predictions),
                "first_signal": predictions["trade_date"].min(),
                "last_signal": predictions["trade_date"].max(),
                "first_entry": predictions["entry_date"].min(),
                "last_exit": predictions["exit_date"].max(),
            }
        )

    return {
        "validation": pd.concat(validation_frames, ignore_index=True),
        "test": pd.concat(test_frames, ignore_index=True),
        "logistic": (
            pd.concat(logistic_frames, ignore_index=True)
            if logistic_frames
            else pd.DataFrame()
        ),
        "strategy": pd.concat(strategy_frames, ignore_index=True),
        "residual": pd.concat(residual_frames, ignore_index=True),
        "dates": pd.DataFrame(date_checks),
    }


def apply_eligibility_rule(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if "daily_data_trading_eligible" in validation.columns:
        trading_eligible = (
            validation["daily_data_trading_eligible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(True)
        )
    else:
        trading_eligible = pd.Series(True, index=validation.index)

    eligible = validation[
        trading_eligible
        & (validation["relative_mse_zero"] < 1.0)
        & (validation["validation_correlation"] > 0.0)
        & (validation["validation_direction_accuracy"] > 0.50)
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No model/horizon combination passes the eligibility rule."
        )

    best_relative_mse = eligible["relative_mse_zero"].min()
    eligible["within_near_best_tolerance"] = (
        eligible["relative_mse_zero"]
        <= best_relative_mse + NEAR_BEST_TOLERANCE
    )
    near_best = eligible[eligible["within_near_best_tolerance"]]

    # Prefer the shortest horizon among near-best candidates. If more than
    # one model remains at that horizon, choose the lowest relative MSE.
    selected = near_best.sort_values(
        ["horizon", "relative_mse_zero", "model"]
    ).iloc[0]
    return eligible.sort_values(
        ["within_near_best_tolerance", "relative_mse_zero"],
        ascending=[False, True],
    ), selected


def save_chart(
    validation: pd.DataFrame,
    selected: pd.Series,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    for model_name, frame in validation.groupby("model"):
        ordered = frame.sort_values("horizon")
        axes[0].plot(
            ordered["horizon"],
            ordered["relative_mse_zero"],
            marker="o",
            label=model_name,
        )
        axes[1].plot(
            ordered["horizon"],
            ordered["validation_correlation"],
            marker="o",
            label=model_name,
        )
        axes[2].plot(
            ordered["horizon"],
            ordered["validation_direction_accuracy"],
            marker="o",
            label=model_name,
        )

    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].scatter(
        [selected["horizon"]],
        [selected["relative_mse_zero"]],
        color="red",
        marker="*",
        s=180,
        zorder=5,
        label="Selected",
    )
    axes[0].set_ylabel("Relative MSE vs zero")
    axes[0].set_title(
        "Fixed-horizon validation comparison (selection uses validation only)"
    )

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Validation correlation")

    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Direction accuracy")
    axes[2].set_xlabel("Forecast horizon (trading sessions)")
    axes[2].set_xticks(list(HORIZONS))

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.16, 0.5),
    )
    figure.tight_layout(rect=[0, 0, 0.84, 1])
    figure.savefig(
        OUTPUT_DIR / "horizon_validation_comparison.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = load_horizon_outputs()
    validation = outputs["validation"]
    test = outputs["test"]

    eligible, selected = apply_eligibility_rule(validation)
    selected_horizon = int(selected["horizon"])
    selected_model = str(selected["model"])

    selected_test_rows = test[
        (test["horizon"] == selected_horizon)
        & (test["model"] == selected_model)
    ]
    if len(selected_test_rows) != 1:
        raise RuntimeError(
            "Expected exactly one test row for the frozen selection."
        )
    selected_test = selected_test_rows.iloc[0]

    merged = validation.merge(
        test,
        on=["horizon", "model"],
        how="left",
        suffixes=("_validation", "_test"),
        validate="one_to_one",
    )

    validation.to_csv(
        OUTPUT_DIR / "horizon_validation_metrics.csv",
        index=False,
    )
    test.to_csv(
        OUTPUT_DIR / "horizon_test_metrics.csv",
        index=False,
    )
    merged.to_csv(
        OUTPUT_DIR / "horizon_model_metrics.csv",
        index=False,
    )
    eligible.to_csv(
        OUTPUT_DIR / "eligible_horizon_models.csv",
        index=False,
    )
    outputs["logistic"].to_csv(
        OUTPUT_DIR / "horizon_logistic_metrics.csv",
        index=False,
    )
    outputs["strategy"].to_csv(
        OUTPUT_DIR / "horizon_strategy_metrics.csv",
        index=False,
    )
    outputs["residual"].to_csv(
        OUTPUT_DIR / "horizon_residual_diagnostics.csv",
        index=False,
    )
    outputs["dates"].to_csv(
        OUTPUT_DIR / "horizon_date_checks.csv",
        index=False,
    )

    selection_record = {
        "model": selected_model,
        "horizon": selected_horizon,
        "entry": "next_session_open",
        "exit": (
            "same_session_close"
            if selected_horizon == 1
            else f"session_{selected_horizon}_close"
        ),
        "selection_data": "validation_only",
        "selection_metric": "relative_mse_zero",
        "eligibility_rule": {
            "daily_data_trading_eligible": "True",
            "relative_mse_zero": "< 1.0",
            "validation_correlation": "> 0.0",
            "validation_direction_accuracy": "> 0.50",
        },
        "near_best_tolerance": NEAR_BEST_TOLERANCE,
        "tie_break": (
            "shortest horizon, then lowest relative MSE, then model name"
        ),
        "validation_metrics": {
            "mse": make_json_safe(selected["validation_mse"]),
            "zero_forecast_mse": make_json_safe(
                selected["zero_forecast_mse"]
            ),
            "relative_mse_zero": make_json_safe(
                selected["relative_mse_zero"]
            ),
            "correlation": make_json_safe(
                selected["validation_correlation"]
            ),
            "direction_accuracy": make_json_safe(
                selected["validation_direction_accuracy"]
            ),
            "mae": make_json_safe(selected.get("validation_mae")),
            "magnitude_mae": make_json_safe(
                selected.get("validation_magnitude_mae")
            ),
            "magnitude_correlation": make_json_safe(
                selected.get("validation_magnitude_correlation")
            ),
            "amplitude_ratio": make_json_safe(
                selected.get("validation_amplitude_ratio")
            ),
            "selected_parameters": str(
                selected["selected_parameters"]
            ),
        },
        "test_metrics_after_selection": {
            key: make_json_safe(selected_test[key])
            for key in [
                "evaluation_rows",
                "mse",
                "mae",
                "return_bias",
                "magnitude_mae",
                "magnitude_correlation",
                "amplitude_ratio",
                "r2",
                "correlation",
                "direction_accuracy",
                "prediction_std",
                "price_mae_cny_per_gram",
            ]
        },
        "test_evaluation_filter": {
            "event_exclusions_applied": make_json_safe(
                selected_test.get("event_exclusions_applied", False)
            ),
            "event_excluded_rows": make_json_safe(
                selected_test.get("event_excluded_rows", 0)
            ),
            "event_exclusion_file": str(
                selected_test.get("event_exclusion_file", "")
            ),
            "methodological_warning": (
                "These event windows were defined after inspecting abnormal "
                "test-period residuals. Filtered test metrics are diagnostic "
                "and must not be treated as untouched out-of-sample evidence."
            ),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (OUTPUT_DIR / "selected_horizon.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selection_record, file, indent=2, ensure_ascii=False)

    save_chart(validation, selected)

    print("Selected fixed-horizon specification:")
    print(
        f"  model={selected_model}, horizon={selected_horizon}, "
        f"relative_mse_zero={selected['relative_mse_zero']:.6f}, "
        f"validation_correlation="
        f"{selected['validation_correlation']:.6f}, "
        f"validation_direction_accuracy="
        f"{selected['validation_direction_accuracy']:.2%}"
    )
    print(
        "Test metrics were attached only after the validation selection "
        "was frozen."
    )
    print(f"Outputs saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
