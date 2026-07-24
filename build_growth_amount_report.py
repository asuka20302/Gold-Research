"""Create human-readable predicted-versus-actual gold growth reports."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_experiment_utils import PROJECT, experiment_frames


INPUT = PROJECT / "outputs" / "weighted_ensemble_h1"
OUTPUT = PROJECT / "outputs" / "growth_amount_report_h1"
IMPORTANT_CANDIDATES = ["BaggedHuber", "AbsolutePCA11"]


def amount_rows(
    sample: str,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    candidates: list[str],
) -> pd.DataFrame:
    common = frame.index.intersection(predictions.index)
    frame = frame.loc[common]
    predictions = predictions.loc[common]
    actual_growth = np.expm1(frame["target"].to_numpy(float)) * 100.0
    rows: list[pd.DataFrame] = []

    for candidate in candidates:
        predicted_return = predictions[candidate].to_numpy(float)
        predicted_growth = np.expm1(predicted_return) * 100.0
        entry_open = frame["entry_open"].to_numpy(float)
        actual_exit = frame["exit_close"].to_numpy(float)
        predicted_exit = entry_open * np.exp(predicted_return)
        growth_error = predicted_growth - actual_growth
        magnitude_error = np.abs(predicted_growth) - np.abs(actual_growth)
        price_change_error = predicted_exit - actual_exit
        assessment = np.where(
            magnitude_error > 1e-12,
            "overestimated move size",
            np.where(
                magnitude_error < -1e-12,
                "underestimated move size",
                "matched move size",
            ),
        )
        rows.append(
            pd.DataFrame(
                {
                    "sample": sample,
                    "trade_date": common,
                    "entry_date": frame["entry_date"].to_numpy(),
                    "exit_date": frame["exit_date"].to_numpy(),
                    "candidate": candidate,
                    "entry_open_cny_per_gram": entry_open,
                    "actual_exit_close_cny_per_gram": actual_exit,
                    "predicted_exit_close_cny_per_gram": predicted_exit,
                    "actual_growth_pct": actual_growth,
                    "predicted_growth_pct": predicted_growth,
                    "growth_error_percentage_points": growth_error,
                    "absolute_growth_error_percentage_points": np.abs(
                        growth_error
                    ),
                    "actual_move_size_pct": np.abs(actual_growth),
                    "predicted_move_size_pct": np.abs(predicted_growth),
                    "move_size_error_percentage_points": magnitude_error,
                    "actual_change_cny_per_gram": actual_exit - entry_open,
                    "predicted_change_cny_per_gram": predicted_exit - entry_open,
                    "change_error_cny_per_gram": price_change_error,
                    "absolute_change_error_cny_per_gram": np.abs(
                        price_change_error
                    ),
                    "amount_assessment": assessment,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for (sample, candidate), group in rows.groupby(
        ["sample", "candidate"], sort=False
    ):
        growth_error = group["growth_error_percentage_points"].to_numpy(float)
        magnitude_error = group[
            "move_size_error_percentage_points"
        ].to_numpy(float)
        price_error = group["change_error_cny_per_gram"].to_numpy(float)
        summaries.append(
            {
                "sample": sample,
                "candidate": candidate,
                "observations": len(group),
                "mean_actual_growth_pct": group["actual_growth_pct"].mean(),
                "mean_predicted_growth_pct": group[
                    "predicted_growth_pct"
                ].mean(),
                "growth_bias_pp": growth_error.mean(),
                "growth_mae_pp": np.abs(growth_error).mean(),
                "growth_rmse_pp": np.sqrt(np.mean(growth_error**2)),
                "growth_median_absolute_error_pp": np.median(
                    np.abs(growth_error)
                ),
                "growth_p90_absolute_error_pp": np.quantile(
                    np.abs(growth_error), 0.90
                ),
                "move_size_mae_pp": np.abs(magnitude_error).mean(),
                "move_size_bias_pp": magnitude_error.mean(),
                "price_change_bias_cny_per_gram": price_error.mean(),
                "price_change_mae_cny_per_gram": np.abs(price_error).mean(),
                "price_change_rmse_cny_per_gram": np.sqrt(
                    np.mean(price_error**2)
                ),
                "move_size_underestimated_share": float(
                    np.mean(magnitude_error < 0)
                ),
                "move_size_overestimated_share": float(
                    np.mean(magnitude_error > 0)
                ),
            }
        )
    return pd.DataFrame(summaries)


def save_selected_chart(selected_rows: pd.DataFrame, selected_name: str) -> None:
    test = selected_rows[selected_rows["sample"] == "test"].copy()
    display = test.tail(150)
    dates = pd.to_datetime(display["trade_date"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(
        dates,
        display["actual_growth_pct"],
        label="Actual growth/drop (%)",
        linewidth=1.6,
    )
    axes[0].plot(
        dates,
        display["predicted_growth_pct"],
        label="Predicted growth/drop (%)",
        linewidth=1.4,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Growth/drop (%)")
    axes[0].set_title(
        f"{selected_name}: predicted and actual gold growth/drop "
        "(last 150 diagnostic test rows)"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].bar(
        dates,
        display["growth_error_percentage_points"],
        width=2.0,
        color=np.where(
            display["growth_error_percentage_points"] >= 0,
            "#d95f02",
            "#1b9e77",
        ),
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Predicted − actual (pp)")
    axes[1].set_xlabel("Signal date")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "selected_growth_amount_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, validation, _, test = experiment_frames()
    with (INPUT / "selected_weighted_ensemble.json").open(
        encoding="utf-8"
    ) as file:
        selected_name = json.load(file)["selected_candidate"]
    candidates = [*IMPORTANT_CANDIDATES, selected_name]

    validation_predictions = pd.read_csv(
        INPUT / "weighted_validation_predictions.csv",
        index_col="trade_date",
        parse_dates=True,
    )
    test_predictions = pd.read_csv(
        INPUT / "weighted_test_predictions.csv",
        index_col="trade_date",
        parse_dates=True,
    )
    validation_rows = amount_rows(
        "validation", validation, validation_predictions, candidates
    )
    test_rows = amount_rows("test", test, test_predictions, candidates)
    all_rows = pd.concat([validation_rows, test_rows], ignore_index=True)
    summary = summarize(all_rows)

    all_rows.to_csv(OUTPUT / "all_candidate_growth_amount_errors.csv", index=False)
    selected_rows = all_rows[all_rows["candidate"] == selected_name].copy()
    selected_rows.to_csv(
        OUTPUT / "selected_ensemble_growth_amount_errors.csv", index=False
    )
    summary.to_csv(OUTPUT / "growth_amount_summary.csv", index=False)
    selected_rows.nlargest(
        25, "absolute_growth_error_percentage_points"
    ).to_csv(OUTPUT / "selected_largest_growth_errors.csv", index=False)
    save_selected_chart(selected_rows, selected_name)

    print("Amount-based summary:")
    print(summary.to_string(index=False))
    print(f"\nSelected candidate: {selected_name}")
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
