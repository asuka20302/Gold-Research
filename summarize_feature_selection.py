"""Create a compact summary of statistical feature selection by horizon."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "horizon_comparison"

rows: list[dict[str, object]] = []
for horizon in range(1, 8):
    folder = OUTPUT / f"h{horizon}"
    selections = pd.read_csv(folder / "selected_models.csv")
    best = selections.sort_values("validation_mse").iloc[0]
    rows.append(
        {
            "horizon": horizon,
            "feature_selection_method": best["feature_selection_method"],
            "selected_feature_count": int(best["active_feature_count"]),
            "selected_features": best["active_features"],
            "best_validation_model": best["model"],
            "validation_mse": best["validation_mse"],
            "relative_mse_zero": best["relative_mse_zero"],
            "validation_correlation": best["validation_correlation"],
            "validation_direction_accuracy": best[
                "validation_direction_accuracy"
            ],
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT / "feature_selection_summary.csv", index=False)
print(summary.to_string(index=False))
