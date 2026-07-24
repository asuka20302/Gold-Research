"""Export selected out-of-sample gold forecasts in a QMT-friendly format.

The model/horizon are read from outputs/horizon_comparison/selected_horizon.json
so the QMT strategy uses the same fixed-horizon choice selected by the research
pipeline instead of an old hard-coded model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_DIR / "outputs" / "qmt_gold_signals.csv"
SELECTION_FILE = (
    PROJECT_DIR / "outputs" / "horizon_comparison" / "selected_horizon.json"
)


def find_gold_cache() -> Path:
    cache_files = sorted(
        (PROJECT_DIR / "data").glob("Au99_99_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not cache_files:
        raise FileNotFoundError(
            "No Au99.99 cache exists under the data directory."
        )
    return cache_files[0]


def main() -> None:
    with open(SELECTION_FILE, "r", encoding="utf-8") as file:
        selection = json.load(file)

    model_name = selection["model"]
    horizon = int(selection["horizon"])
    input_dir = PROJECT_DIR / "outputs" / "horizon_comparison" / f"h{horizon}"
    prediction_file = input_dir / "gold_non_overlapping_predictions.csv"
    price_file = input_dir / "gold_price_predictions.csv"

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Run horizon {horizon} first; missing {prediction_file}"
        )
    if not price_file.exists():
        raise FileNotFoundError(
            f"Run horizon {horizon} first; missing {price_file}"
        )

    predictions = pd.read_csv(prediction_file, dtype={"trade_date": str})
    if model_name not in predictions:
        raise KeyError(f"{model_name} is not present in {prediction_file}")

    prices = pd.read_csv(price_file, dtype={"trade_date": str})
    price_columns = ["trade_date", "actual_exit_price", model_name]
    missing_price_columns = [
        column for column in price_columns if column not in prices.columns
    ]
    if missing_price_columns:
        raise KeyError(
            f"Missing columns in {price_file}: {missing_price_columns}"
        )

    signals = pd.DataFrame(
        {
            "trade_date": predictions["trade_date"],
            "signal_date": predictions["trade_date"],
            "entry_date": predictions["entry_date"],
            "exit_date": predictions["exit_date"],
            "model": model_name,
            "horizon": horizon,
            "predicted_return": pd.to_numeric(
                predictions[model_name], errors="coerce"
            ),
            "actual_return": pd.to_numeric(
                predictions["actual_return"], errors="coerce"
            ),
            "entry_open": pd.to_numeric(
                predictions["entry_open"], errors="coerce"
            ),
            "exit_close": pd.to_numeric(
                predictions["exit_close"], errors="coerce"
            ),
        }
    ).dropna()

    price_lookup = prices[price_columns].rename(
        columns={
            "actual_exit_price": "actual_exit_price",
            model_name: "predicted_exit_price",
        }
    )
    signals = signals.merge(price_lookup, on="trade_date", how="left")
    if signals["predicted_exit_price"].isna().any():
        raise RuntimeError("Predicted exit prices are missing after merge.")

    signals["signal_date"] = signals["signal_date"].str.replace(
        "-", "", regex=False
    )
    signals["entry_date"] = signals["entry_date"].str.replace(
        "-", "", regex=False
    )
    signals["exit_date"] = signals["exit_date"].str.replace(
        "-", "", regex=False
    )
    signals["execution_date"] = signals["entry_date"]

    gold = pd.read_csv(
        find_gold_cache(),
        dtype={"trade_date": str},
        usecols=["trade_date", "close"],
    )
    gold["signal_date"] = gold["trade_date"].str.replace(
        "-", "", regex=False
    )
    gold["gold_close"] = pd.to_numeric(gold["close"], errors="coerce")
    gold = (
        gold[["signal_date", "gold_close"]]
        .dropna()
        .sort_values("signal_date")
        .drop_duplicates("signal_date", keep="last")
    )

    signals = signals.merge(
        gold[["signal_date", "gold_close"]],
        on="signal_date",
        how="left",
    )
    if signals["gold_close"].isna().any():
        missing_dates = signals.loc[
            signals["gold_close"].isna(), "signal_date"
        ].tolist()
        raise RuntimeError(
            "Gold closes are missing for QMT signal dates: "
            + ", ".join(missing_dates[:10])
        )

    # Leakage-free indicative target. It uses the signal-date close, which is
    # known when the model runs. The executable target can be recalculated at
    # the following open as entry_open * exp(predicted_return).
    signals["predicted_gold_price_from_signal_close"] = (
        signals["gold_close"] * np.exp(signals["predicted_return"])
    )
    signals["predicted_gold_price"] = signals["predicted_exit_price"]
    signals["predicted_change_cny_per_gram"] = (
        signals["predicted_gold_price"] - signals["gold_close"]
    )

    signals.to_csv(OUTPUT_FILE, index=False, encoding="gbk")
    print(
        f"Saved {len(signals)} QMT signals to {OUTPUT_FILE} "
        f"using model={model_name}, horizon={horizon}"
    )


if __name__ == "__main__":
    main()
