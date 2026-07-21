import numpy as np
import pandas as pd

import gold_model_comparison as gm


def synthetic_gold(rows: int = 280) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=rows)
    trend = np.linspace(300.0, 420.0, rows)
    wave = 2.0 * np.sin(np.arange(rows) / 7.0)
    close = trend + wave
    open_price = close * (1.0 + 0.001 * np.cos(np.arange(rows) / 5.0))
    return pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "close": close,
            "vol": 10_000 + np.arange(rows) * 5,
        }
    )


def test_horizon_one_target_is_next_open_to_next_close() -> None:
    model_data = gm.build_dataset(synthetic_gold(), horizon=1)

    assert (model_data["entry_date"] == model_data["exit_date"]).all()
    assert (model_data["entry_date"] > model_data.index).all()
    expected = np.log(model_data["exit_close"] / model_data["entry_open"])
    np.testing.assert_allclose(model_data["target"], expected)


def test_longer_horizon_keeps_entry_after_signal_and_exit_after_entry() -> None:
    model_data = gm.build_dataset(synthetic_gold(), horizon=3)

    assert (model_data["entry_date"] > model_data.index).all()
    assert (model_data["exit_date"] > model_data["entry_date"]).all()
    assert model_data.index.is_monotonic_increasing
    assert not model_data.index.has_duplicates


def test_maintained_regression_set_contains_exactly_five_models() -> None:
    assert set(gm.regression_candidates()) == {
        "OLS",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "Huber",
    }
