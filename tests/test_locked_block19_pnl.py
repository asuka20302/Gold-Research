import numpy as np
import pandas as pd

from locked_block19_pnl import (
    TradingRule,
    round_trip_cost_threshold_pp,
    simulate_buy_and_hold,
    simulate_long_flat,
)


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "entry_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "exit_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "entry_open_cny_per_gram": [100.0, 100.0],
            "exit_close_cny_per_gram": [110.0, 90.0],
            "predicted_log_return": [0.02, -0.02],
            "predicted_growth_pp": [2.0, -2.0],
            "actual_log_return": [np.log(1.1), np.log(0.9)],
            "trailing_volatility_20d_pp": [1.0, 1.0],
        }
    )


def test_long_flat_ledger_uses_next_open_and_skips_negative_signal() -> None:
    rule = TradingRule("test", threshold_pp=0.0, sizing="full")
    ledger, summary = simulate_long_flat(
        small_frame(),
        rule,
        initial_capital=1_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )

    assert ledger["position"].tolist() == [1, 0]
    assert np.isclose(ledger.iloc[0]["gold_grams"], 10.0)
    assert np.isclose(ledger.iloc[0]["net_pnl"], 100.0)
    assert np.isclose(summary["final_capital"], 1_100.0)
    assert summary["trades"] == 1


def test_costs_reduce_pnl_and_create_positive_break_even_threshold() -> None:
    rule = TradingRule("test", threshold_pp=0.0, sizing="full")
    no_cost_ledger, _ = simulate_long_flat(
        small_frame(), rule, 1_000.0, commission_bps=0.0, slippage_bps=0.0
    )
    cost_ledger, _ = simulate_long_flat(
        small_frame(), rule, 1_000.0, commission_bps=3.0, slippage_bps=2.0
    )

    assert cost_ledger["net_pnl"].sum() < no_cost_ledger["net_pnl"].sum()
    assert cost_ledger["total_trading_cost"].sum() > 0
    assert round_trip_cost_threshold_pp(3.0, 2.0) > 0


def test_buy_and_hold_uses_first_open_and_final_close() -> None:
    ledger, summary = simulate_buy_and_hold(
        small_frame(),
        initial_capital=1_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )

    assert np.isclose(ledger.iloc[-1]["equity"], 900.0)
    assert np.isclose(summary["total_return"], -0.1)
    assert summary["trades"] == 1
