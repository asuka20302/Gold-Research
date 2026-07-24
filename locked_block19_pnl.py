"""Development-selected Bagged Huber trading rule and locked Block 19 P&L.

The forecast target is the Au99.99 next-session open-to-close log return.  A
signal is known on ``trade_date``, entry occurs at ``entry_date`` open, and the
position is closed at the same session's close.  This is a theoretical direct-
gold, long/flat ledger; it is not an ETF execution backtest.

All thresholds and sizing choices are selected from rolling out-of-sample
Bagged Huber predictions on Blocks 10-18.  Block 19 is used only after those
rules are frozen.  An earlier model diagnostic exposed Block 19 outcomes, so
this is a computationally locked common holdout rather than a pristine human-
blind test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from advanced_experiment_utils import PROJECT, experiment_frames
from huber_rolling_19block_cv import equal_chronological_blocks


PREDICTION_ROOT = PROJECT / "outputs" / "locked_block19_ensemble"
DEFAULT_OUTPUT = PROJECT / "outputs" / "locked_block19_pnl"
TRADING_DAYS = 252
HOLDOUT_BLOCK = 19
DEVELOPMENT_LAST_BLOCK = 18
MIN_DEVELOPMENT_TRADES = 50


@dataclass(frozen=True)
class TradingRule:
    name: str
    threshold_pp: float
    sizing: str
    risk_multiplier: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Bagged Huber long/flat rules on Blocks 10-18 and calculate "
            "a locked Block 19 direct-gold P&L ledger."
        )
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=3.0,
        help="Commission per side in basis points (default: 3).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=2.0,
        help="Adverse execution slippage per side in basis points (default: 2).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_costs(commission_bps: float, slippage_bps: float) -> None:
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("Commission and slippage cannot be negative.")
    if commission_bps + slippage_bps >= 5_000:
        raise ValueError("Per-side costs must be below 50%.")


def round_trip_cost_threshold_pp(
    commission_bps: float,
    slippage_bps: float,
) -> float:
    """Approximate break-even predicted move for a long round trip."""
    commission = commission_bps / 10_000.0
    slippage = slippage_bps / 10_000.0
    entry_multiplier = (1.0 + slippage) * (1.0 + commission)
    exit_multiplier = (1.0 - slippage) * (1.0 - commission)
    return float((entry_multiplier / exit_multiplier - 1.0) * 100.0)


def position_weights(frame: pd.DataFrame, rule: TradingRule) -> np.ndarray:
    prediction = frame["predicted_growth_pp"].to_numpy(float)
    active = prediction > rule.threshold_pp
    if rule.sizing == "full":
        return active.astype(float)
    if rule.sizing == "volatility":
        if rule.risk_multiplier is None or rule.risk_multiplier <= 0:
            raise ValueError("Volatility sizing requires a positive risk multiplier.")
        trailing_volatility = frame["trailing_volatility_20d_pp"].to_numpy(float)
        denominator = np.maximum(
            rule.risk_multiplier * trailing_volatility,
            1e-12,
        )
        sized = np.clip(prediction / denominator, 0.0, 1.0)
        return np.where(active, sized, 0.0)
    raise ValueError(f"Unknown sizing method: {rule.sizing}")


def simulate_long_flat(
    frame: pd.DataFrame,
    rule: TradingRule,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Simulate non-overlapping next-open/same-close long/flat trades."""
    validate_costs(commission_bps, slippage_bps)
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    required = {
        "signal_date",
        "entry_date",
        "exit_date",
        "entry_open_cny_per_gram",
        "exit_close_cny_per_gram",
        "predicted_log_return",
        "predicted_growth_pp",
        "actual_log_return",
        "trailing_volatility_20d_pp",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing P&L columns: {sorted(missing)}")

    ordered = frame.sort_values("signal_date").reset_index(drop=True).copy()
    signal_date = pd.to_datetime(ordered["signal_date"])
    entry_date = pd.to_datetime(ordered["entry_date"])
    exit_date = pd.to_datetime(ordered["exit_date"])
    if not ((signal_date < entry_date) & (entry_date == exit_date)).all():
        raise RuntimeError("Expected signal < next-open entry = same-day close exit.")
    if entry_date.duplicated().any():
        raise RuntimeError("The horizon-one ledger contains overlapping entry dates.")

    weights = position_weights(ordered, rule)
    commission_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    capital = float(initial_capital)
    rows: list[dict[str, object]] = []

    for index, source in ordered.iterrows():
        capital_before = capital
        target_weight = float(weights[index])
        entry_open = float(source["entry_open_cny_per_gram"])
        exit_close = float(source["exit_close_cny_per_gram"])
        allocated_capital = capital_before * target_weight
        if target_weight > 0:
            entry_fill = entry_open * (1.0 + slippage_rate)
            exit_fill = exit_close * (1.0 - slippage_rate)
            # The allocated budget includes the entry commission, preventing a
            # nominal 100% position from borrowing cash to pay its buy fee.
            gold_grams = allocated_capital / (
                entry_fill * (1.0 + commission_rate)
            )
            gross_pnl = gold_grams * (exit_close - entry_open)
            slippage_cost = gold_grams * (
                (entry_fill - entry_open) + (exit_close - exit_fill)
            )
            commission_cost = commission_rate * gold_grams * (
                entry_fill + exit_fill
            )
            net_pnl = gross_pnl - slippage_cost - commission_cost
        else:
            entry_fill = 0.0
            exit_fill = 0.0
            gold_grams = 0.0
            gross_pnl = 0.0
            slippage_cost = 0.0
            commission_cost = 0.0
            net_pnl = 0.0
        capital = capital_before + net_pnl
        if capital <= 0:
            raise RuntimeError("Strategy capital became non-positive.")
        rows.append(
            {
                "strategy": rule.name,
                "score_block": source.get("score_block", np.nan),
                "signal_date": source["signal_date"],
                "entry_date": source["entry_date"],
                "exit_date": source["exit_date"],
                "predicted_log_return": source["predicted_log_return"],
                "predicted_growth_pp": source["predicted_growth_pp"],
                "actual_log_return": source["actual_log_return"],
                "actual_growth_pp": np.expm1(source["actual_log_return"]) * 100.0,
                "trailing_volatility_20d_pp": source[
                    "trailing_volatility_20d_pp"
                ],
                "threshold_pp": rule.threshold_pp,
                "sizing": rule.sizing,
                "risk_multiplier": rule.risk_multiplier,
                "position": int(target_weight > 0),
                "target_weight": target_weight,
                "entry_open_cny_per_gram": entry_open,
                "exit_close_cny_per_gram": exit_close,
                "entry_fill_cny_per_gram": entry_fill,
                "exit_fill_cny_per_gram": exit_fill,
                "gold_grams": gold_grams,
                "allocated_capital": allocated_capital,
                "gross_pnl": gross_pnl,
                "slippage_cost": slippage_cost,
                "commission_cost": commission_cost,
                "total_trading_cost": slippage_cost + commission_cost,
                "net_pnl": net_pnl,
                "capital_before": capital_before,
                "capital_after": capital,
                "net_daily_return": net_pnl / capital_before,
            }
        )

    ledger = pd.DataFrame(rows)
    prior_peak = np.maximum.accumulate(
        np.concatenate([[initial_capital], ledger["capital_after"].to_numpy(float)])
    )[1:]
    ledger["equity"] = ledger["capital_after"]
    ledger["running_peak"] = prior_peak
    ledger["drawdown"] = ledger["equity"] / ledger["running_peak"] - 1.0

    daily_returns = ledger["net_daily_return"].to_numpy(float)
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    total_return = float(capital / initial_capital - 1.0)
    annualized_return = float(
        (capital / initial_capital) ** (TRADING_DAYS / len(ledger)) - 1.0
    )
    annualized_volatility = float(daily_std * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(np.mean(daily_returns) / daily_std * np.sqrt(TRADING_DAYS))
        if daily_std > 0
        else 0.0
    )
    active = ledger["position"] == 1
    summary: dict[str, float | int | str] = {
        "strategy": rule.name,
        "threshold_pp": rule.threshold_pp,
        "sizing": rule.sizing,
        "risk_multiplier": (
            np.nan if rule.risk_multiplier is None else rule.risk_multiplier
        ),
        "observations": len(ledger),
        "trades": int(active.sum()),
        "average_target_weight": float(ledger["target_weight"].mean()),
        "turnover_one_way_weight": float(ledger["target_weight"].sum()),
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_net_pnl": float(ledger["net_pnl"].sum()),
        "total_gross_pnl": float(ledger["gross_pnl"].sum()),
        "total_trading_cost": float(ledger["total_trading_cost"].sum()),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_net_pnl_per_trade": (
            float(ledger.loc[active, "net_pnl"].mean()) if active.any() else 0.0
        ),
    }
    return ledger, summary


def simulate_buy_and_hold(
    frame: pd.DataFrame,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Buy at the first holdout open and sell at the final holdout close."""
    validate_costs(commission_bps, slippage_bps)
    ordered = frame.sort_values("signal_date").reset_index(drop=True).copy()
    commission_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    first_open = float(ordered.iloc[0]["entry_open_cny_per_gram"])
    final_close = float(ordered.iloc[-1]["exit_close_cny_per_gram"])
    entry_fill = first_open * (1.0 + slippage_rate)
    exit_fill = final_close * (1.0 - slippage_rate)
    gold_grams = initial_capital / (entry_fill * (1.0 + commission_rate))
    buy_commission = gold_grams * entry_fill * commission_rate
    cash = initial_capital - gold_grams * entry_fill - buy_commission
    sell_commission = gold_grams * exit_fill * commission_rate

    equity = cash + gold_grams * ordered["exit_close_cny_per_gram"].to_numpy(float)
    equity[-1] = cash + gold_grams * exit_fill - sell_commission
    daily_returns = np.diff(np.concatenate([[initial_capital], equity])) / np.concatenate(
        [[initial_capital], equity[:-1]]
    )
    running_peak = np.maximum.accumulate(
        np.concatenate([[initial_capital], equity])
    )[1:]
    ledger = pd.DataFrame(
        {
            "strategy": "BuyAndHold",
            "signal_date": ordered["signal_date"],
            "entry_date": ordered["entry_date"],
            "exit_date": ordered["exit_date"],
            "position": 1,
            "target_weight": 1.0,
            "gold_grams": gold_grams,
            "equity": equity,
            "capital_after": equity,
            "net_daily_return": daily_returns,
            "running_peak": running_peak,
            "drawdown": equity / running_peak - 1.0,
        }
    )
    gross_pnl = gold_grams * (final_close - first_open)
    slippage_cost = gold_grams * (
        (entry_fill - first_open) + (final_close - exit_fill)
    )
    commission_cost = buy_commission + sell_commission
    total_return = float(equity[-1] / initial_capital - 1.0)
    daily_std = float(np.std(daily_returns, ddof=1))
    summary: dict[str, float | int | str] = {
        "strategy": "BuyAndHold",
        "threshold_pp": np.nan,
        "sizing": "continuous holding",
        "risk_multiplier": np.nan,
        "observations": len(ordered),
        "trades": 1,
        "average_target_weight": 1.0,
        "turnover_one_way_weight": 1.0,
        "initial_capital": initial_capital,
        "final_capital": float(equity[-1]),
        "total_net_pnl": float(equity[-1] - initial_capital),
        "total_gross_pnl": gross_pnl,
        "total_trading_cost": slippage_cost + commission_cost,
        "total_return": total_return,
        "annualized_return": float(
            (equity[-1] / initial_capital) ** (TRADING_DAYS / len(ordered)) - 1.0
        ),
        "annualized_volatility": float(daily_std * np.sqrt(TRADING_DAYS)),
        "sharpe_ratio": (
            float(np.mean(daily_returns) / daily_std * np.sqrt(TRADING_DAYS))
            if daily_std > 0
            else 0.0
        ),
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_net_pnl_per_trade": float(equity[-1] - initial_capital),
    }
    return ledger, summary


def candidate_rules(cost_threshold_pp: float) -> list[TradingRule]:
    thresholds = sorted(
        {
            0.0,
            round(cost_threshold_pp, 6),
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
        }
    )
    rules: list[TradingRule] = []
    for threshold in thresholds:
        rules.append(
            TradingRule(
                name=f"Full_t{threshold:.3f}",
                threshold_pp=threshold,
                sizing="full",
            )
        )
        for risk_multiplier in [0.5, 1.0, 1.5, 2.0]:
            rules.append(
                TradingRule(
                    name=f"Vol_t{threshold:.3f}_r{risk_multiplier:.1f}",
                    threshold_pp=threshold,
                    sizing="volatility",
                    risk_multiplier=risk_multiplier,
                )
            )
    return rules


def select_development_rules(
    development: pd.DataFrame,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, dict[str, TradingRule]]:
    """Select fixed-size, vol-size and primary rules without Block 19."""
    rows: list[dict[str, object]] = []
    rules_by_name: dict[str, TradingRule] = {}
    cost_threshold = round_trip_cost_threshold_pp(commission_bps, slippage_bps)
    for rule in candidate_rules(cost_threshold):
        _, summary = simulate_long_flat(
            development,
            rule,
            initial_capital,
            commission_bps,
            slippage_bps,
        )
        rows.append(summary)
        rules_by_name[rule.name] = rule
    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates["trades"] >= MIN_DEVELOPMENT_TRADES].copy()
    if eligible.empty:
        raise RuntimeError("No trading rule satisfies the minimum-trade rule.")

    def best_for_sizing(sizing: str) -> TradingRule:
        group = eligible[eligible["sizing"] == sizing]
        selected = group.sort_values(
            ["sharpe_ratio", "total_return", "max_drawdown", "threshold_pp"],
            ascending=[False, False, False, True],
        ).iloc[0]
        return rules_by_name[str(selected["strategy"])]

    best_full = best_for_sizing("full")
    best_volatility = best_for_sizing("volatility")
    finalist_names = [best_full.name, best_volatility.name]
    primary_row = eligible[eligible["strategy"].isin(finalist_names)].sort_values(
        ["sharpe_ratio", "total_return", "max_drawdown"],
        ascending=[False, False, False],
    ).iloc[0]
    return candidates, {
        "best_full": best_full,
        "best_volatility": best_volatility,
        "primary": rules_by_name[str(primary_row["strategy"])],
    }


def load_experiment_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Attach prices and strictly lagged volatility to saved OOF predictions."""
    _, _, pretest, test = experiment_frames()
    all_data = pd.concat([pretest, test]).sort_index()
    excluded, blocks = equal_chronological_blocks(all_data, n_blocks=HOLDOUT_BLOCK)
    development_dates = pd.concat(blocks[9:18]).index
    holdout_dates = blocks[18].index

    market = all_data.copy()
    actual_growth = np.expm1(market["target"].to_numpy(float)) * 100.0
    market["trailing_volatility_20d_pp"] = (
        pd.Series(actual_growth, index=market.index)
        .shift(1)
        .rolling(20, min_periods=10)
        .std()
    )
    market_columns = [
        "entry_date",
        "exit_date",
        "entry_open",
        "exit_close",
        "target",
        "trailing_volatility_20d_pp",
    ]

    development_predictions = pd.read_csv(
        PREDICTION_ROOT / "development_oof_predictions.csv",
        parse_dates=["trade_date"],
    ).set_index("trade_date")
    if not development_predictions.index.equals(development_dates):
        raise RuntimeError("Development predictions do not match Blocks 10-18.")
    development = development_predictions.join(market[market_columns], how="left")
    development = development.rename(
        columns={
            "BaggedHuber": "predicted_log_return",
            "target": "market_target_check",
            "entry_open": "entry_open_cny_per_gram",
            "exit_close": "exit_close_cny_per_gram",
        }
    )
    if not np.allclose(
        development["actual_log_return"],
        development["market_target_check"],
    ):
        raise RuntimeError("Development actual returns do not match market data.")
    development["predicted_growth_pp"] = (
        np.expm1(development["predicted_log_return"]) * 100.0
    )
    development = development.reset_index(names="signal_date")

    holdout_predictions = pd.read_csv(
        PREDICTION_ROOT / "locked_block19_predictions.csv",
        parse_dates=["trade_date"],
    ).set_index("trade_date")
    if not holdout_predictions.index.equals(holdout_dates):
        raise RuntimeError("Locked predictions do not match Block 19.")
    holdout = holdout_predictions.join(
        market[["trailing_volatility_20d_pp"]], how="left"
    ).rename(
        columns={
            "trade_date": "signal_date",
            "BaggedHuber_predicted_log_return": "predicted_log_return",
            "BaggedHuber_predicted_growth_pct": "predicted_growth_pp",
            "actual_exit_close_cny_per_gram": "exit_close_cny_per_gram",
        }
    )
    holdout = holdout.reset_index(names="signal_date")
    if development["trailing_volatility_20d_pp"].isna().any():
        raise RuntimeError("Development volatility contains missing values.")
    if holdout["trailing_volatility_20d_pp"].isna().any():
        raise RuntimeError("Holdout volatility contains missing values.")
    if development["signal_date"].max() >= holdout["signal_date"].min():
        raise RuntimeError("Development dates reach into Block 19.")

    metadata = {
        "available_rows": len(all_data),
        "excluded_oldest_remainder_rows": len(excluded),
        "development_blocks": "10-18 rolling OOF predictions",
        "development_rows": len(development),
        "development_start": development["signal_date"].min(),
        "development_end": development["signal_date"].max(),
        "locked_holdout_block": HOLDOUT_BLOCK,
        "holdout_rows": len(holdout),
        "holdout_start": holdout["signal_date"].min(),
        "holdout_end": holdout["signal_date"].max(),
    }
    return development, holdout, metadata


def always_long_rule() -> TradingRule:
    return TradingRule(name="AlwaysLongIntraday", threshold_pp=-1e12, sizing="full")


def zero_threshold_rule() -> TradingRule:
    return TradingRule(name="BaggedHuberZeroThreshold", threshold_pp=0.0, sizing="full")


def always_flat_rule() -> TradingRule:
    return TradingRule(name="AlwaysFlat", threshold_pp=1e12, sizing="full")


def save_equity_chart(ledgers: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for name, ledger in ledgers.items():
        axes[0].plot(
            pd.to_datetime(ledger["entry_date"]),
            ledger["equity"],
            label=name,
            linewidth=1.3,
        )
        axes[1].plot(
            pd.to_datetime(ledger["entry_date"]),
            ledger["drawdown"] * 100.0,
            label=name,
            linewidth=1.1,
        )
    axes[0].set_title("Locked Block 19 direct-gold strategy equity")
    axes[0].set_ylabel("capital (CNY)")
    axes[0].legend()
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("drawdown (%)")
    axes[1].set_xlabel("entry date")
    axes[1].legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_costs(args.commission_bps, args.slippage_bps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development, holdout, split_metadata = load_experiment_frames()

    candidates, selected = select_development_rules(
        development,
        args.initial_capital,
        args.commission_bps,
        args.slippage_bps,
    )
    candidate_summary = candidates.copy()
    candidate_summary["selected_best_full"] = (
        candidate_summary["strategy"] == selected["best_full"].name
    )
    candidate_summary["selected_best_volatility"] = (
        candidate_summary["strategy"] == selected["best_volatility"].name
    )
    candidate_summary["selected_primary"] = (
        candidate_summary["strategy"] == selected["primary"].name
    )
    candidate_summary.to_csv(
        args.output_dir / "development_rule_candidates.csv", index=False
    )

    block_rows = []
    for score_block, group in development.groupby("score_block", sort=True):
        _, block_summary = simulate_long_flat(
            group,
            selected["primary"],
            args.initial_capital,
            args.commission_bps,
            args.slippage_bps,
        )
        block_rows.append(
            {
                "score_block": int(score_block),
                "start": group["signal_date"].min(),
                "end": group["signal_date"].max(),
                **block_summary,
            }
        )
    pd.DataFrame(block_rows).to_csv(
        args.output_dir / "development_primary_block_stability.csv", index=False
    )

    comparison_rules = [
        always_flat_rule(),
        always_long_rule(),
        zero_threshold_rule(),
        selected["best_full"],
        selected["best_volatility"],
    ]
    unique_rules = {rule.name: rule for rule in comparison_rules}
    holdout_ledgers: dict[str, pd.DataFrame] = {}
    holdout_summaries: list[dict[str, object]] = []
    for rule in unique_rules.values():
        ledger, summary = simulate_long_flat(
            holdout,
            rule,
            args.initial_capital,
            args.commission_bps,
            args.slippage_bps,
        )
        holdout_ledgers[rule.name] = ledger
        summary["selected_primary_on_development"] = rule.name == selected["primary"].name
        holdout_summaries.append(summary)
        ledger.to_csv(
            args.output_dir / f"locked_block19_ledger_{rule.name}.csv",
            index=False,
        )

    buy_hold_ledger, buy_hold_summary = simulate_buy_and_hold(
        holdout,
        args.initial_capital,
        args.commission_bps,
        args.slippage_bps,
    )
    buy_hold_summary["selected_primary_on_development"] = False
    holdout_ledgers["BuyAndHold"] = buy_hold_ledger
    holdout_summaries.append(buy_hold_summary)
    buy_hold_ledger.to_csv(
        args.output_dir / "locked_block19_ledger_BuyAndHold.csv", index=False
    )

    summaries = pd.DataFrame(holdout_summaries).sort_values(
        "total_return", ascending=False
    )
    summaries.to_csv(args.output_dir / "locked_block19_strategy_summary.csv", index=False)
    primary_ledger = holdout_ledgers[selected["primary"].name]
    primary_ledger.to_csv(
        args.output_dir / "locked_block19_primary_trade_ledger.csv", index=False
    )
    save_equity_chart(
        holdout_ledgers,
        args.output_dir / "locked_block19_equity_comparison.png",
    )

    cost_scenarios = [
        ("no_costs", 0.0, 0.0),
        (
            "half_costs",
            args.commission_bps / 2.0,
            args.slippage_bps / 2.0,
        ),
        ("base_costs", args.commission_bps, args.slippage_bps),
        (
            "double_costs",
            args.commission_bps * 2.0,
            args.slippage_bps * 2.0,
        ),
    ]
    sensitivity_rows = []
    for scenario, commission_bps, slippage_bps in cost_scenarios:
        _, sensitivity = simulate_long_flat(
            holdout,
            selected["primary"],
            args.initial_capital,
            commission_bps,
            slippage_bps,
        )
        sensitivity_rows.append(
            {
                "scenario": scenario,
                "commission_bps_per_side": commission_bps,
                "slippage_bps_per_side": slippage_bps,
                **sensitivity,
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output_dir / "locked_block19_cost_sensitivity.csv", index=False
    )

    configuration = {
        **split_metadata,
        "forecast_model": "BaggedHuber",
        "forecast_horizon_sessions": 1,
        "signal_timing": "after signal-date close",
        "entry_timing": "next-session open",
        "exit_timing": "same-session close",
        "position_constraint": "theoretical direct Au99.99 long/flat",
        "initial_capital_cny": args.initial_capital,
        "commission_bps_per_side": args.commission_bps,
        "slippage_bps_per_side": args.slippage_bps,
        "round_trip_break_even_threshold_pp": round_trip_cost_threshold_pp(
            args.commission_bps, args.slippage_bps
        ),
        "minimum_development_trades": MIN_DEVELOPMENT_TRADES,
        "selection_objective": (
            "highest development OOF net Sharpe; tie-break total return, "
            "max drawdown and lower threshold"
        ),
        "selected_rules": {key: asdict(rule) for key, rule in selected.items()},
        "holdout_used_for_rule_selection": False,
        "holdout_used_for_model_training": False,
        "historical_exposure_warning": (
            "Block 19 was exposed in earlier model diagnostics. This code locks "
            "it computationally but does not restore human-level blindness."
        ),
        "cost_warning": (
            "Default costs are research assumptions. Replace them with the exact "
            "broker, venue and instrument schedule before live use."
        ),
    }
    with (args.output_dir / "selected_trading_rules.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2, default=str)

    print("Development-selected rules:")
    for key, rule in selected.items():
        print(f"  {key}: {rule}")
    print("\nLocked Block 19 strategy results:")
    print(
        summaries[
            [
                "strategy",
                "trades",
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "total_trading_cost",
                "selected_primary_on_development",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
