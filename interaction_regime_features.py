"""Training-fitted interaction terms and categorical market regimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


INTERACTION_FEATURES = [
    "momentum_5d_x_volatility_20d",
    "real_yield_change_x_usdcnh",
    "premium_x_usdcnh",
    "xauusd_return_x_usdcnh",
    "real_yield_change_x_xauusd_return",
]

REGIME_FEATURES = [
    "market_state_high_volatility",
    "market_state_strong_uptrend",
    "market_state_strong_downtrend",
    "market_state_liquidity_stress",
]

REGIME_INTERACTION_FEATURES = [
    "real_yield_change_x_high_volatility",
    "xauusd_return_x_high_volatility",
    "premium_change_x_liquidity_stress",
]

INTERACTION_PARENTS = {
    "momentum_5d_x_volatility_20d": ["momentum_5d", "volatility_20d"],
    "real_yield_change_x_usdcnh": [
        "real_yield_change_5d", "usdcnh_momentum_5d"
    ],
    "premium_x_usdcnh": [
        "shanghai_gold_premium", "usdcnh_momentum_5d"
    ],
    "xauusd_return_x_usdcnh": [
        "xauusd_return_1d", "usdcnh_momentum_5d"
    ],
    "real_yield_change_x_xauusd_return": [
        "real_yield_change_5d", "xauusd_return_1d"
    ],
    "real_yield_change_x_high_volatility": [
        "real_yield_change_5d", "market_state_high_volatility"
    ],
    "xauusd_return_x_high_volatility": [
        "xauusd_return_1d", "market_state_high_volatility"
    ],
    "premium_change_x_liquidity_stress": [
        "premium_change_5d", "market_state_liquidity_stress"
    ],
}


def enforce_feature_hierarchy(features: list[str]) -> list[str]:
    """Retain both parent terms whenever an interaction is retained."""
    selected = list(features)
    for interaction, parents in INTERACTION_PARENTS.items():
        if interaction not in selected:
            continue
        for parent in parents:
            if parent not in selected:
                selected.append(parent)
    return selected


@dataclass(frozen=True)
class RegimeThresholds:
    volatility_20d_high: float
    momentum_20d_low: float
    momentum_20d_high: float
    price_range_high: float
    volume_change_5d_low: float


def fit_regime_thresholds(train: pd.DataFrame) -> RegimeThresholds:
    """Estimate every regime boundary from training data only."""
    return RegimeThresholds(
        volatility_20d_high=float(train["volatility_20d"].quantile(0.80)),
        momentum_20d_low=float(train["momentum_20d"].quantile(0.20)),
        momentum_20d_high=float(train["momentum_20d"].quantile(0.80)),
        price_range_high=float(train["price_range"].quantile(0.80)),
        volume_change_5d_low=float(train["volume_change_5d"].quantile(0.20)),
    )


def transform_interactions_and_regimes(
    frame: pd.DataFrame,
    thresholds: RegimeThresholds,
) -> pd.DataFrame:
    """Apply fixed thresholds and construct point-in-time feature columns."""
    data = frame.copy()

    data["momentum_5d_x_volatility_20d"] = (
        data["momentum_5d"] * data["volatility_20d"]
    )
    data["real_yield_change_x_usdcnh"] = (
        data["real_yield_change_5d"] * data["usdcnh_momentum_5d"]
    )
    data["premium_x_usdcnh"] = (
        data["shanghai_gold_premium"] * data["usdcnh_momentum_5d"]
    )
    data["xauusd_return_x_usdcnh"] = (
        data["xauusd_return_1d"] * data["usdcnh_momentum_5d"]
    )
    data["real_yield_change_x_xauusd_return"] = (
        data["real_yield_change_5d"] * data["xauusd_return_1d"]
    )

    high_volatility = data["volatility_20d"] > thresholds.volatility_20d_high
    strong_uptrend = data["momentum_20d"] > thresholds.momentum_20d_high
    strong_downtrend = data["momentum_20d"] < thresholds.momentum_20d_low
    liquidity_stress = (
        (data["price_range"] > thresholds.price_range_high)
        & (data["volume_change_5d"] < thresholds.volume_change_5d_low)
    )

    # State 0 is the omitted normal-state reference. Stress overrides high
    # volatility, which overrides the directional trend states.
    state = pd.Series(0, index=data.index, dtype="int64")
    state.loc[strong_uptrend] = 2
    state.loc[strong_downtrend] = 3
    state.loc[high_volatility] = 1
    state.loc[liquidity_stress] = 4
    data["market_state"] = state
    for number, name in enumerate(REGIME_FEATURES, start=1):
        data[name] = (state == number).astype(float)

    data["real_yield_change_x_high_volatility"] = (
        data["real_yield_change_5d"]
        * data["market_state_high_volatility"]
    )
    data["xauusd_return_x_high_volatility"] = (
        data["xauusd_return_1d"]
        * data["market_state_high_volatility"]
    )
    data["premium_change_x_liquidity_stress"] = (
        data["premium_change_5d"]
        * data["market_state_liquidity_stress"]
    )
    return data


def feature_blocks(
    baseline_features: list[str],
) -> dict[str, list[str]]:
    """Candidate feature pools evaluated with the same AIC reducer."""
    return {
        "baseline": list(baseline_features),
        "interactions": list(baseline_features) + INTERACTION_FEATURES,
        "regimes": list(baseline_features) + REGIME_FEATURES,
        "combined": (
            list(baseline_features)
            + INTERACTION_FEATURES
            + REGIME_FEATURES
            + REGIME_INTERACTION_FEATURES
        ),
    }


def threshold_record(thresholds: RegimeThresholds) -> dict[str, float]:
    return asdict(thresholds)


def market_state_counts(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = {
        0: "normal",
        1: "high_volatility",
        2: "strong_uptrend",
        3: "strong_downtrend",
        4: "liquidity_stress_proxy",
    }
    for sample, frame in frames.items():
        counts = frame["market_state"].value_counts().sort_index()
        for state, count in counts.items():
            rows.append(
                {
                    "sample": sample,
                    "market_state": int(state),
                    "label": labels[int(state)],
                    "observations": int(count),
                    "share": float(count / len(frame)),
                }
            )
    return pd.DataFrame(rows)
