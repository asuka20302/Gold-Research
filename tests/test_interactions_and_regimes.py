import pandas as pd

from interaction_regime_features import (
    RegimeThresholds,
    enforce_feature_hierarchy,
    transform_interactions_and_regimes,
)


def test_interaction_hierarchy_adds_both_parent_features() -> None:
    selected = enforce_feature_hierarchy(
        ["momentum_5d_x_volatility_20d"]
    )
    assert "momentum_5d" in selected
    assert "volatility_20d" in selected


def test_regime_labels_are_exclusive_and_use_documented_precedence() -> None:
    frame = pd.DataFrame(
        {
            "momentum_5d": [0.1, 0.1, 0.1, 0.1, 0.1],
            "volatility_20d": [0.1, 0.9, 0.1, 0.1, 0.9],
            "real_yield_change_5d": [0.01] * 5,
            "usdcnh_momentum_5d": [0.02] * 5,
            "shanghai_gold_premium": [0.03] * 5,
            "xauusd_return_1d": [0.04] * 5,
            "momentum_20d": [0.0, 0.0, 0.8, -0.8, 0.0],
            "price_range": [0.1, 0.1, 0.1, 0.1, 0.9],
            "volume_change_5d": [0.0, 0.0, 0.0, 0.0, -0.9],
            "premium_change_5d": [0.01] * 5,
        }
    )
    thresholds = RegimeThresholds(
        volatility_20d_high=0.5,
        momentum_20d_low=-0.5,
        momentum_20d_high=0.5,
        price_range_high=0.5,
        volume_change_5d_low=-0.5,
    )

    transformed = transform_interactions_and_regimes(frame, thresholds)

    assert transformed["market_state"].tolist() == [0, 1, 2, 3, 4]
    dummy_columns = [
        "market_state_high_volatility",
        "market_state_strong_uptrend",
        "market_state_strong_downtrend",
        "market_state_liquidity_stress",
    ]
    assert (transformed[dummy_columns].sum(axis=1) <= 1).all()
    assert transformed.loc[4, "market_state_liquidity_stress"] == 1.0
    assert transformed.loc[4, "market_state_high_volatility"] == 0.0
