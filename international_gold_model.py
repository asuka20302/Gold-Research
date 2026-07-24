"""Five-model XAU/USD forecast with a common, untouched Block 19 holdout.

This is the maintained international-gold entry point.  It never loads SGE
Au99.99 as the prediction target.  With full FXCM bid/ask OHLC it predicts the
next-session open-to-horizon-close return.  With the project's recovered FXCM
close cache it predicts close-to-close returns and explicitly marks the result
as research-only rather than an executable P&L backtest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from international_gold_data import load_international_gold
from statistical_feature_selection import (
    information_criterion_backward,
    ols_statistics,
)


MODEL_NAMES = ["OLS", "Ridge", "Lasso", "ElasticNet", "BaggedHuber"]
BASE_FEATURES = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_5d",
    "volatility_20d",
    "ma_gap_20d",
]
EXTERNAL_FEATURES = [
    "usdcnh_return_1d",
    "usdcnh_momentum_5d",
    "dollar_return_1d",
    "dollar_momentum_5d",
    "oil_return_1d",
    "oil_momentum_5d",
    "sp500_return_1d",
    "sp500_momentum_5d",
    "real_yield_10y",
    "real_yield_change_1d",
    "real_yield_change_5d",
    "nominal_yield_change_1d",
    "nominal_yield_change_5d",
    "breakeven_inflation_10y",
    "vix_close_known",
    "vix_momentum_5d",
    "gvz_close_known",
    "gvz_momentum_5d",
]
N_BLOCKS = 19
TRAIN_BLOCKS = 9
HOLDOUT_BLOCK = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast international XAU/USD prices with five linear/robust "
            "models and one amount-error weighted ensemble."
        )
    )
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD")
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y%m%d"),
        help="YYYYMMDD",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--target-mode",
        choices=["auto", "open_to_close", "close_to_close"],
        default="auto",
        help=(
            "auto uses executable open-to-close only with full bid/ask OHLC; "
            "otherwise it uses research-only close-to-close."
        ),
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/international_gold"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/international_gold"),
    )
    parser.add_argument("--bagging-estimators", type=int, default=50)
    parser.add_argument("--bagging-block-size", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def make_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


class MovingBlockBaggedRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """Average Huber models fitted to moving-block bootstrap samples."""

    def __init__(
        self,
        base_model: Pipeline,
        n_estimators: int = 50,
        block_size: int = 20,
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.block_size = block_size
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "MovingBlockBaggedRegressor":
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least one.")
        if not 1 <= self.block_size <= len(X):
            raise ValueError("block_size must be between one and len(X).")
        generator = np.random.default_rng(self.random_state)
        maximum_start = len(X) - self.block_size
        self.estimators_: list[Pipeline] = []
        for _ in range(self.n_estimators):
            sampled: list[int] = []
            while len(sampled) < len(X):
                start = int(generator.integers(0, maximum_start + 1))
                sampled.extend(range(start, start + self.block_size))
            indices = np.asarray(sampled[: len(X)])
            model = clone(self.base_model)
            model.fit(X.iloc[indices], y.iloc[indices])
            self.estimators_.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise RuntimeError("The bagged model is not fitted.")
        predictions = np.column_stack(
            [model.predict(X) for model in self.estimators_]
        )
        return predictions.mean(axis=1)


def _factor_cache(cache_dir: Path) -> Path | None:
    candidates = sorted(
        (cache_dir / "factors").glob("macro_factors_v3_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_external_factors(cache_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load only non-Chinese, non-target macro factors from the local cache."""
    path = _factor_cache(cache_dir)
    if path is None:
        return pd.DataFrame(), ""
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    available = [column for column in EXTERNAL_FEATURES if column in frame]
    if not available:
        return pd.DataFrame(), str(path)
    factors = frame.set_index("trade_date")[available].apply(
        pd.to_numeric, errors="coerce"
    )
    return factors.sort_index(), str(path)


def resolve_target_mode(prices: pd.DataFrame, requested: str) -> str:
    has_open = "mid_open" in prices and prices["mid_open"].notna().any()
    if requested == "open_to_close" and not has_open:
        raise RuntimeError(
            "open_to_close requires full FXCM bid/ask OHLC. Refresh with a "
            "valid TUSHARE_TOKEN or choose close_to_close."
        )
    if requested == "auto":
        return "open_to_close" if has_open else "close_to_close"
    return requested


def build_dataset(
    prices: pd.DataFrame,
    horizon: int,
    target_mode: str,
    external_factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct close-known XAU features and a strictly future target."""
    if horizon < 1:
        raise ValueError("horizon must be at least one session.")
    data = prices.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = (
        data.dropna(subset=["trade_date", "mid_close"])
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")
    )
    data["mid_close"] = pd.to_numeric(data["mid_close"], errors="coerce")
    if (data["mid_close"] <= 0).any():
        raise ValueError("XAU/USD mid closes must be positive.")

    log_close = np.log(data["mid_close"])
    daily_return = log_close.diff()
    data["return_1d"] = daily_return
    data["momentum_5d"] = log_close.diff(5)
    data["momentum_20d"] = log_close.diff(20)
    data["volatility_5d"] = daily_return.rolling(5).std()
    data["volatility_20d"] = daily_return.rolling(20).std()
    data["ma_gap_20d"] = (
        data["mid_close"] / data["mid_close"].rolling(20).mean() - 1.0
    )

    if external_factors is not None and not external_factors.empty:
        # These cached macro factors were already shifted one source session
        # for the old Shanghai close.  Keeping that lag is conservative for
        # an international signal and avoids same-calendar-time ambiguity.
        aligned = external_factors.reindex(data.index, method="ffill")
        data = data.join(aligned, how="left")

    dates = pd.Series(data.index, index=data.index)
    data["signal_close"] = data["mid_close"]
    data["exit_date"] = dates.shift(-horizon)
    data["exit_close"] = data["mid_close"].shift(-horizon)
    if target_mode == "open_to_close":
        data["entry_date"] = dates.shift(-1)
        data["entry_price"] = pd.to_numeric(
            data["mid_open"], errors="coerce"
        ).shift(-1)
    elif target_mode == "close_to_close":
        data["entry_date"] = dates
        data["entry_price"] = data["mid_close"]
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    data["target"] = np.log(data["exit_close"] / data["entry_price"])

    present_external = [
        feature for feature in EXTERNAL_FEATURES if feature in data
    ]
    columns = [
        *BASE_FEATURES,
        *present_external,
        "entry_date",
        "exit_date",
        "signal_close",
        "entry_price",
        "exit_close",
        "target",
    ]
    model_data = data[columns].replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(
        subset=[*BASE_FEATURES, "entry_price", "exit_close", "target"]
    )
    if len(model_data) < 500:
        raise RuntimeError(
            f"Only {len(model_data)} usable XAU/USD rows were constructed."
        )
    return model_data


def equal_chronological_blocks(
    frame: pd.DataFrame,
    n_blocks: int = N_BLOCKS,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Drop the oldest remainder, then create equal consecutive blocks."""
    ordered = frame.sort_index()
    block_size = len(ordered) // n_blocks
    if block_size < 20:
        raise RuntimeError("Too few rows for 19 meaningful blocks.")
    used_rows = block_size * n_blocks
    excluded = ordered.iloc[: len(ordered) - used_rows].copy()
    used = ordered.iloc[len(ordered) - used_rows :].copy()
    blocks = [
        used.iloc[index * block_size : (index + 1) * block_size].copy()
        for index in range(n_blocks)
    ]
    return excluded, blocks


def purge_unavailable_labels(
    training: pd.DataFrame,
    scoring_start: pd.Timestamp,
) -> pd.DataFrame:
    """Remove training rows whose future close is not known at score time."""
    exit_dates = pd.to_datetime(training["exit_date"])
    purged = training.loc[exit_dates < scoring_start].copy()
    if purged.empty:
        raise RuntimeError("Label purging removed the entire training window.")
    return purged


def growth_mae(actual: pd.Series | np.ndarray, prediction: np.ndarray) -> float:
    actual_growth = np.expm1(np.asarray(actual, dtype=float)) * 100.0
    predicted_growth = np.expm1(np.asarray(prediction, dtype=float)) * 100.0
    return float(np.mean(np.abs(predicted_growth - actual_growth)))


def candidate_models() -> dict[str, list[Pipeline]]:
    return {
        "OLS": [make_pipeline(LinearRegression())],
        "Ridge": [
            make_pipeline(Ridge(alpha=alpha))
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]
        ],
        "Lasso": [
            make_pipeline(Lasso(alpha=alpha, max_iter=50_000))
            for alpha in [1e-6, 1e-5, 1e-4, 1e-3]
        ],
        "ElasticNet": [
            make_pipeline(
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=50_000,
                )
            )
            for alpha in [1e-6, 1e-5, 1e-4, 1e-3]
            for l1_ratio in [0.1, 0.5, 0.9]
        ],
        "BaggedHuber": [
            make_pipeline(
                HuberRegressor(
                    epsilon=epsilon,
                    alpha=alpha,
                    max_iter=5_000,
                )
            )
            for epsilon in [1.1, 1.35, 1.5]
            for alpha in [0.0001, 0.001, 0.01]
        ],
    }


def select_model_templates(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Select every model by validation growth MAE, not direction accuracy."""
    selected: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []
    for name, candidates in candidate_models().items():
        best_score = np.inf
        best: Pipeline | None = None
        for candidate in candidates:
            fitted = clone(candidate).fit(
                training[features], training["target"]
            )
            prediction = np.asarray(
                fitted.predict(validation[features]), dtype=float
            )
            score = growth_mae(validation["target"], prediction)
            if score < best_score:
                best_score = score
                best = clone(candidate)
        if best is None:
            raise RuntimeError(f"No valid {name} candidate.")
        selected[name] = best
        rows.append(
            {
                "model": name,
                "selection_metric": "growth_mae_percentage_points",
                "validation_growth_mae_pp": best_score,
                "selected_parameters": json.dumps(
                    best.named_steps["model"].get_params(),
                    default=str,
                    sort_keys=True,
                ),
            }
        )
    return selected, pd.DataFrame(rows)


def build_fitted_model(
    name: str,
    template: Pipeline,
    training: pd.DataFrame,
    features: list[str],
    bagging_estimators: int,
    bagging_block_size: int,
    random_state: int,
) -> object:
    if name == "BaggedHuber":
        model: object = MovingBlockBaggedRegressor(
            base_model=template,
            n_estimators=bagging_estimators,
            block_size=min(bagging_block_size, len(training)),
            random_state=random_state,
        )
    else:
        model = clone(template)
    model.fit(training[features], training["target"])
    return model


def amount_metrics(
    actual: pd.Series | np.ndarray,
    prediction: np.ndarray,
    entry_price: pd.Series | np.ndarray,
    exit_close: pd.Series | np.ndarray,
) -> dict[str, float]:
    actual_log = np.asarray(actual, dtype=float)
    predicted_log = np.asarray(prediction, dtype=float)
    entry = np.asarray(entry_price, dtype=float)
    actual_exit = np.asarray(exit_close, dtype=float)
    predicted_exit = entry * np.exp(predicted_log)
    actual_growth = np.expm1(actual_log) * 100.0
    predicted_growth = np.expm1(predicted_log) * 100.0
    growth_error = predicted_growth - actual_growth
    price_error = predicted_exit - actual_exit
    actual_std = float(np.std(actual_growth))
    correlation = (
        float(np.corrcoef(actual_growth, predicted_growth)[0, 1])
        if actual_std > 0 and np.std(predicted_growth) > 0
        else np.nan
    )
    return {
        "growth_mae_pp": float(np.mean(np.abs(growth_error))),
        "growth_rmse_pp": float(np.sqrt(np.mean(growth_error**2))),
        "growth_bias_pp": float(np.mean(growth_error)),
        "growth_median_ae_pp": float(np.median(np.abs(growth_error))),
        "growth_p90_ae_pp": float(np.quantile(np.abs(growth_error), 0.90)),
        "price_mae_usd_per_oz": float(np.mean(np.abs(price_error))),
        "price_rmse_usd_per_oz": float(
            np.sqrt(np.mean(price_error**2))
        ),
        "price_bias_usd_per_oz": float(np.mean(price_error)),
        "correlation": correlation,
        "amplitude_ratio": (
            float(np.std(predicted_growth) / actual_std)
            if actual_std > 0
            else np.nan
        ),
        "log_return_mae": float(
            mean_absolute_error(actual_log, predicted_log)
        ),
        "log_return_mse": float(
            mean_squared_error(actual_log, predicted_log)
        ),
    }


def model_coefficients(
    fitted_models: dict[str, object],
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, fitted in fitted_models.items():
        if name == "BaggedHuber":
            coefficients = np.mean(
                np.vstack(
                    [
                        estimator.named_steps["model"].coef_
                        for estimator in fitted.estimators_
                    ]
                ),
                axis=0,
            )
        else:
            coefficients = np.asarray(
                fitted.named_steps["model"].coef_, dtype=float
            )
        for feature, coefficient in zip(features, coefficients, strict=True):
            rows.append(
                {
                    "model": name,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                }
            )
    return pd.DataFrame(rows)


def save_figure(predictions: pd.DataFrame, path: Path) -> None:
    dated = predictions.copy()
    dated["exit_date"] = pd.to_datetime(dated["exit_date"])
    figure, axes = plt.subplots(
        2, 1, figsize=(14, 10), constrained_layout=True
    )
    axes[0].plot(
        dated["exit_date"],
        dated["actual_exit_close_usd_per_oz"],
        color="black",
        linewidth=1.8,
        label="actual XAU/USD",
    )
    for name in [*MODEL_NAMES, "WeightedEnsemble"]:
        axes[0].plot(
            dated["exit_date"],
            dated[f"{name}_predicted_close_usd_per_oz"],
            linewidth=1.0,
            alpha=0.8,
            label=name,
        )
    axes[0].set_title("Locked Block 19 XAU/USD price forecasts")
    axes[0].set_ylabel("USD per troy ounce")
    axes[0].legend(ncol=3)

    actual_growth = dated["actual_growth_pct"]
    axes[1].plot(
        dated["exit_date"],
        actual_growth,
        color="black",
        linewidth=1.5,
        label="actual growth",
    )
    axes[1].plot(
        dated["exit_date"],
        dated["WeightedEnsemble_predicted_growth_pct"],
        color="tab:blue",
        linewidth=1.2,
        label="weighted forecast",
    )
    axes[1].axhline(0.0, color="grey", linewidth=0.8)
    axes[1].set_ylabel("ordinary growth (%)")
    axes[1].set_title("Predicted amount versus actual amount")
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    prices, source_metadata = load_international_gold(
        args.start_date,
        args.end_date,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    target_mode = resolve_target_mode(prices, args.target_mode)
    factors, factor_cache = load_external_factors(args.cache_dir)
    dataset = build_dataset(
        prices,
        horizon=args.horizon,
        target_mode=target_mode,
        external_factors=factors,
    )
    excluded, blocks = equal_chronological_blocks(dataset)
    block_size = len(blocks[0])
    holdout = blocks[HOLDOUT_BLOCK - 1]

    initial_training = pd.concat(blocks[:TRAIN_BLOCKS]).sort_index()
    initial_training = purge_unavailable_labels(
        initial_training, blocks[TRAIN_BLOCKS].index.min()
    )
    active_features = [
        feature
        for feature in [*BASE_FEATURES, *EXTERNAL_FEATURES]
        if feature in initial_training
        and initial_training[feature].notna().any()
        and initial_training[feature].nunique(dropna=True) > 1
    ]
    selected_features, elimination_rows = information_criterion_backward(
        initial_training,
        active_features,
        criterion="aic",
        alpha=0.05,
    )
    initial_ols = ols_statistics(initial_training, active_features)
    coefficient_tests = pd.DataFrame(
        {
            "feature": initial_ols.coefficients.index,
            "coefficient": initial_ols.coefficients.to_numpy(float),
            "standard_error": initial_ols.standard_errors.to_numpy(float),
            "t_statistic": initial_ols.t_statistics.to_numpy(float),
            "p_value": initial_ols.p_values.to_numpy(float),
        }
    )

    selection_validation = blocks[TRAIN_BLOCKS]
    templates, selections = select_model_templates(
        initial_training,
        selection_validation,
        selected_features,
    )
    selections["active_features"] = ",".join(selected_features)

    oof_parts: list[pd.DataFrame] = []
    # Block 10 selected hyperparameters; Blocks 11-18 estimate voting weights.
    for score_block in range(11, 19):
        start = score_block - TRAIN_BLOCKS - 1
        training = pd.concat(
            blocks[start : start + TRAIN_BLOCKS]
        ).sort_index()
        scoring = blocks[score_block - 1]
        training = purge_unavailable_labels(
            training, scoring.index.min()
        )
        part = pd.DataFrame(
            {
                "signal_date": scoring.index,
                "actual_log_return": scoring["target"].to_numpy(float),
            }
        ).set_index("signal_date")
        for offset, name in enumerate(MODEL_NAMES):
            fitted = build_fitted_model(
                name,
                templates[name],
                training,
                selected_features,
                args.bagging_estimators,
                args.bagging_block_size,
                args.random_state + score_block * 10 + offset,
            )
            part[name] = np.asarray(
                fitted.predict(scoring[selected_features]), dtype=float
            )
        oof_parts.append(part)
        print(
            f"Weight OOF Block {score_block:02d}: "
            f"train={len(training)}, score={len(scoring)}"
        )
    development_oof = pd.concat(oof_parts).sort_index()
    oof_mae = pd.Series(
        {
            name: growth_mae(
                development_oof["actual_log_return"],
                development_oof[name].to_numpy(float),
            )
            for name in MODEL_NAMES
        },
        name="development_growth_mae_pp",
    )
    inverse = 1.0 / np.maximum(oof_mae.to_numpy(float), 1e-15)
    weights = pd.Series(
        inverse / inverse.sum(),
        index=MODEL_NAMES,
        name="weight",
    )
    development_oof["WeightedEnsemble"] = (
        development_oof[MODEL_NAMES].to_numpy(float)
        @ weights.to_numpy(float)
    )

    final_training = pd.concat(blocks[9:18]).sort_index()
    final_training = purge_unavailable_labels(
        final_training, holdout.index.min()
    )
    fitted_models: dict[str, object] = {}
    holdout_predictions: dict[str, np.ndarray] = {}
    for offset, name in enumerate(MODEL_NAMES):
        fitted = build_fitted_model(
            name,
            templates[name],
            final_training,
            selected_features,
            args.bagging_estimators,
            args.bagging_block_size,
            args.random_state + offset,
        )
        fitted_models[name] = fitted
        holdout_predictions[name] = np.asarray(
            fitted.predict(holdout[selected_features]), dtype=float
        )
        joblib.dump(fitted, model_dir / f"{name}.joblib")
    holdout_predictions["WeightedEnsemble"] = (
        np.column_stack(
            [holdout_predictions[name] for name in MODEL_NAMES]
        )
        @ weights.to_numpy(float)
    )
    holdout_predictions["HistoricalMean"] = np.full(
        len(holdout), final_training["target"].mean(), dtype=float
    )
    holdout_predictions["ZeroForecast"] = np.zeros(len(holdout), dtype=float)

    metric_rows = []
    for name, prediction in holdout_predictions.items():
        metric_rows.append(
            {
                "candidate": name,
                "candidate_type": (
                    "ensemble"
                    if name == "WeightedEnsemble"
                    else (
                        "baseline"
                        if name in {"HistoricalMean", "ZeroForecast"}
                        else "model"
                    )
                ),
                **amount_metrics(
                    holdout["target"],
                    prediction,
                    holdout["entry_price"],
                    holdout["exit_close"],
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("growth_mae_pp")

    prediction_table = pd.DataFrame(
        {
            "signal_date": holdout.index,
            "entry_date": pd.to_datetime(
                holdout["entry_date"]
            ).to_numpy(),
            "exit_date": pd.to_datetime(
                holdout["exit_date"]
            ).to_numpy(),
            "entry_price_usd_per_oz": holdout[
                "entry_price"
            ].to_numpy(float),
            "actual_exit_close_usd_per_oz": holdout[
                "exit_close"
            ].to_numpy(float),
            "actual_log_return": holdout["target"].to_numpy(float),
            "actual_growth_pct": np.expm1(
                holdout["target"].to_numpy(float)
            )
            * 100.0,
        }
    )
    for name, prediction in holdout_predictions.items():
        prediction_table[f"{name}_predicted_log_return"] = prediction
        prediction_table[f"{name}_predicted_growth_pct"] = (
            np.expm1(prediction) * 100.0
        )
        prediction_table[f"{name}_predicted_close_usd_per_oz"] = (
            prediction_table["entry_price_usd_per_oz"].to_numpy(float)
            * np.exp(prediction)
        )

    coefficient_table = model_coefficients(
        fitted_models, selected_features
    )
    weight_table = pd.concat([oof_mae, weights], axis=1).reset_index(
        names="model"
    )
    elimination = pd.DataFrame(elimination_rows)
    metadata = {
        "instrument": "XAUUSD.FXCM",
        "unit": "USD per troy ounce",
        "source": source_metadata,
        "factor_cache": factor_cache,
        "target_mode": target_mode,
        "target_definition": (
            "log(horizon close / next-session mid open)"
            if target_mode == "open_to_close"
            else "log(horizon close / signal-session mid close)"
        ),
        "pnl_status": (
            "executable research target; bid/ask P&L still requires a "
            "separate frozen strategy rule"
            if target_mode == "open_to_close"
            else "not executable: close-only fallback has no next-open quote"
        ),
        "horizon_sessions": args.horizon,
        "dataset_rows": len(dataset),
        "dataset_start": dataset.index.min().isoformat(),
        "dataset_end": dataset.index.max().isoformat(),
        "oldest_remainder_rows_excluded": len(excluded),
        "block_size": block_size,
        "selection_train_blocks": "1-9",
        "hyperparameter_validation_block": 10,
        "weight_oof_blocks": "11-18",
        "final_training_blocks": "10-18 with unavailable labels purged",
        "holdout_block": 19,
        "holdout_start": holdout.index.min().isoformat(),
        "holdout_end": holdout.index.max().isoformat(),
        "feature_selection": (
            "training-only p>0.05 removal candidates accepted only when "
            "AIC improves"
        ),
        "active_features_before_aic": active_features,
        "selected_features": selected_features,
        "models": MODEL_NAMES,
        "diagnostic_baselines": ["HistoricalMean", "ZeroForecast"],
        "ensemble_weight_rule": (
            "normalized inverse growth MAE on Blocks 11-18"
        ),
        "directional_accuracy_used": False,
    }

    outputs = {
        "locked_block19_metrics.csv": metrics,
        "locked_block19_predictions.csv": prediction_table,
        "development_weights.csv": weight_table,
        "selected_models.csv": selections,
        "aic_elimination_path.csv": elimination,
        "initial_ols_coefficient_tests.csv": coefficient_tests,
        "final_model_coefficients.csv": coefficient_table,
    }
    for filename, table in outputs.items():
        table.to_csv(args.output_dir / filename, index=False)
        table.to_csv(args.report_dir / filename, index=False)
    development_oof.reset_index().to_csv(
        args.output_dir / "development_oof_predictions.csv", index=False
    )
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    with (args.report_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    save_figure(
        prediction_table,
        args.output_dir / "locked_block19_price_comparison.png",
    )

    print("\nInternational-gold source:")
    print(json.dumps(source_metadata, ensure_ascii=False, indent=2))
    print(f"Target mode: {target_mode}")
    print(f"Selected AIC features ({len(selected_features)}):")
    print(", ".join(selected_features))
    print("\nDevelopment voting weights:")
    print(weight_table.to_string(index=False))
    print("\nLocked Block 19 amount-error metrics:")
    print(
        metrics[
            [
                "candidate",
                "growth_mae_pp",
                "growth_rmse_pp",
                "price_mae_usd_per_oz",
                "price_rmse_usd_per_oz",
                "correlation",
                "amplitude_ratio",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
