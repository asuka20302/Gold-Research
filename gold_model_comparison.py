"""Compare linear models for forecasting Shanghai Gold Au99.99 returns.

Run this file from PyCharm after setting the TUSHARE_TOKEN environment variable.
The script downloads (and caches) daily SGE data, builds lagged features, tunes
each model on a chronological validation set, and evaluates it on an untouched
test set.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tushare_factors import (
    TROY_OUNCE_GRAMS,
    build_tushare_factor_features,
)
from statistical_feature_selection import build_feature_subsets
from interaction_regime_features import (
    REGIME_FEATURES,
    enforce_feature_hierarchy,
    feature_blocks,
    fit_regime_thresholds,
    market_state_counts,
    threshold_record,
    transform_interactions_and_regimes,
)

BASE_FEATURES = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_5d",
    "volatility_20d",
    "price_range",
    "volume_change_5d",
    "ma_gap_20d",
]

FACTOR_FEATURES = [
    "xauusd_return_1d",
    "xauusd_momentum_5d",
    "xauusd_momentum_20d",
    "xauusd_volatility_20d",
    "usdcnh_return_1d",
    "usdcnh_momentum_5d",
    "dollar_momentum_5d",
    "real_yield_10y",
    "real_yield_change_5d",
    "nominal_yield_change_5d",
    "breakeven_inflation_10y",
    "sp500_momentum_5d",
    "sp500_volatility_20d",
    "vix_close_known",
    "vix_momentum_5d",
    "gvz_close_known",
    "gvz_momentum_5d",
    "oil_momentum_5d",
    "shfe_gold_return_1d",
    "shfe_gold_settle_return_1d",
    "shfe_gold_volume_change_5d",
    "shfe_gold_oi_change_5d",
    "shfe_gold_spot_basis",
    "shanghai_gold_premium",
    "premium_change_5d",
    "gold_etf_share_change_1d",
    "gold_etf_share_change_5d",
]

# This is the production set selected on validation data only. Candidate
# factors remain downloaded and available for future research, but are not
# allowed into model fitting until they pass the predeclared MSE/correlation/
# direction rule. The stale legacy dollar series is deliberately excluded.
VALIDATED_FACTOR_FEATURES = [
    "xauusd_return_1d",
    "xauusd_momentum_5d",
    "xauusd_momentum_20d",
    "xauusd_volatility_20d",
    "usdcnh_return_1d",
    "usdcnh_momentum_5d",
    "real_yield_10y",
    "real_yield_change_5d",
    "nominal_yield_change_5d",
    "breakeven_inflation_10y",
    "sp500_momentum_5d",
    "sp500_volatility_20d",
    "oil_momentum_5d",
    "shanghai_gold_premium",
    "premium_change_5d",
    "shfe_gold_return_1d",
    "shfe_gold_settle_return_1d",
]

MAX_HORIZON = 7
VALIDATION_START = pd.Timestamp("2023-01-17")
TEST_START = pd.Timestamp("2024-10-09")
COMMON_TEST_END = pd.Timestamp("2026-06-23")

FEATURES = BASE_FEATURES + VALIDATED_FACTOR_FEATURES

OPEN_UPDATE_EXTRA_FEATURES = [
    "entry_open_jump",
    "entry_open_jump_squared",
    "entry_open_jump_x_volatility_20d",
    "entry_open_jump_x_momentum_5d",
]

OPEN_UPDATE_FEATURE_SETS = {
    "gap_only": FEATURES + ["entry_open_jump"],
    "gap_nonlinear": FEATURES
    + ["entry_open_jump", "entry_open_jump_squared"],
    "gap_interactions": FEATURES + OPEN_UPDATE_EXTRA_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare linear forecasting models on SGE Au99.99."
    )
    parser.add_argument("--start-date", default="20150101", help="YYYYMMDD")
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y%m%d"),
        help="YYYYMMDD",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=range(1, MAX_HORIZON + 1),
        default=5,
    )
    parser.add_argument("--symbol", default="Au99.99")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--bagging-estimators", type=int, default=50)
    parser.add_argument("--bagging-block-size", type=int, default=20)
    parser.add_argument(
        "--disable-huber-bagging",
        action="store_true",
        help="Disable the validation-gated horizon-1 block-bagging wrapper.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-external-factors",
        action="store_true",
        help="Run only the original Au99.99 price/volume features.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--event-exclusion-file",
        type=Path,
        default=Path("data/event_exclusions.csv"),
        help=(
            "CSV containing start_date and end_date event windows. Rows whose "
            "signal, entry, or exit date falls in a window are removed before "
            "the chronological split."
        ),
    )
    parser.add_argument(
        "--ignore-event-exclusions",
        action="store_true",
        help="Run the model without applying the event-exclusion CSV.",
    )
    parser.add_argument(
        "--large-open-jump-threshold",
        type=float,
        default=0.03,
        help=(
            "Flag entry opens where abs(log(entry_open / signal-day close)) "
            "exceeds this value. Default 0.03 is about 3%%."
        ),
    )
    parser.add_argument(
        "--feature-selection",
        choices=["t_test", "anova", "aic", "bic"],
        default="aic",
        help=(
            "Training-only feature selector. The production default is the "
            "p-value-constrained AIC reduced model."
        ),
    )
    parser.add_argument(
        "--feature-selection-alpha",
        type=float,
        default=0.05,
        help="Significance threshold for t-test and ANOVA elimination.",
    )
    return parser.parse_args()


def download_gold_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Load SGE data from a local cache or download it through Tushare."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace(".", "_")
    cache_file = cache_dir / f"{safe_symbol}_{start_date}_{end_date}.csv"

    if cache_file.exists() and not refresh:
        print(f"Loading cached data: {cache_file}")
        return pd.read_csv(cache_file)

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "TUSHARE_TOKEN is not set. In PyCharm, open Run > Edit "
            "Configurations and add TUSHARE_TOKEN=your_token under "
            "Environment variables."
        )

    print(f"Downloading {symbol} from Tushare...")
    pro = ts.pro_api(token)
    first_date = pd.to_datetime(start_date, format="%Y%m%d")
    last_date = pd.to_datetime(end_date, format="%Y%m%d")
    cursor = first_date
    chunks: list[pd.DataFrame] = []

    # sge_daily returns at most 2,000 rows. Yearly requests stay comfortably
    # below that limit and make long histories complete.
    while cursor <= last_date:
        chunk_end = min(
            cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1),
            last_date,
        )
        print(f"  {cursor.date()} to {chunk_end.date()}")
        chunk = pro.sge_daily(
            ts_code=symbol,
            start_date=cursor.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
        )
        if chunk is not None and not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(
            "Tushare returned no gold observations for "
            f"{symbol}. The official contract code is Au99.99 (without "
            "'.SGE'). Also check that your account has the 2,000 Tushare "
            "points required for the sge_daily endpoint."
        )

    frame = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    )

    required = {"trade_date", "open", "high", "low", "close", "vol"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"Tushare response is missing required columns: {sorted(missing)}. "
            f"Returned columns: {frame.columns.tolist()}"
        )

    frame.to_csv(cache_file, index=False)
    print(f"Saved raw data: {cache_file}")
    return frame


def build_dataset(
    raw: pd.DataFrame,
    horizon: int,
    external_factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create close-known features and a next-session executable target."""
    if horizon < 1:
        raise ValueError("horizon must be at least one trading day.")

    data = raw.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], format="%Y%m%d")
    data = (
        data.sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")
    )

    numeric_columns = ["open", "high", "low", "close", "vol"]
    if "amount" in data.columns:
        numeric_columns.append("amount")
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if (data["close"] <= 0).any():
        raise ValueError("Gold close prices must be positive.")

    log_price = np.log(data["close"])
    daily_return = log_price.diff()

    data["return_1d"] = daily_return
    data["momentum_5d"] = log_price.diff(5)
    data["momentum_20d"] = log_price.diff(20)
    data["volatility_5d"] = daily_return.rolling(5).std()
    data["volatility_20d"] = daily_return.rolling(20).std()
    data["price_range"] = (data["high"] - data["low"]) / data["close"]
    data["volume_change_5d"] = np.log1p(data["vol"]).diff(5)
    data["ma_gap_20d"] = (
        data["close"] / data["close"].rolling(20).mean() - 1
    )

    if external_factors is not None and not external_factors.empty:
        known_factors = external_factors.sort_index().reindex(
            data.index, method="ffill"
        )
        data = data.join(known_factors, how="left")

        if {
            "xauusd_close_known",
            "usdcnh_close_known",
        }.issubset(data.columns):
            international_rmb_per_gram = (
                data["xauusd_close_known"]
                * data["usdcnh_close_known"]
                / TROY_OUNCE_GRAMS
            )
            data["shanghai_gold_premium"] = (
                data["close"] / international_rmb_per_gram - 1
            )
            data["premium_change_5d"] = data[
                "shanghai_gold_premium"
            ].diff(5)

        if "shfe_gold_close_known" in data.columns:
            data["shfe_gold_spot_basis"] = (
                data["shfe_gold_close_known"] / data["close"] - 1
            )

    missing_factor_features = [
        feature for feature in FACTOR_FEATURES if feature not in data
    ]
    if missing_factor_features:
        print(
            "Unavailable factor features will be neutral constants: "
            + ", ".join(missing_factor_features)
        )
        for feature in missing_factor_features:
            data[feature] = 0.0

    for feature in FACTOR_FEATURES:
        if data[feature].notna().sum() == 0:
            data[feature] = 0.0

    # Features are known after today's close. A realistic trade therefore
    # enters at the next session's open and exits at the horizon-day close.
    dates = pd.Series(data.index, index=data.index)

    data["entry_date"] = dates.shift(-1)
    data["entry_open"] = data["open"].shift(-1)
    data["entry_previous_close"] = data["close"]
    data["entry_open_jump"] = np.log(
        data["entry_open"] / data["entry_previous_close"]
    )
    data["entry_open_jump_squared"] = data["entry_open_jump"].pow(2)
    data["entry_open_jump_x_volatility_20d"] = (
        data["entry_open_jump"] * data["volatility_20d"]
    )
    data["entry_open_jump_x_momentum_5d"] = (
        data["entry_open_jump"] * data["momentum_5d"]
    )

    data["exit_date"] = dates.shift(-horizon)
    data["exit_close"] = data["close"].shift(-horizon)

    data["target"] = np.log(
        data["exit_close"] / data["entry_open"]
    )

    data["target_up"] = (data["target"] > 0).astype(int)

    model_data = data[
        FEATURES
        + [
            "entry_date",
            "exit_date",
            "entry_previous_close",
            "entry_open",
            "entry_open_jump",
            "entry_open_jump_squared",
            "entry_open_jump_x_volatility_20d",
            "entry_open_jump_x_momentum_5d",
            "exit_close",
            "target",
            "target_up",
             ]
        ]
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(
        subset=BASE_FEATURES
        + ["entry_open", "exit_close", "target"]
    )

    if len(model_data) < 200:
        raise RuntimeError(
            f"Only {len(model_data)} usable rows remain. Use a longer date range."
        )

    return model_data


def load_event_exclusions(path: Path) -> pd.DataFrame:
    """Load and validate reproducible event windows from a CSV file."""
    columns = [
        "start_date",
        "end_date",
        "event_label",
        "event_reason",
        "source_url",
    ]
    if not path.exists():
        print(f"Event-exclusion file not found; no rows removed: {path}")
        return pd.DataFrame(columns=columns)

    exclusions = pd.read_csv(path)
    required = {"start_date", "end_date"}
    missing = required.difference(exclusions.columns)
    if missing:
        raise ValueError(
            f"Event-exclusion file is missing columns: {sorted(missing)}"
        )

    exclusions["start_date"] = pd.to_datetime(
        exclusions["start_date"], errors="raise"
    )
    exclusions["end_date"] = pd.to_datetime(
        exclusions["end_date"], errors="raise"
    )
    if (exclusions["end_date"] < exclusions["start_date"]).any():
        raise ValueError(
            "Each event exclusion must have end_date on or after start_date."
        )

    for column in columns[2:]:
        if column not in exclusions:
            exclusions[column] = ""
    return exclusions[columns].copy()


def apply_event_exclusions(
    data: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove rows exposed to an event on signal, entry, or exit date."""
    if exclusions.empty:
        return data.copy(), pd.DataFrame(), pd.DataFrame()

    remove_mask = pd.Series(False, index=data.index)
    reports: list[dict[str, object]] = []
    matched_rows: list[pd.DataFrame] = []

    for event_id, event in exclusions.reset_index(drop=True).iterrows():
        start = event["start_date"]
        end = event["end_date"]
        signal_hit = (data.index >= start) & (data.index <= end)
        entry_hit = data["entry_date"].between(start, end)
        exit_hit = data["exit_date"].between(start, end)
        event_mask = signal_hit | entry_hit | exit_hit
        remove_mask |= event_mask

        reports.append(
            {
                "event_id": event_id + 1,
                **event.to_dict(),
                "signal_date_matches": int(signal_hit.sum()),
                "entry_date_matches": int(entry_hit.sum()),
                "exit_date_matches": int(exit_hit.sum()),
                "unique_rows_matched": int(event_mask.sum()),
            }
        )

        if event_mask.any():
            detail = data.loc[
                event_mask,
                ["entry_date", "exit_date", "target"],
            ].copy()
            detail.insert(0, "event_id", event_id + 1)
            detail.insert(1, "event_label", event["event_label"])
            detail["signal_date_in_window"] = signal_hit[event_mask]
            detail["entry_date_in_window"] = entry_hit[event_mask]
            detail["exit_date_in_window"] = exit_hit[event_mask]
            matched_rows.append(detail)

    filtered = data.loc[~remove_mask].copy()
    details = (
        pd.concat(matched_rows).sort_index()
        if matched_rows
        else pd.DataFrame()
    )
    print(
        f"Applied {len(exclusions)} event windows: removed "
        f"{int(remove_mask.sum())} of {len(data)} model rows."
    )
    return filtered, pd.DataFrame(reports), details


def audit_model_features(
    pretest: pd.DataFrame,
    feature_columns: list[str] | None = None,
    variance_tolerance: float = 1e-14,
) -> tuple[list[str], pd.DataFrame]:
    """Exclude unavailable or constant features without using test data."""
    active_features: list[str] = []
    rows: list[dict[str, object]] = []

    audited_features = feature_columns or FEATURES
    for feature in audited_features:
        values = pd.to_numeric(pretest[feature], errors="coerce")
        available = values.dropna()
        unique_values = int(available.nunique())
        standard_deviation = (
            float(available.std(ddof=0)) if not available.empty else np.nan
        )
        included = bool(
            len(available) > 0
            and unique_values > 1
            and np.isfinite(standard_deviation)
            and standard_deviation > variance_tolerance
        )
        reason = "active"
        if available.empty:
            reason = "unavailable"
        elif unique_values <= 1 or standard_deviation <= variance_tolerance:
            reason = "constant_or_zero_variance"
        if included:
            active_features.append(feature)

        rows.append(
            {
                "feature": feature,
                "pretest_observations": len(available),
                "pretest_unique_values": unique_values,
                "pretest_standard_deviation": standard_deviation,
                "included": included,
                "reason": reason,
            }
        )

    missing_base_features = set(BASE_FEATURES).difference(active_features)
    if missing_base_features:
        raise RuntimeError(
            "Required Au99.99 features are unavailable or constant: "
            f"{sorted(missing_base_features)}"
        )
    if not active_features:
        raise RuntimeError("No usable model features remain after auditing.")

    excluded = [row["feature"] for row in rows if not row["included"]]
    if excluded:
        print(
            "Excluded unavailable/constant model features: "
            + ", ".join(excluded)
        )
    return active_features, pd.DataFrame(rows)


def chronological_split(
    data: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_pool = data.loc[
        data.index < VALIDATION_START
    ]

    validation_pool = data.loc[
        (data.index >= VALIDATION_START)
        & (data.index < TEST_START)
    ]

    # Always purge seven rows so every horizon receives the same treatment.
    train = train_pool.iloc[:-MAX_HORIZON].copy()
    validation = validation_pool.iloc[:-MAX_HORIZON].copy()

    pretest = data.loc[
        data.index < TEST_START
    ].iloc[:-MAX_HORIZON].copy()

    test = data.loc[
        (data.index >= TEST_START)
        & (data.index <= COMMON_TEST_END)
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("One chronological split is empty.")

    print("Fixed chronological date ranges:")
    for name, frame in {
        "Train": train,
        "Validation": validation,
        "Test": test,
    }.items():
        print(
            f"  {name:<10} {len(frame):>4} rows | "
            f"{frame.index.min().date()} to "
            f"{frame.index.max().date()}"
        )

    return train, validation, pretest, test


def build_pipeline(estimator: object) -> Pipeline:
    """Combine missing-value handling, scaling, and a model."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


class BlockBaggedRegressor:
    """Average models fitted to moving-block bootstrap samples."""

    def __init__(
        self,
        base_model: Pipeline,
        n_estimators: int = 100,
        block_size: int = 20,
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.block_size = block_size
        self.random_state = random_state
        self.estimators_: list[Pipeline] = []

    def _bootstrap_indices(
        self,
        n_rows: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.block_size < 1:
            raise ValueError("Bagging block size must be at least one.")
        if self.block_size > n_rows:
            raise ValueError(
                "Bagging block size cannot exceed the training-set length."
            )

        sampled: list[int] = []
        maximum_start = n_rows - self.block_size

        while len(sampled) < n_rows:
            start = int(rng.integers(0, maximum_start + 1))
            sampled.extend(range(start, start + self.block_size))

        return np.asarray(sampled[:n_rows])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlockBaggedRegressor":
        if self.n_estimators < 1:
            raise ValueError("Bagging requires at least one estimator.")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            indices = self._bootstrap_indices(len(X), rng)
            estimator = clone(self.base_model)
            estimator.fit(X.iloc[indices], y.iloc[indices])
            self.estimators_.append(estimator)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("The bagged model has not been fitted.")

        individual_predictions = self.predict_members(X)
        return individual_predictions.mean(axis=1)

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        """Return one prediction column per bootstrap estimator."""
        if not self.estimators_:
            raise RuntimeError("The bagged model has not been fitted.")
        return np.column_stack(
            [estimator.predict(X) for estimator in self.estimators_]
        )

    def mean_coefficients(self) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("The bagged model has not been fitted.")

        coefficients = [
            np.asarray(estimator.named_steps["model"].coef_).ravel()
            for estimator in self.estimators_
        ]
        return np.mean(np.vstack(coefficients), axis=0)


def regression_candidates() -> dict[str, list[Pipeline]]:
    return {
        "OLS": [build_pipeline(LinearRegression())],
        "Ridge": [
            build_pipeline(Ridge(alpha=alpha))
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        ],
        "Lasso": [
            build_pipeline(Lasso(alpha=alpha, max_iter=50_000))
            for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ],
        "ElasticNet": [
            build_pipeline(
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000)
            )
            for alpha in [1e-6, 1e-5, 1e-4, 1e-3]
            for l1_ratio in [0.1, 0.5, 0.9]
        ],
        "Huber": [
            build_pipeline(
                HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=5_000)
            )
            for epsilon in [1.1, 1.35, 1.5]
            for alpha in [0.0001, 0.001, 0.01]
        ],
    }


def select_regression_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Tune models on the common validation dates."""
    fitted: dict[str, Pipeline] = {}
    selections: list[dict[str, object]] = []
    validation_scoring = validation

    for model_name, candidates in regression_candidates().items():
        best_model: Pipeline | None = None
        best_mse = np.inf
        best_prediction: np.ndarray | None = None
        for candidate in candidates:
            candidate.fit(train[feature_columns], train["target"])
            prediction = candidate.predict(
                validation_scoring[feature_columns]
            )
            mse = mean_squared_error(
                validation_scoring["target"], prediction
            )

            if mse < best_mse:
                best_mse = mse
                best_model = clone(candidate)
                best_prediction = np.asarray(prediction).copy()

        if best_model is None or best_prediction is None:
            raise RuntimeError(f"No valid candidate found for {model_name}.")

        zero_mse = mean_squared_error(
            validation_scoring["target"],
            np.zeros(len(validation_scoring)),
        )

        relative_mse_zero = best_mse / zero_mse

        correlation = safe_correlation(
            validation_scoring["target"],
            best_prediction,
        )

        direction_accuracy = np.mean(
            np.sign(best_prediction)
            == np.sign(validation_scoring["target"])
        )
        actual_values = validation_scoring["target"].to_numpy()
        validation_mae = mean_absolute_error(
            actual_values, best_prediction
        )
        validation_magnitude_mae = mean_absolute_error(
            np.abs(actual_values), np.abs(best_prediction)
        )
        validation_magnitude_correlation = safe_correlation(
            np.abs(actual_values), np.abs(best_prediction)
        )
        actual_std = float(np.std(actual_values))
        validation_amplitude_ratio = (
            float(np.std(best_prediction)) / actual_std
            if actual_std > 0
            else np.nan
        )
        best_model.fit(pretest[feature_columns], pretest["target"])
        fitted[model_name] = best_model
        selections.append(
            {
                "horizon": horizon,
                "model": model_name,
                "validation_mse": best_mse,
                "zero_forecast_mse": zero_mse,
                "relative_mse_zero": relative_mse_zero,
                "validation_correlation": correlation,
                "validation_direction_accuracy": direction_accuracy,
                "validation_mae": validation_mae,
                "validation_magnitude_mae": validation_magnitude_mae,
                "validation_magnitude_correlation": (
                    validation_magnitude_correlation
                ),
                "validation_amplitude_ratio": validation_amplitude_ratio,
                "selected_parameters": str(
                    best_model.named_steps["model"].get_params()
                ),
            }
        )
        print(f"Selected {model_name}; validation MSE={best_mse:.8f}")

    return fitted, pd.DataFrame(selections)


def select_statistical_feature_method(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    horizon: int,
    active_features: list[str],
    requested_method: str,
    alpha: float,
) -> tuple[
    list[str],
    dict[str, Pipeline],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Select a training-only subset using validation metrics, never test."""
    subsets, coefficient_tests, elimination_path = build_feature_subsets(
        train, active_features, alpha=alpha
    )
    methods = [requested_method]
    model_sets: dict[str, dict[str, Pipeline]] = {}
    selection_sets: dict[str, pd.DataFrame] = {}
    comparison_rows: list[pd.DataFrame] = []

    for method in methods:
        features = enforce_feature_hierarchy(subsets[method])
        subsets[method] = features
        models, selections = select_regression_models(
            train,
            validation,
            pretest,
            horizon=horizon,
            feature_columns=features,
        )
        selections = selections.copy()
        selections.insert(1, "feature_selection_method", method)
        selections["selected_feature_count"] = len(features)
        selections["selected_features"] = ",".join(features)
        model_sets[method] = models
        selection_sets[method] = selections
        comparison_rows.append(selections)

    comparison = pd.concat(comparison_rows, ignore_index=True)
    selected_method = requested_method

    comparison["feature_method_selected"] = (
        comparison["feature_selection_method"] == selected_method
    )
    subset_rows = []
    for method, features in subsets.items():
        for order, feature in enumerate(features, start=1):
            subset_rows.append(
                {
                    "method": method,
                    "feature_order": order,
                    "feature": feature,
                    "selected_method": method == selected_method,
                }
            )
    subset_table = pd.DataFrame(subset_rows)
    print(
        f"Selected feature method={selected_method}; "
        f"features={len(subsets[selected_method])}"
    )
    return (
        subsets[selected_method],
        model_sets[selected_method],
        selection_sets[selected_method],
        comparison,
        coefficient_tests,
        pd.concat(
            [
                elimination_path,
                subset_table.assign(record_type="retained_feature"),
            ],
            ignore_index=True,
            sort=False,
        ),
    )


def apply_huber_bagging_if_eligible(
    models: dict[str, object],
    selections: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    n_estimators: int,
    block_size: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame, bool]:
    """Replace horizon-1 Huber only when block bagging passes validation."""
    if horizon != 1 or "Huber" not in models:
        return models, selections, False

    base_template = clone(models["Huber"])
    validation_bag = BlockBaggedRegressor(
        base_model=base_template,
        n_estimators=n_estimators,
        block_size=block_size,
        random_state=random_state,
    )
    validation_bag.fit(train[feature_columns], train["target"])
    prediction = validation_bag.predict(validation[feature_columns])
    mse = mean_squared_error(validation["target"], prediction)
    zero_mse = mean_squared_error(
        validation["target"], np.zeros(len(validation))
    )
    correlation = safe_correlation(validation["target"], prediction)
    direction = float(
        np.mean(np.sign(prediction) == np.sign(validation["target"]))
    )

    huber_mask = selections["model"] == "Huber"
    baseline = selections.loc[huber_mask].iloc[0]
    passes = bool(
        mse < baseline["validation_mse"]
        and correlation > baseline["validation_correlation"]
        and direction >= baseline["validation_direction_accuracy"]
    )
    if not passes:
        return models, selections, False

    final_bag = BlockBaggedRegressor(
        base_model=base_template,
        n_estimators=n_estimators,
        block_size=block_size,
        random_state=random_state,
    )
    final_bag.fit(pretest[feature_columns], pretest["target"])
    models = dict(models)
    models["Huber"] = final_bag
    selections = selections.copy()
    selections.loc[huber_mask, "validation_mse"] = mse
    selections.loc[huber_mask, "zero_forecast_mse"] = zero_mse
    selections.loc[huber_mask, "relative_mse_zero"] = mse / zero_mse
    selections.loc[huber_mask, "validation_correlation"] = correlation
    selections.loc[huber_mask, "validation_direction_accuracy"] = direction
    selections.loc[huber_mask, "validation_mae"] = mean_absolute_error(
        validation["target"], prediction
    )
    selections.loc[huber_mask, "validation_magnitude_mae"] = (
        mean_absolute_error(
            np.abs(validation["target"]), np.abs(prediction)
        )
    )
    selections.loc[huber_mask, "validation_magnitude_correlation"] = (
        safe_correlation(
            np.abs(validation["target"]), np.abs(prediction)
        )
    )
    actual_std = float(np.std(validation["target"]))
    selections.loc[huber_mask, "validation_amplitude_ratio"] = (
        float(np.std(prediction)) / actual_std if actual_std > 0 else np.nan
    )
    selections.loc[huber_mask, "selected_parameters"] = (
        selections.loc[huber_mask, "selected_parameters"].astype(str)
        + f"; block_bagging(n_estimators={n_estimators}, "
        f"block_size={block_size}, random_state={random_state})"
    )
    return models, selections, True


def coefficient_diagnostics(
    coefficients: pd.DataFrame,
    tolerance: float = 1e-14,
) -> pd.DataFrame:
    """Explain whether each fitted coefficient is nonzero or regularized."""
    long = coefficients.reset_index().melt(
        id_vars="feature",
        var_name="model",
        value_name="coefficient",
    )
    near_zero = long["coefficient"].abs() <= tolerance
    regularized_model = long["model"].isin(["Lasso", "ElasticNet"])
    long["status"] = "nonzero"
    long.loc[near_zero & regularized_model, "status"] = (
        "regularized_to_zero"
    )
    long.loc[near_zero & ~regularized_model, "status"] = "numerically_zero"
    return long.sort_values(["model", "feature"]).reset_index(drop=True)


def select_open_updated_huber(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    horizon: int,
) -> tuple[Pipeline, dict[str, object], list[str]]:
    """Tune a Huber forecast that is updated after the next open is known.

    This model is diagnostic with daily data. A live strategy cannot observe
    the official open and then assume execution at that same open; intraday
    data and a later execution price are required for a tradable backtest.
    """
    best_mse = np.inf
    best_model: Pipeline | None = None
    best_prediction: np.ndarray | None = None
    best_feature_set = ""
    best_features: list[str] = []

    for feature_set_name, feature_columns in OPEN_UPDATE_FEATURE_SETS.items():
        for candidate in regression_candidates()["Huber"]:
            candidate.fit(train[feature_columns], train["target"])
            prediction = candidate.predict(validation[feature_columns])
            mse = mean_squared_error(validation["target"], prediction)
            if mse < best_mse:
                best_mse = mse
                best_model = clone(candidate)
                best_prediction = np.asarray(prediction).copy()
                best_feature_set = feature_set_name
                best_features = feature_columns

    if best_model is None or best_prediction is None:
        raise RuntimeError("No valid OpenUpdatedHuber candidate found.")

    row = validation_metric_row(
        model_name="OpenUpdatedHuber",
        horizon=horizon,
        validation=validation,
        prediction=best_prediction,
        selected_parameters=(
            f"feature_set={best_feature_set}; "
            f"{best_model.named_steps['model'].get_params()}"
        ),
    )
    best_model.fit(pretest[best_features], pretest["target"])
    return best_model, row, best_features


def volatility_weights(frame: pd.DataFrame) -> pd.Series:
    """Give less influence to unusually volatile observations."""
    weights = 1.0 / (frame["volatility_20d"].pow(2) + 1e-8)
    lower, upper = weights.quantile([0.01, 0.99])
    return weights.clip(lower=lower, upper=upper)


def fit_wls(pretest: pd.DataFrame) -> Pipeline:
    wls = build_pipeline(LinearRegression())
    wls.fit(
        pretest[FEATURES],
        pretest["target"],
        model__sample_weight=volatility_weights(pretest),
    )
    return wls


def tune_logistic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    horizon: int,
) -> tuple[Pipeline, float, float]:
    """Tune direction classification on non-overlapping validation rows."""
    best_model: Pipeline | None = None
    best_auc = -np.inf
    best_c = np.nan
    validation_scoring = validation

    for c_value in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidate = build_pipeline(
            LogisticRegression(C=c_value, max_iter=5_000)
        )
        candidate.fit(train[FEATURES], train["target_up"])
        probability = candidate.predict_proba(
            validation_scoring[FEATURES]
        )[:, 1]
        auc = roc_auc_score(
            validation_scoring["target_up"], probability
        )

        if auc > best_auc:
            best_auc = auc
            best_c = c_value
            best_model = clone(candidate)

    if best_model is None:
        raise RuntimeError("No valid Logistic Regression candidate found.")

    best_model.fit(pretest[FEATURES], pretest["target_up"])
    return best_model, best_c, best_auc


def safe_correlation(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    if np.std(actual_array) == 0 or np.std(predicted_array) == 0:
        return np.nan
    return float(np.corrcoef(actual_array, predicted_array)[0, 1])


def validation_metric_row(
    model_name: str,
    horizon: int,
    validation: pd.DataFrame,
    prediction: np.ndarray,
    selected_parameters: str,
) -> dict[str, object]:
    """Build comparable validation metrics for an additional model."""
    actual = validation["target"].to_numpy()
    predicted = np.asarray(prediction)
    validation_mse = mean_squared_error(actual, predicted)
    zero_mse = mean_squared_error(actual, np.zeros(len(actual)))
    return {
        "horizon": horizon,
        "model": model_name,
        "validation_mse": validation_mse,
        "zero_forecast_mse": zero_mse,
        "relative_mse_zero": validation_mse / zero_mse,
        "validation_correlation": safe_correlation(actual, predicted),
        "validation_direction_accuracy": float(
            np.mean(np.sign(predicted) == np.sign(actual))
        ),
        "validation_mae": mean_absolute_error(actual, predicted),
        "validation_magnitude_mae": mean_absolute_error(
            np.abs(actual), np.abs(predicted)
        ),
        "validation_magnitude_correlation": safe_correlation(
            np.abs(actual), np.abs(predicted)
        ),
        "validation_amplitude_ratio": (
            float(np.std(predicted)) / float(np.std(actual))
            if float(np.std(actual)) > 0
            else np.nan
        ),
        "selected_parameters": selected_parameters,
    }


def evaluate_models(
    models: dict[str, object],
    test: pd.DataFrame,
    horizon: int,
    model_features: dict[str, list[str]] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Predict and score every common test date."""
    predictions = pd.DataFrame(
        {
            "entry_date": test["entry_date"],
            "exit_date": test["exit_date"],
            "entry_previous_close": test["entry_previous_close"],
            "entry_open": test["entry_open"],
            "entry_open_jump": test["entry_open_jump"],
            "exit_close": test["exit_close"],
            "actual_return": test["target"],
        },
        index=test.index,
    )
    evaluation_index = test.index
    predictions["used_for_evaluation"] = predictions.index.isin(
        evaluation_index
    )
    metric_rows: list[dict[str, float | str]] = []
    coefficient_columns: dict[str, pd.Series] = {}
    feature_map = model_features or {}

    def append_metrics(model_name: str, prediction: np.ndarray) -> None:
        actual = test.loc[evaluation_index, "target"]
        actual_exit_price = test.loc[evaluation_index, "exit_close"]
        entry_open = test.loc[evaluation_index, "entry_open"]
        scored_prediction = pd.Series(
            prediction, index=test.index
        ).loc[evaluation_index]
        predicted_exit_price = entry_open * np.exp(scored_prediction)
        metric_rows.append(
            {
                "model": model_name,
                "evaluation_rows": len(evaluation_index),
                "mse": mean_squared_error(actual, scored_prediction),
                "mae": mean_absolute_error(actual, scored_prediction),
                "return_bias": float(
                    np.mean(scored_prediction.to_numpy() - actual.to_numpy())
                ),
                "magnitude_mae": mean_absolute_error(
                    np.abs(actual), np.abs(scored_prediction)
                ),
                "magnitude_correlation": safe_correlation(
                    np.abs(actual), np.abs(scored_prediction)
                ),
                "amplitude_ratio": (
                    float(np.std(scored_prediction)) / float(np.std(actual))
                    if float(np.std(actual)) > 0
                    else np.nan
                ),
                "r2": r2_score(actual, scored_prediction),
                "correlation": safe_correlation(
                    actual, scored_prediction
                ),
                "direction_accuracy": float(
                    np.mean(
                        np.sign(scored_prediction)
                        == np.sign(actual)
                    )
                ),
                "prediction_std": float(np.std(scored_prediction)),
                "price_mae_cny_per_gram": mean_absolute_error(
                    actual_exit_price,
                    predicted_exit_price,
                ),
            }
        )

    for model_name, model in models.items():
        feature_columns = feature_map.get(model_name, FEATURES)
        prediction = model.predict(test[feature_columns])
        predictions[model_name] = prediction
        append_metrics(model_name, prediction)

        if isinstance(model, Pipeline):
            estimator = model.named_steps["model"]
            if hasattr(estimator, "coef_"):
                coefficient_columns[model_name] = pd.Series(
                    np.asarray(estimator.coef_).ravel(),
                    index=feature_columns,
                )
        elif isinstance(model, BlockBaggedRegressor):
            coefficient_columns[model_name] = pd.Series(
                model.mean_coefficients(), index=feature_columns
            )

    regression_metrics = (
        pd.DataFrame(metric_rows).sort_values("mse").reset_index(drop=True)
    )
    coefficients = pd.DataFrame(coefficient_columns)
    coefficients.index.name = "feature"
    evaluation_predictions = predictions.loc[evaluation_index].copy()
    return (
        predictions,
        evaluation_predictions,
        regression_metrics,
        coefficients,
    )


def residual_diagnostics(
    evaluation_predictions: pd.DataFrame,
    model_names: Iterable[str],
) -> pd.DataFrame:
    """Measure distribution shape and dependence of forecast residuals."""
    rows: list[dict[str, float | str]] = []

    for model_name in model_names:
        residual = (
            evaluation_predictions["actual_return"]
            - evaluation_predictions[model_name]
        ).dropna()
        jarque_bera = stats.jarque_bera(residual)
        shapiro = stats.shapiro(residual)

        rows.append(
            {
                "model": model_name,
                "observations": len(residual),
                "mean": residual.mean(),
                "std": residual.std(ddof=1),
                "skew": stats.skew(residual, bias=False),
                "excess_kurtosis": stats.kurtosis(
                    residual, fisher=True, bias=False
                ),
                "jarque_bera_p": jarque_bera.pvalue,
                "shapiro_p": shapiro.pvalue,
                "lag1_autocorrelation": residual.autocorr(lag=1),
                "max_absolute_residual": residual.abs().max(),
            }
        )

    return pd.DataFrame(rows)


def open_jump_diagnostics(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Identify large next-open jumps that can dominate executable returns."""
    diagnostics = predictions[
        [
            "entry_date",
            "entry_previous_close",
            "entry_open",
            "entry_open_jump",
            "exit_date",
            "exit_close",
            "actual_return",
        ]
    ].copy()
    diagnostics["absolute_entry_open_jump"] = diagnostics[
        "entry_open_jump"
    ].abs()
    diagnostics["large_open_jump_threshold"] = threshold
    diagnostics["is_large_open_jump"] = (
        diagnostics["absolute_entry_open_jump"] > threshold
    )
    diagnostics["same_day_open_to_close_return"] = np.log(
        diagnostics["exit_close"] / diagnostics["entry_open"]
    )
    return diagnostics.sort_values(
        "absolute_entry_open_jump", ascending=False
    )


def strategy_backtest(
    evaluation_predictions: pd.DataFrame,
    model_names: Iterable[str],
    round_trip_cost: float = 0.001,
) -> pd.DataFrame:
    """Run non-overlapping, next-open executable long/short trades."""
    rows: list[dict[str, float | str]] = []
    actual = evaluation_predictions["actual_return"].to_numpy()

    for model_name in model_names:
        signal = np.sign(
            evaluation_predictions[model_name].to_numpy()
        )
        active_trade = np.abs(signal)
        gross_log_return = signal * actual
        net_log_return = (
            gross_log_return - round_trip_cost * active_trade
        )
        net_curve = np.exp(np.cumsum(net_log_return))
        peak = np.maximum.accumulate(net_curve)

        rows.append(
            {
                "model": model_name,
                "trades": len(actual),
                "long_trades": int(np.sum(signal > 0)),
                "short_trades": int(np.sum(signal < 0)),
                "flat_periods": int(np.sum(signal == 0)),
                "gross_return": np.exp(gross_log_return.sum()) - 1,
                "net_return": net_curve[-1] - 1,
                "net_profit_per_100k": 100_000 * (net_curve[-1] - 1),
                "net_win_rate": np.mean(
                    np.exp(net_log_return) - 1 > 0
                ),
                "max_drawdown": np.min(net_curve / peak - 1),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "net_return", ascending=False
    )


def implied_price_predictions(
    evaluation_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Convert predicted executable returns into implied exit prices."""
    excluded_columns = {
        "entry_date",
        "exit_date",
        "entry_previous_close",
        "entry_open",
        "entry_open_jump",
        "exit_close",
        "actual_return",
        "actual_up",
        "used_for_evaluation",
        "Logistic_up_probability",
    }
    model_columns = [
        column
        for column in evaluation_predictions.columns
        if column not in excluded_columns
    ]

    prices = pd.DataFrame(
        {
            "entry_open": evaluation_predictions["entry_open"],
            "actual_exit_price": evaluation_predictions["exit_close"],
        },
        index=evaluation_predictions.index,
    )
    for model_name in model_columns:
        prices[model_name] = (
            evaluation_predictions["entry_open"]
            * np.exp(evaluation_predictions[model_name])
        )
    return prices


def save_charts(
    evaluation_predictions: pd.DataFrame,
    diagnostic_predictions: pd.DataFrame,
    price_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    event_exclusions_applied: bool = False,
) -> None:
    diagnostic_suffix = (
        " - event-filtered diagnostic"
        if event_exclusions_applied
        else ""
    )
    plot_columns = [
        column
        for column in evaluation_predictions.columns
        if column
        not in {
            "entry_date",
            "exit_date",
            "entry_previous_close",
            "entry_open",
            "entry_open_jump",
            "exit_close",
            "actual_up",
            "used_for_evaluation",
            "Logistic_up_probability",
        }
    ]

    ax = evaluation_predictions[plot_columns].plot(
        figsize=(13, 7),
        title=(
            "Common-date executable gold returns and predictions "
            f"(next-open entry){diagnostic_suffix}"
        ),
        alpha=0.8,
    )
    ax.set_ylabel("Log return")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "prediction_comparison.png", dpi=160)
    plt.close(ax.figure)

    price_plot_columns = [
        column
        for column in price_predictions.columns
        if column != "entry_open"
    ]
    ax = price_predictions[price_plot_columns].plot(
        figsize=(13, 7),
        title=(
            "Actual and predicted Au99.99 exit prices "
            f"(next-open entry){diagnostic_suffix}"
        ),
        alpha=0.82,
    )
    ax.set_ylabel("CNY per gram")
    ax.set_xlabel("trade_date")
    ax.figure.tight_layout()
    ax.figure.savefig(
        output_dir / "gold_price_prediction_comparison.png",
        dpi=160,
    )
    plt.close(ax.figure)

    ax = metrics.sort_values("mse").plot.bar(
        x="model",
        y="mse",
        legend=False,
        figsize=(10, 6),
        title=f"Out-of-sample mean squared error{diagnostic_suffix}",
    )
    ax.set_ylabel("MSE (lower is better)")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "model_mse.png", dpi=160)
    plt.close(ax.figure)

    diagnostic_models = [
        model
        for model in ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]
        if model in evaluation_predictions.columns
    ]
    figure, axes = plt.subplots(
        len(diagnostic_models),
        2,
        figsize=(13, 4.5 * len(diagnostic_models)),
        squeeze=False,
    )

    for row, model_name in enumerate(diagnostic_models):
        residual = (
            diagnostic_predictions["actual_return"]
            - diagnostic_predictions[model_name]
        )
        axes[row, 0].hist(
            residual,
            bins=20,
            density=True,
            alpha=0.7,
            label="Residuals",
        )
        x_values = np.linspace(residual.min(), residual.max(), 300)
        axes[row, 0].plot(
            x_values,
            stats.norm.pdf(
                x_values,
                residual.mean(),
                residual.std(ddof=1),
            ),
            linewidth=2,
            label="Fitted normal",
        )
        axes[row, 0].set_title(
            f"{model_name}: residual distribution{diagnostic_suffix}"
        )
        axes[row, 0].legend()
        stats.probplot(residual, dist="norm", plot=axes[row, 1])
        axes[row, 1].set_title(
            f"{model_name}: normal Q-Q plot{diagnostic_suffix}"
        )

    figure.tight_layout()
    figure.savefig(output_dir / "residual_normality.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = download_gold_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    external_factors = None
    if not args.skip_external_factors:
        external_factors = build_tushare_factor_features(
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
        if not external_factors.empty:
            availability_rows = []
            for column in external_factors.columns:
                available = external_factors[column].dropna()
                availability_rows.append(
                    {
                        "factor": column,
                        "observations": len(available),
                        "first_date": (
                            available.index.min()
                            if not available.empty
                            else pd.NaT
                        ),
                        "last_date": (
                            available.index.max()
                            if not available.empty
                            else pd.NaT
                        ),
                    }
                )
            pd.DataFrame(availability_rows).to_csv(
                args.output_dir / "factor_availability.csv",
                index=False,
            )

    data = build_dataset(
        raw,
        horizon=args.horizon,
        external_factors=external_factors,
    )
    exclusion_report = pd.DataFrame()
    excluded_rows = pd.DataFrame()
    if not args.ignore_event_exclusions:
        event_exclusions = load_event_exclusions(args.event_exclusion_file)
        data, exclusion_report, excluded_rows = apply_event_exclusions(
            data,
            event_exclusions,
        )
        if not exclusion_report.empty:
            exclusion_report.to_csv(
                args.output_dir / "event_exclusion_report.csv",
                index=False,
            )
        if not excluded_rows.empty:
            excluded_rows.to_csv(
                args.output_dir / "event_excluded_model_rows.csv"
            )

    train, validation, pretest, test = chronological_split(
        data, horizon=args.horizon
    )
    regime_thresholds = fit_regime_thresholds(train)
    transformed_frames = {
        "train": transform_interactions_and_regimes(
            train, regime_thresholds
        ),
        "validation": transform_interactions_and_regimes(
            validation, regime_thresholds
        ),
        "pretest": transform_interactions_and_regimes(
            pretest, regime_thresholds
        ),
        "test": transform_interactions_and_regimes(
            test, regime_thresholds
        ),
    }
    train = transformed_frames["train"]
    validation = transformed_frames["validation"]
    pretest = transformed_frames["pretest"]
    test = transformed_frames["test"]

    # Validation retained ordinary interactions but rejected the categorical
    # regime block. Regime labels remain exported as diagnostics.
    combined_feature_pool = feature_blocks(FEATURES)["interactions"]
    # Availability checks and statistical feature selection use training data
    # only. Validation chooses among precomputed subsets; test is untouched.
    active_features, feature_audit = audit_model_features(
        train, feature_columns=combined_feature_pool
    )
    feature_audit.to_csv(
        args.output_dir / "model_feature_audit.csv",
        index=False,
    )

    (
        selected_features,
        models,
        selections,
        feature_selection_comparison,
        coefficient_significance,
        feature_selection_audit,
    ) = select_statistical_feature_method(
        train,
        validation,
        pretest,
        horizon=args.horizon,
        active_features=active_features,
        requested_method=args.feature_selection,
        alpha=args.feature_selection_alpha,
    )
    bagging_applied = False
    if not args.disable_huber_bagging:
        models, selections, bagging_applied = apply_huber_bagging_if_eligible(
            models=models,
            selections=selections,
            train=train,
            validation=validation,
            pretest=pretest,
            feature_columns=selected_features,
            horizon=args.horizon,
            n_estimators=args.bagging_estimators,
            block_size=args.bagging_block_size,
            random_state=args.random_state,
        )
    model_feature_map = {
        model_name: selected_features for model_name in models
    }

    (
        predictions,
        evaluation_predictions,
        metrics,
        coefficients,
    ) = evaluate_models(
        models,
        test,
        horizon=args.horizon,
        model_features=model_feature_map,
    )
    regression_model_names = list(models)
    non_overlapping_predictions = evaluation_predictions.iloc[
        :: args.horizon
    ].copy()
    diagnostics = residual_diagnostics(
        non_overlapping_predictions,
        regression_model_names,
    )
    open_jump_report = open_jump_diagnostics(
        non_overlapping_predictions,
        threshold=args.large_open_jump_threshold,
    )
    backtest = strategy_backtest(
        non_overlapping_predictions,
        regression_model_names,
    )
    price_predictions = implied_price_predictions(
        evaluation_predictions
    )
    selections["signal_timing"] = "previous_close"
    selections["daily_data_trading_eligible"] = True
    selections["event_exclusions_applied"] = not exclusion_report.empty
    selections["event_excluded_rows"] = excluded_rows.index.nunique()
    selections["event_exclusion_file"] = args.event_exclusion_file.as_posix()
    metrics["signal_timing"] = "previous_close"
    metrics["daily_data_trading_eligible"] = True
    metrics["event_exclusions_applied"] = not exclusion_report.empty
    metrics["event_excluded_rows"] = excluded_rows.index.nunique()
    metrics["event_exclusion_file"] = args.event_exclusion_file.as_posix()
    selections["active_feature_count"] = len(selected_features)
    selections["active_features"] = ",".join(selected_features)
    selections["feature_selection_alpha"] = args.feature_selection_alpha
    metrics["active_feature_count"] = len(selected_features)
    metrics["feature_selection_method"] = selections[
        "feature_selection_method"
    ].iloc[0]
    selections["interaction_regime_pool"] = "interactions"
    metrics["interaction_regime_pool"] = "interactions"
    selections["huber_block_bagging_applied"] = (
        selections["model"].eq("Huber") & bagging_applied
    )
    metrics["huber_block_bagging_applied"] = (
        metrics["model"].eq("Huber") & bagging_applied
    )

    coefficient_report = coefficient_diagnostics(coefficients)

    stale_logistic_output = args.output_dir / "gold_logistic_comparison.csv"
    if stale_logistic_output.exists():
        stale_logistic_output.unlink()

    predictions.to_csv(args.output_dir / "gold_model_predictions.csv")
    evaluation_predictions.to_csv(
        args.output_dir / "gold_common_date_predictions.csv"
    )
    non_overlapping_predictions.to_csv(
        args.output_dir / "gold_non_overlapping_predictions.csv"
    )
    price_predictions.to_csv(
        args.output_dir / "gold_price_predictions.csv"
    )
    metrics.to_csv(args.output_dir / "gold_model_comparison.csv", index=False)
    coefficients.to_csv(args.output_dir / "gold_model_coefficients.csv")
    coefficient_report.to_csv(
        args.output_dir / "coefficient_diagnostics.csv",
        index=False,
    )
    selections.to_csv(args.output_dir / "selected_models.csv", index=False)
    feature_selection_comparison.to_csv(
        args.output_dir / "feature_selection_validation_comparison.csv",
        index=False,
    )
    coefficient_significance.to_csv(
        args.output_dir / "ols_training_coefficient_tests.csv", index=False
    )
    feature_selection_audit.to_csv(
        args.output_dir / "feature_selection_audit.csv", index=False
    )
    market_state_counts(transformed_frames).to_csv(
        args.output_dir / "market_state_counts.csv", index=False
    )
    dated_state_frames = []
    for sample in ["train", "validation", "test"]:
        state_frame = transformed_frames[sample][
            ["market_state", *REGIME_FEATURES]
        ].copy()
        state_frame.insert(0, "sample", sample)
        dated_state_frames.append(state_frame)
    pd.concat(dated_state_frames).reset_index(names="trade_date").to_csv(
        args.output_dir / "market_state_labels.csv", index=False
    )
    with (args.output_dir / "regime_thresholds.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(threshold_record(regime_thresholds), file, indent=2)
    diagnostics.to_csv(
        args.output_dir / "residual_diagnostics.csv", index=False
    )
    open_jump_report.to_csv(
        args.output_dir / "open_jump_diagnostics.csv"
    )
    backtest.to_csv(
        args.output_dir / "gold_strategy_backtest.csv", index=False
    )
    save_charts(
        evaluation_predictions,
        non_overlapping_predictions,
        price_predictions,
        metrics,
        diagnostics,
        args.output_dir,
        event_exclusions_applied=not exclusion_report.empty,
    )

    print("\nRegression test results:")
    print(metrics.round(6).to_string(index=False))
    print(f"\nFiles saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
