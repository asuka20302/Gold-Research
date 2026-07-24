"""Load international XAU/USD daily prices without mixing Chinese gold bars.

The preferred source is Tushare's ``fx_daily`` endpoint for ``XAUUSD.FXCM``.
That endpoint supplies bid and ask OHLC bars in GMT.  When the current Tushare
token cannot refresh the endpoint, this module can recover the already cached
FXCM mid-close series that the old Shanghai-gold pipeline stored as an external
factor.  The recovery path is close-only and is therefore suitable for price
forecasting, but not for an executable open-to-close P&L backtest.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd
import tushare as ts


FXCM_SYMBOL = "XAUUSD.FXCM"
FULL_OHLC_COLUMNS = [
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]


def _as_dates(values: pd.Series) -> pd.Series:
    """Parse compact or ISO date text without depending on locale settings."""
    text = values.astype(str).str.strip()
    compact = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    unresolved = compact.isna()
    if unresolved.any():
        compact.loc[unresolved] = pd.to_datetime(
            text.loc[unresolved], errors="coerce"
        )
    return compact


def normalize_fxcm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validated FXCM mid prices, preserving executable bid/ask fields."""
    if frame is None or frame.empty or "trade_date" not in frame:
        return pd.DataFrame()

    data = frame.copy()
    data["trade_date"] = _as_dates(data["trade_date"])
    available_ohlc = [column for column in FULL_OHLC_COLUMNS if column in data]
    for column in available_ohlc:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    has_full_ohlc = set(FULL_OHLC_COLUMNS).issubset(data.columns)
    if has_full_ohlc:
        for field in ["open", "high", "low", "close"]:
            data[f"mid_{field}"] = (
                data[f"bid_{field}"] + data[f"ask_{field}"]
            ) / 2.0
    elif "mid_close" in data:
        data["mid_close"] = pd.to_numeric(data["mid_close"], errors="coerce")
    elif {"bid_close", "ask_close"}.issubset(data.columns):
        data["mid_close"] = (
            data["bid_close"] + data["ask_close"]
        ) / 2.0
    elif "close" in data:
        data["mid_close"] = pd.to_numeric(data["close"], errors="coerce")
    else:
        return pd.DataFrame()

    required = ["trade_date", "mid_close"]
    data = data.dropna(subset=required)
    data = (
        data.sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .reset_index(drop=True)
    )
    price_columns = [column for column in data if column.startswith("mid_")]
    if (data[price_columns] <= 0).any().any():
        raise ValueError("XAU/USD prices must all be positive.")

    if has_full_ohlc:
        invalid_spread = (
            (data["bid_open"] > data["ask_open"])
            | (data["bid_close"] > data["ask_close"])
        )
        if invalid_spread.any():
            raise ValueError("FXCM bid prices exceed ask prices in the cache.")
        columns = [
            "trade_date",
            *FULL_OHLC_COLUMNS,
            "mid_open",
            "mid_high",
            "mid_low",
            "mid_close",
        ]
        if "tick_qty" in data:
            data["tick_qty"] = pd.to_numeric(
                data["tick_qty"], errors="coerce"
            )
            columns.append("tick_qty")
        return data[columns]

    return data[["trade_date", "mid_close"]]


def download_fxcm_ohlc(
    start_date: str,
    end_date: str,
    token: str,
) -> pd.DataFrame:
    """Download complete XAU/USD bid/ask bars in sub-1000-row chunks."""
    pro = ts.pro_api(token)
    first = pd.to_datetime(start_date, format="%Y%m%d")
    last = pd.to_datetime(end_date, format="%Y%m%d")
    cursor = first
    chunks: list[pd.DataFrame] = []

    while cursor <= last:
        chunk_end = min(
            cursor + pd.DateOffset(years=2) - pd.Timedelta(days=1),
            last,
        )
        print(f"  FXCM {cursor.date()} to {chunk_end.date()}")
        chunk = pro.fx_daily(
            ts_code=FXCM_SYMBOL,
            start_date=cursor.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
        )
        if chunk is not None and not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(
            "Tushare returned no XAU/USD observations. Check TUSHARE_TOKEN "
            "and the fx_daily permission (2,000 points are required)."
        )
    combined = pd.concat(chunks, ignore_index=True)
    normalized = normalize_fxcm_frame(combined)
    if normalized.empty or "mid_open" not in normalized:
        raise RuntimeError(
            "Tushare XAU/USD response did not contain complete bid/ask OHLC."
        )
    return normalized


def _factor_cache_candidates(cache_dir: Path) -> list[Path]:
    factor_dir = cache_dir / "factors"
    return sorted(
        factor_dir.glob("macro_factors_v3_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) + sorted(
        factor_dir.glob("macro_factors_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def recover_shifted_fxcm_close(
    factor_file: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Undo the old one-FX-session Shanghai alignment of cached XAU closes.

    The old factor builder applied ``series.shift(1)`` on the FXCM trading
    index.  If the stored non-null sequence is ``[raw_0, raw_1, ...]`` at
    dates ``[date_1, date_2, ...]``, pairing values from position 1 onward
    with dates through position -1 reconstructs ``raw_1`` at ``date_1``.
    Only the unavailable first and last endpoints are lost.
    """
    frame = pd.read_csv(
        factor_file,
        usecols=["trade_date", "xauusd_close_known"],
    )
    frame["trade_date"] = _as_dates(frame["trade_date"])
    frame["xauusd_close_known"] = pd.to_numeric(
        frame["xauusd_close_known"], errors="coerce"
    )
    stored = (
        frame.dropna()
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
    )
    if len(stored) < 3:
        raise RuntimeError(f"No usable cached XAU/USD closes in {factor_file}.")

    recovered = pd.DataFrame(
        {
            "trade_date": stored["trade_date"].iloc[:-1].to_numpy(),
            "mid_close": stored["xauusd_close_known"].iloc[1:].to_numpy(float),
        }
    )
    first = pd.to_datetime(start_date, format="%Y%m%d")
    last = pd.to_datetime(end_date, format="%Y%m%d")
    recovered = recovered[
        recovered["trade_date"].between(first, last)
    ].reset_index(drop=True)
    if len(recovered) < 200:
        raise RuntimeError(
            "The recovered international-gold cache has fewer than 200 rows."
        )
    return normalize_fxcm_frame(recovered)


def load_international_gold(
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data"),
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load full FXCM OHLC when available, otherwise a proven close-only cache."""
    destination = cache_dir / "international"
    destination.mkdir(parents=True, exist_ok=True)
    raw_cache = destination / f"XAUUSD_FXCM_{start_date}_{end_date}.csv"
    recovered_cache = (
        destination
        / f"XAUUSD_FXCM_close_recovered_{start_date}_{end_date}.csv"
    )

    if raw_cache.exists() and not refresh:
        data = normalize_fxcm_frame(pd.read_csv(raw_cache))
        if not data.empty:
            return data, {
                "provider": "tushare_fx_daily",
                "instrument": FXCM_SYMBOL,
                "quote": "USD per troy ounce",
                "timezone": "GMT source dates",
                "price_fields": "bid_ask_ohlc",
                "cache_file": str(raw_cache),
            }

    refresh_error = ""
    if refresh or not recovered_cache.exists():
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if token:
            try:
                print(f"Downloading international gold {FXCM_SYMBOL}...")
                data = download_fxcm_ohlc(start_date, end_date, token)
                data.to_csv(raw_cache, index=False)
                return data, {
                    "provider": "tushare_fx_daily",
                    "instrument": FXCM_SYMBOL,
                    "quote": "USD per troy ounce",
                    "timezone": "GMT source dates",
                    "price_fields": "bid_ask_ohlc",
                    "cache_file": str(raw_cache),
                }
            except Exception as exc:  # provider failure must keep provenance
                refresh_error = str(exc)
                warnings.warn(
                    "Fresh Tushare FXCM download failed; checking the "
                    f"verified local close cache instead. Reason: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if recovered_cache.exists() and not refresh:
        data = normalize_fxcm_frame(pd.read_csv(recovered_cache))
        if not data.empty:
            candidates = _factor_cache_candidates(cache_dir)
            return data, {
                "provider": "recovered_tushare_fxcm_factor_cache",
                "instrument": FXCM_SYMBOL,
                "quote": "USD per troy ounce",
                "timezone": "GMT source dates",
                "price_fields": "mid_close_only",
                "cache_file": str(recovered_cache),
                "source_factor_cache": (
                    str(candidates[0]) if candidates else ""
                ),
                "refresh_error": refresh_error,
            }

    for candidate in _factor_cache_candidates(cache_dir):
        try:
            data = recover_shifted_fxcm_close(
                candidate, start_date, end_date
            )
        except (ValueError, RuntimeError, pd.errors.ParserError):
            continue
        data.to_csv(recovered_cache, index=False)
        return data, {
            "provider": "recovered_tushare_fxcm_factor_cache",
            "instrument": FXCM_SYMBOL,
            "quote": "USD per troy ounce",
            "timezone": "GMT source dates",
            "price_fields": "mid_close_only",
            "cache_file": str(recovered_cache),
            "source_factor_cache": str(candidate),
            "refresh_error": refresh_error,
        }

    raise RuntimeError(
        "No international-gold target is available. Use a valid TUSHARE_TOKEN "
        "with fx_daily permission and run with --refresh."
    )
