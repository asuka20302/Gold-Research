"""Download and construct point-in-time external factors with fallbacks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import akshare as ak
import tushare as ts
import yfinance as yf


TROY_OUNCE_GRAMS = 31.1034768


def _discover_fx_codes(
    pro: object,
    expected_codes: list[str],
    catalogue_file: Path | None = None,
) -> dict[str, str]:
    """Match expected FXCM symbols against the account's current catalogue."""
    resolved = {code: code for code in expected_codes}

    try:
        basics = pro.fx_obasic(exchange="FXCM")
    except Exception as exc:
        print(f"  FXCM catalogue unavailable; using standard codes: {exc}")
        return resolved

    if basics is None or basics.empty or "ts_code" not in basics:
        return resolved

    if catalogue_file is not None:
        basics.to_csv(catalogue_file, index=False)

    searchable_columns = [
        column
        for column in ["ts_code", "symbol", "name", "fullname"]
        if column in basics
    ]
    searchable = basics[searchable_columns].fillna("").astype(str)
    combined_text = searchable.agg(" ".join, axis=1).str.upper()
    normalized_codes = basics["ts_code"].astype(str).str.upper()

    aliases = {
        "XAUUSD.FXCM": ["XAUUSD", "XAU", "GOLD", "黄金"],
        "USDCNH.FXCM": ["USDCNH", "USD/CNH", "CNH", "离岸人民币"],
        "USDOLLAR.FXCM": [
            "USDOLLAR",
            "DOLLAR INDEX",
            "美元指数",
        ],
        "USOIL.FXCM": ["USOIL", "WTI", "CRUDE", "原油"],
    }

    for expected in expected_codes:
        exact_matches = basics.loc[
            normalized_codes == expected.upper(),
            "ts_code",
        ]
        if not exact_matches.empty:
            resolved[expected] = str(exact_matches.iloc[0])
            continue

        search_terms = aliases.get(
            expected,
            [expected.split(".")[0].upper()],
        )
        match_mask = pd.Series(False, index=basics.index)
        for term in search_terms:
            match_mask |= combined_text.str.contains(
                term.upper(), regex=False
            )
        matches = basics.loc[match_mask, "ts_code"]
        if not matches.empty:
            resolved[expected] = str(matches.iloc[0])

    return resolved


def _fetch_in_chunks(
    fetch: Callable[[str, str], pd.DataFrame],
    start_date: str,
    end_date: str,
    date_column: str,
    chunk_years: int = 3,
) -> pd.DataFrame:
    first_date = pd.to_datetime(start_date, format="%Y%m%d")
    last_date = pd.to_datetime(end_date, format="%Y%m%d")
    cursor = first_date
    chunks: list[pd.DataFrame] = []

    while cursor <= last_date:
        chunk_end = min(
            cursor
            + pd.DateOffset(years=chunk_years)
            - pd.Timedelta(days=1),
            last_date,
        )
        frame = fetch(
            cursor.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        )
        if frame is not None and not frame.empty:
            chunks.append(frame)
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    if date_column in combined.columns:
        combined = combined.drop_duplicates(
            subset=[date_column], keep="last"
        )
    return combined


def _fetch_yahoo_daily(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download Yahoo closes and return Tushare-compatible columns."""
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end_exclusive = (
        pd.to_datetime(end_date, format="%Y%m%d")
        + pd.Timedelta(days=1)
    )
    frame = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        repair=True,
        progress=False,
        threads=False,
        multi_level_index=False,
        timeout=20,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if "Close" not in frame:
        return pd.DataFrame()

    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)

    return pd.DataFrame(
        {
            "trade_date": index.strftime("%Y%m%d"),
            "close": pd.to_numeric(
                frame["Close"], errors="coerce"
            ).to_numpy(),
        }
    ).dropna()


def _fetch_fred_daily(
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download a public FRED series without requiring an API key."""
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}"
    )
    frame = pd.read_csv(url)
    if frame.empty or "observation_date" not in frame or series_id not in frame:
        return pd.DataFrame()

    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    first_date = pd.to_datetime(start_date, format="%Y%m%d")
    last_date = pd.to_datetime(end_date, format="%Y%m%d")
    mask = dates.between(first_date, last_date)
    return pd.DataFrame(
        {
            "trade_date": dates.loc[mask].dt.strftime("%Y%m%d"),
            "close": values.loc[mask],
        }
    ).dropna()


def _normalize_futures_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tushare or AKShare SHFE futures fields."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    aliases = {
        "trade_date": ["trade_date", "date", "日期"],
        "close": ["close", "收盘价"],
        "settle": ["settle", "动态结算价"],
        "vol": ["vol", "volume", "成交量"],
        "oi": ["oi", "hold", "持仓量"],
    }
    selected: dict[str, pd.Series] = {}
    for target, candidates in aliases.items():
        source = next((name for name in candidates if name in frame), None)
        if source is None:
            if target in {"trade_date", "close"}:
                return pd.DataFrame()
            selected[target] = pd.Series(np.nan, index=frame.index)
        else:
            selected[target] = frame[source]

    data = pd.DataFrame(selected)
    data["trade_date"] = pd.to_datetime(
        data["trade_date"].astype(str), errors="coerce"
    )
    for column in ["close", "settle", "vol", "oi"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["trade_date", "close"])
    data["trade_date"] = data["trade_date"].dt.strftime("%Y%m%d")
    return data.sort_values("trade_date").drop_duplicates(
        "trade_date", keep="last"
    )


def _shfe_gold_futures_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct same-day-known features from the SHFE gold main contract."""
    data = _normalize_futures_frame(frame)
    if data.empty:
        return pd.DataFrame()

    data["trade_date"] = pd.to_datetime(
        data["trade_date"], format="%Y%m%d"
    )
    data = data.set_index("trade_date").sort_index()
    features = pd.DataFrame(index=data.index)
    features["shfe_gold_close_known"] = data["close"]
    features["shfe_gold_return_1d"] = np.log(data["close"]).diff()
    features["shfe_gold_settle_return_1d"] = np.log(
        data["settle"].where(data["settle"] > 0)
    ).diff()
    features["shfe_gold_volume_change_5d"] = np.log1p(
        data["vol"].clip(lower=0)
    ).diff(5)
    features["shfe_gold_oi_change_5d"] = np.log(
        data["oi"].where(data["oi"] > 0)
    ).diff(5)
    return features


def _fetch_akshare_daily(
    endpoint: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download an AKShare series and return Tushare-compatible columns."""
    frame = pd.DataFrame()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if endpoint == "futures":
                frame = ak.futures_global_hist_em(symbol=symbol)
            elif endpoint == "forex":
                frame = ak.forex_hist_em(symbol=symbol)
            elif endpoint == "index":
                frame = ak.index_global_hist_em(symbol=symbol)
            else:
                raise ValueError(
                    f"Unknown AKShare endpoint: {endpoint}"
                )
            break
        except (
            ConnectionError,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))

    if (frame is None or frame.empty) and last_error is not None:
        raise last_error

    if frame is None or frame.empty:
        return pd.DataFrame()

    date_column = "日期"
    close_candidates = ["最新价", "收盘", "收盘价"]
    close_column = next(
        (column for column in close_candidates if column in frame.columns),
        None,
    )
    if date_column not in frame or close_column is None:
        return pd.DataFrame()

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    closes = pd.to_numeric(frame[close_column], errors="coerce")
    first_date = pd.to_datetime(start_date, format="%Y%m%d")
    last_date = pd.to_datetime(end_date, format="%Y%m%d")
    mask = dates.between(first_date, last_date)

    return pd.DataFrame(
        {
            "trade_date": dates.loc[mask].dt.strftime("%Y%m%d"),
            "close": closes.loc[mask],
        }
    ).dropna()


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize close fields used by Tushare FXCM and fallback providers."""
    if frame is None or frame.empty or "trade_date" not in frame:
        return pd.DataFrame()

    data = frame.copy()
    if "close" not in data:
        if {"bid_close", "ask_close"}.issubset(data.columns):
            bid = pd.to_numeric(data["bid_close"], errors="coerce")
            ask = pd.to_numeric(data["ask_close"], errors="coerce")
            data["close"] = (bid + ask) / 2
        elif "bid_close" in data:
            data["close"] = pd.to_numeric(
                data["bid_close"], errors="coerce"
            )
        else:
            return pd.DataFrame()

    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["trade_date"] = pd.to_datetime(
        data["trade_date"].astype(str), errors="coerce"
    )
    data = data.dropna(subset=["trade_date", "close"])
    data["trade_date"] = data["trade_date"].dt.strftime("%Y%m%d")
    return data[["trade_date", "close"]]


def _needs_price_fallback(
    frame: pd.DataFrame,
    end_date: str,
    tolerance_days: int = 10,
) -> bool:
    normalized = _normalize_price_frame(frame)
    if normalized.empty:
        return True
    latest = pd.to_datetime(
        normalized["trade_date"].max(), format="%Y%m%d"
    )
    requested_end = pd.to_datetime(end_date, format="%Y%m%d")
    return latest < requested_end - pd.Timedelta(days=tolerance_days)


def _merge_price_frames(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        normalized
        for normalized in [
            _normalize_price_frame(primary),
            _normalize_price_frame(fallback),
        ]
        if not normalized.empty
    ]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
    )


def _price_features(
    frame: pd.DataFrame,
    prefix: str,
    lag_for_shanghai: bool = True,
) -> pd.DataFrame:
    frame = _normalize_price_frame(frame)
    if frame.empty:
        return pd.DataFrame()

    data = frame[["trade_date", "close"]].copy()
    data["trade_date"] = pd.to_datetime(
        data["trade_date"].astype(str), format="%Y%m%d"
    )
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = (
        data.dropna()
        .sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")
    )

    log_close = np.log(data["close"])
    features = pd.DataFrame(index=data.index)
    features[f"{prefix}_close_known"] = data["close"]
    features[f"{prefix}_return_1d"] = log_close.diff()
    features[f"{prefix}_momentum_5d"] = log_close.diff(5)
    features[f"{prefix}_momentum_20d"] = log_close.diff(20)
    features[f"{prefix}_volatility_20d"] = (
        log_close.diff().rolling(20).std()
    )

    # US and international closes occur after the Shanghai close on the same
    # calendar date. Shift them before aligning with SGE dates.
    if lag_for_shanghai:
        features = features.shift(1)

    return features


def _yield_features(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty or "date" not in frame or "y10" not in frame:
        return pd.DataFrame()

    data = frame[["date", "y10"]].copy()
    data["date"] = pd.to_datetime(
        data["date"].astype(str), format="%Y%m%d"
    )
    data["y10"] = pd.to_numeric(data["y10"], errors="coerce")
    data = (
        data.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )

    features = pd.DataFrame(index=data.index)
    features[f"{prefix}_yield_10y"] = data["y10"]
    features[f"{prefix}_yield_change_1d"] = data["y10"].diff()
    features[f"{prefix}_yield_change_5d"] = data["y10"].diff(5)
    return features.shift(1)


def _gold_etf_share_features(
    pro: object,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    basics = pro.etf_basic(list_status="L")
    if basics is None or basics.empty:
        return pd.DataFrame()

    names = basics["name"].fillna("").astype(str)
    gold_mask = names.str.contains("黄金", regex=False)
    equity_gold_mask = names.str.contains(
        "黄金股|黄金股票|金矿|矿业|有色",
        regex=True,
    )
    codes = (
        basics.loc[gold_mask & ~equity_gold_mask, "ts_code"]
        .dropna()
        .drop_duplicates()
        .head(12)
        .tolist()
    )

    share_series: list[pd.Series] = []
    for code in codes:
        try:
            frame = _fetch_in_chunks(
                lambda start, end, fund_code=code: pro.fund_share(
                    ts_code=fund_code,
                    start_date=start,
                    end_date=end,
                ),
                start_date,
                end_date,
                date_column="trade_date",
            )
        except Exception as exc:
            print(f"  ETF shares unavailable for {code}: {exc}")
            continue

        if frame.empty or "fd_share" not in frame:
            continue

        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"].astype(str), format="%Y%m%d"
        )
        values = pd.to_numeric(frame["fd_share"], errors="coerce")
        series = pd.Series(
            values.to_numpy(),
            index=frame["trade_date"],
            name=code,
        )
        share_series.append(series[~series.index.duplicated(keep="last")])

    if not share_series:
        return pd.DataFrame()

    total_shares = pd.concat(share_series, axis=1).sum(
        axis=1, min_count=1
    )
    total_shares = total_shares.sort_index()
    log_shares = np.log(total_shares.where(total_shares > 0))

    features = pd.DataFrame(index=total_shares.index)
    features["gold_etf_share_change_1d"] = log_shares.diff()
    features["gold_etf_share_change_5d"] = log_shares.diff(5)
    return features.shift(1)


def build_tushare_factor_features(
    start_date: str,
    end_date: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return lagged external factors indexed by their source dates."""
    factor_dir = cache_dir / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)
    # Version the cache because v1 could silently carry stale dollar data and
    # had no GVZ or SHFE gold-futures features.
    cache_file = factor_dir / f"macro_factors_v3_{start_date}_{end_date}.csv"

    if cache_file.exists() and not refresh:
        print(f"Loading cached external factors: {cache_file}")
        cached = pd.read_csv(cache_file, parse_dates=["trade_date"])
        return cached.set_index("trade_date")

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "TUSHARE_TOKEN is required to download external factors."
        )

    pro = ts.pro_api(token)
    feature_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    # Preserve already downloaded point-in-time series while upgrading the
    # cache. This is especially useful when a provider is temporarily rate
    # limited. Repaired series appended below take precedence by column/date.
    legacy_file = factor_dir / f"macro_factors_{start_date}_{end_date}.csv"
    legacy_columns: set[str] = set()
    if legacy_file.exists() and not refresh:
        legacy = pd.read_csv(legacy_file, parse_dates=["trade_date"])
        legacy = legacy.set_index("trade_date").sort_index()
        feature_frames.append(legacy)
        legacy_columns = set(legacy.columns)
        status_rows.append(
            {
                "source": "legacy_cache_seed",
                "code": legacy_file.name,
                "provider": "local_cache",
                "status": "ok",
                "rows": len(legacy),
                "message": "Preserved existing factors before overlaying repairs",
            }
        )

    def save_status() -> None:
        pd.DataFrame(status_rows).to_csv(
            factor_dir / "factor_source_status.csv",
            index=False,
        )

    standard_fx_codes = [
        "XAUUSD.FXCM",
        "USDCNH.FXCM",
        "USDOLLAR.FXCM",
        "USOIL.FXCM",
    ]
    resolved_fx_codes = _discover_fx_codes(
        pro,
        standard_fx_codes,
        catalogue_file=factor_dir / "fxcm_catalogue.csv",
    )
    print(f"Resolved FXCM codes: {resolved_fx_codes}")

    price_sources = [
        (
            resolved_fx_codes["XAUUSD.FXCM"],
            "GC=F",
            ("futures", "GC00Y"),
            "xauusd",
            "fx",
            None,
        ),
        (
            resolved_fx_codes["USDCNH.FXCM"],
            "CNH=X",
            ("forex", "USDCNH"),
            "usdcnh",
            "fx",
            None,
        ),
        (
            resolved_fx_codes["USDOLLAR.FXCM"],
            "DX-Y.NYB",
            ("index", "\u7f8e\u5143\u6307\u6570"),
            "dollar",
            "fx",
            "DTWEXBGS",
        ),
        (
            resolved_fx_codes["USOIL.FXCM"],
            "CL=F",
            ("futures", "CL00Y"),
            "oil",
            "fx",
            None,
        ),
        (
            "SPX",
            "^GSPC",
            ("index", "\u6807\u666e500"),
            "sp500",
            "index",
            None,
        ),
        (None, "^VIX", None, "vix", "yahoo", "VIXCLS"),
        (None, None, None, "gvz", "fred", "GVZCLS"),
    ]

    for (
        code,
        yahoo_ticker,
        akshare_source,
        prefix,
        source_type,
        fred_series,
    ) in price_sources:
        if (
            not refresh
            and legacy_columns
            and prefix not in {"dollar", "vix", "gvz"}
            and any(name.startswith(f"{prefix}_") for name in legacy_columns)
        ):
            print(f"Keeping cached {prefix} series.")
            continue
        raw = pd.DataFrame()
        provider_messages: list[str] = []
        try:
            source_label = code if code is not None else yahoo_ticker
            print(f"Downloading external factor {source_label}...")
            if source_type == "fx":
                raw = _fetch_in_chunks(
                    lambda start, end, symbol=code: pro.fx_daily(
                        ts_code=symbol,
                        start_date=start,
                        end_date=end,
                    ),
                    start_date,
                    end_date,
                    date_column="trade_date",
                )
            elif source_type == "index":
                raw = _fetch_in_chunks(
                    lambda start, end, symbol=code: pro.index_global(
                        ts_code=symbol,
                        start_date=start,
                        end_date=end,
                    ),
                    start_date,
                    end_date,
                    date_column="trade_date",
                )
            elif source_type == "fred" and fred_series:
                raw = _fetch_fred_daily(
                    fred_series, start_date, end_date
                )
        except Exception as exc:
            provider_messages.append(f"Tushare: {exc}")
            print(f"  Tushare unavailable for {code}: {exc}")

        raw = _normalize_price_frame(raw)
        providers = ["tushare"] if not raw.empty else []
        if not providers and source_type != "yahoo":
            provider_messages.append("Tushare: no usable observations")

        if _needs_price_fallback(raw, end_date) and yahoo_ticker:
            print(f"  Trying Yahoo Finance fallback {yahoo_ticker}...")
            try:
                yahoo_raw = _fetch_yahoo_daily(
                    yahoo_ticker,
                    start_date,
                    end_date,
                )
                if not yahoo_raw.empty:
                    raw = _merge_price_frames(raw, yahoo_raw)
                    providers.append("yahoo")
                else:
                    provider_messages.append(
                        "Yahoo: no observations returned"
                    )
            except Exception as exc:
                provider_messages.append(f"Yahoo: {exc}")
                print(f"  Yahoo fallback failed for {yahoo_ticker}: {exc}")

        # FRED is the authoritative repair for the broad dollar index and the
        # volatility indices. Replace, rather than splice, to avoid a level
        # discontinuity between unlike dollar-index definitions.
        if _needs_price_fallback(raw, end_date) and fred_series:
            print(f"  Trying FRED fallback {fred_series}...")
            try:
                fred_raw = _fetch_fred_daily(
                    fred_series, start_date, end_date
                )
                if not fred_raw.empty:
                    raw = _normalize_price_frame(fred_raw)
                    providers = ["fred"]
                else:
                    provider_messages.append(
                        "FRED: no observations returned"
                    )
            except Exception as exc:
                provider_messages.append(f"FRED: {exc}")
                print(f"  FRED fallback failed for {fred_series}: {exc}")

        if (
            _needs_price_fallback(raw, end_date)
            and akshare_source is not None
        ):
            endpoint, akshare_symbol = akshare_source
            print(
                "  Trying AKShare fallback "
                f"{endpoint}:{akshare_symbol}..."
            )
            try:
                akshare_raw = _fetch_akshare_daily(
                    endpoint,
                    akshare_symbol,
                    start_date,
                    end_date,
                )
                if not akshare_raw.empty:
                    raw = _merge_price_frames(raw, akshare_raw)
                    providers.append("akshare")
                else:
                    provider_messages.append(
                        "AKShare: no observations returned"
                    )
            except Exception as exc:
                provider_messages.append(f"AKShare: {exc}")
                print(
                    "  AKShare fallback failed for "
                    f"{akshare_symbol}: {exc}"
                )

        provider = "+".join(providers) if providers else "none"
        try:
            features = _price_features(raw, prefix)
            if not features.empty:
                feature_frames.append(features)
                status_rows.append(
                    {
                        "source": prefix,
                        "code": " | ".join(
                            item
                            for item in [
                                str(code) if code is not None else "",
                                yahoo_ticker,
                                fred_series,
                                (
                                    akshare_source[1]
                                    if akshare_source is not None
                                    else ""
                                ),
                            ]
                            if item
                        ),
                        "provider": provider,
                        "status": "ok",
                        "rows": len(raw),
                        "message": "; ".join(provider_messages),
                    }
                )
            else:
                print(
                    "  No observations returned for "
                    f"{code} or {yahoo_ticker}."
                )
                status_rows.append(
                    {
                        "source": prefix,
                        "code": f"{code} | {yahoo_ticker}",
                        "provider": "none",
                        "status": "empty",
                        "rows": 0,
                        "message": "; ".join(provider_messages)
                        or "No observations returned",
                    }
                )
        except Exception as exc:
            print(f"  Skipping {code} / {yahoo_ticker}: {exc}")
            status_rows.append(
                {
                    "source": prefix,
                    "code": f"{code} | {yahoo_ticker}",
                    "provider": provider,
                    "status": "error",
                    "rows": 0,
                    "message": str(exc),
                }
            )

    # Tushare is attempted first. AKShare supplies the same main-contract
    # fields when the account lacks the fut_daily permission.
    print("Downloading SHFE gold main-contract futures...")
    futures_raw = pd.DataFrame()
    futures_provider = "none"
    futures_messages: list[str] = []
    try:
        futures_raw = _fetch_in_chunks(
            lambda start, end: pro.fut_daily(
                ts_code="AU.SHF", start_date=start, end_date=end
            ),
            start_date,
            end_date,
            date_column="trade_date",
            chunk_years=1,
        )
        futures_raw = _normalize_futures_frame(futures_raw)
        if not futures_raw.empty:
            futures_provider = "tushare"
    except Exception as exc:
        futures_messages.append(f"Tushare: {exc}")

    if futures_raw.empty:
        try:
            start_text = pd.to_datetime(
                start_date, format="%Y%m%d"
            ).strftime("%Y%m%d")
            end_text = pd.to_datetime(
                end_date, format="%Y%m%d"
            ).strftime("%Y%m%d")
            futures_raw = ak.futures_main_sina(
                symbol="AU0", start_date=start_text, end_date=end_text
            )
            futures_raw = _normalize_futures_frame(futures_raw)
            if not futures_raw.empty:
                futures_provider = "akshare"
        except Exception as exc:
            futures_messages.append(f"AKShare: {exc}")

    futures_features = _shfe_gold_futures_features(futures_raw)
    if not futures_features.empty:
        feature_frames.append(futures_features)
        status_rows.append(
            {
                "source": "shfe_gold_futures",
                "code": "AU.SHF | AU0",
                "provider": futures_provider,
                "status": "ok",
                "rows": len(futures_raw),
                "message": "; ".join(futures_messages),
            }
        )
    else:
        status_rows.append(
            {
                "source": "shfe_gold_futures",
                "code": "AU.SHF | AU0",
                "provider": "none",
                "status": "empty",
                "rows": 0,
                "message": "; ".join(futures_messages)
                or "No observations returned",
            }
        )

    yield_sources = [
        (
            "real",
            lambda start, end: pro.us_trycr(
                start_date=start,
                end_date=end,
            ),
        ),
        (
            "nominal",
            lambda start, end: pro.us_tycr(
                start_date=start,
                end_date=end,
            ),
        ),
    ]

    for prefix, fetch in yield_sources:
        if (
            not refresh
            and f"{prefix}_yield_10y" in legacy_columns
        ):
            print(f"Keeping cached US {prefix} yields.")
            continue
        try:
            print(f"Downloading US {prefix} yields...")
            raw = _fetch_in_chunks(
                fetch,
                start_date,
                end_date,
                date_column="date",
            )
            features = _yield_features(raw, prefix)
            if not features.empty:
                feature_frames.append(features)
                status_rows.append(
                    {
                        "source": f"{prefix}_yield",
                        "code": (
                            "us_trycr"
                            if prefix == "real"
                            else "us_tycr"
                        ),
                        "status": "ok",
                        "rows": len(raw),
                        "message": "",
                    }
                )
            else:
                status_rows.append(
                    {
                        "source": f"{prefix}_yield",
                        "code": (
                            "us_trycr"
                            if prefix == "real"
                            else "us_tycr"
                        ),
                        "status": "empty",
                        "rows": 0,
                        "message": "No observations returned",
                    }
                )
        except Exception as exc:
            print(f"  Skipping US {prefix} yields: {exc}")
            status_rows.append(
                {
                    "source": f"{prefix}_yield",
                    "code": (
                        "us_trycr" if prefix == "real" else "us_tycr"
                    ),
                    "status": "error",
                    "rows": 0,
                    "message": str(exc),
                }
            )

    try:
        print("Downloading Chinese gold ETF share data...")
        etf_features = _gold_etf_share_features(
            pro, start_date, end_date
        )
        if not etf_features.empty:
            feature_frames.append(etf_features)
            status_rows.append(
                {
                    "source": "gold_etf_shares",
                    "code": "fund_share",
                    "status": "ok",
                    "rows": len(etf_features),
                    "message": "",
                }
            )
        else:
            status_rows.append(
                {
                    "source": "gold_etf_shares",
                    "code": "fund_share",
                    "status": "empty",
                    "rows": 0,
                    "message": "No eligible ETF share observations",
                }
            )
    except Exception as exc:
        print(f"  Skipping gold ETF shares: {exc}")
        status_rows.append(
            {
                "source": "gold_etf_shares",
                "code": "fund_share",
                "status": "error",
                "rows": 0,
                "message": str(exc),
            }
        )

    if not feature_frames:
        save_status()
        print(
            "No external factor series could be downloaded. Check Tushare "
            "permissions, endpoint codes, and network access. Continuing "
            "with Au99.99-only features."
        )
        return pd.DataFrame()

    factors = pd.concat(feature_frames, axis=1).sort_index()
    if factors.columns.duplicated().any():
        # Later frames are repairs. ``last`` keeps their non-null values and
        # falls back to the seeded cache where a repaired series is missing.
        factors = factors.T.groupby(level=0, sort=False).last().T
    factors = factors[~factors.index.duplicated(keep="last")]

    # Rates are separately sourced but economically useful as a spread.
    if {
        "nominal_yield_10y",
        "real_yield_10y",
    }.issubset(factors.columns):
        factors["breakeven_inflation_10y"] = (
            factors["nominal_yield_10y"]
            - factors["real_yield_10y"]
        )

    save_status()
    factors.reset_index(names="trade_date").to_csv(
        cache_file, index=False
    )
    print(f"Saved external factors: {cache_file}")
    return factors
