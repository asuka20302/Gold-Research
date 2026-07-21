#coding:gbk
"""
QMT daily-bar strategy driven by the selected out-of-sample gold signal.

Default instrument: HuaAn Gold ETF (518880.SH).
This template is long/flat because an ordinary stock account cannot short an
ETF directly. Set the QMT backtest execution rule to the next bar's open.

Current research selection:
    model   = Huber
    horizon = 1
    signal  = today after close
    entry   = next trading session open
    target  = same session close

Important caveat:
    If this script is run as a daily-bar QMT strategy, it can place the entry
    order on the execution date, but same-day close exit depends on QMT's
    execution settings. For an exact open-to-close strategy, use an intraday
    period and explicitly flatten near the close.

The predicted_gold_price field is an Au99.99 CNY/gram forecast. It is shown in
the sub-chart and is not confused with the ETF's own share price.
"""

import csv
import os


DEFAULT_SIGNAL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "qmt_gold_signals.csv",
)
SIGNAL_FILE = os.environ.get("QMT_GOLD_SIGNAL_FILE", DEFAULT_SIGNAL_FILE)
TRADE_CODE = "518880.SH"
ACCOUNT_ID = "test"
BACKTEST_START = "2024-10-09 00:00:00"
BACKTEST_END = "2026-06-23 23:59:59"
BACKTEST_CAPITAL = 100000.0

# Predicted one-session log return required to enter.
# 0.0 means long whenever the selected model predicts a positive open-to-close
# return. Increase this, for example to 0.001, to trade only stronger signals.
ENTRY_THRESHOLD = 0.0
TARGET_WEIGHT = 0.90

# Keep this False while testing. It prevents an order on QMT's live/latest bar.
ENABLE_LIVE_ORDERS = False

g = {}


def init(ContextInfo):
    ContextInfo.set_universe([TRADE_CODE])
    if ContextInfo.do_back_test:
        ContextInfo.start = BACKTEST_START
        ContextInfo.end = BACKTEST_END
        ContextInfo.capital = BACKTEST_CAPITAL

    signals = {}
    with open(SIGNAL_FILE, "r", encoding="gbk") as signal_file:
        reader = csv.DictReader(signal_file)
        for row in reader:
            execution_date = (
                str(row["execution_date"]).replace("-", "")[:8]
            )
            signals[execution_date] = {
                "signal_date": (
                    str(row["signal_date"]).replace("-", "")[:8]
                ),
                "exit_date": str(row.get("exit_date", "")).replace(
                    "-", ""
                )[:8],
                "model": row.get("model", ""),
                "horizon": int(float(row.get("horizon", 1))),
                "predicted_return": float(row["predicted_return"]),
                "gold_close": float(row["gold_close"]),
                "entry_open": float(row.get("entry_open", 0.0)),
                "predicted_gold_price": float(
                    row["predicted_gold_price"]
                ),
            }

    g["signals"] = signals
    g["last_order_date"] = ""
    g["matched_signal_count"] = 0

    signal_dates = sorted(signals.keys())
    print(
        "[GoldStrategy] loaded signals:",
        len(signal_dates),
        "range:",
        signal_dates[0] if signal_dates else "none",
        "to",
        signal_dates[-1] if signal_dates else "none",
    )
    print(
        "[GoldStrategy] mode:",
        "backtest" if ContextInfo.do_back_test else "run/live",
        "period:",
        getattr(ContextInfo, "period", "daily"),
        "instrument:",
        TRADE_CODE,
    )


def handlebar(ContextInfo):
    # Never send live orders unless explicitly enabled. Historical bars must
    # remain active during a QMT backtest.
    if not ContextInfo.do_back_test and not ENABLE_LIVE_ORDERS:
        return

    bar_timetag = ContextInfo.get_bar_timetag(ContextInfo.barpos)
    execution_date = timetag_to_datetime(bar_timetag, "%Y%m%d")

    if execution_date not in g["signals"]:
        return
    if execution_date == g["last_order_date"]:
        return

    row = g["signals"][execution_date]
    signal_date = row["signal_date"]
    exit_date = row["exit_date"]
    model = row["model"]
    horizon = row["horizon"]
    predicted_return = float(row["predicted_return"])
    gold_close = float(row["gold_close"])
    entry_open = float(row["entry_open"])
    predicted_gold_price = float(row["predicted_gold_price"])

    # Long when the predicted gold-price increase exceeds the threshold;
    # otherwise flat. This comparison is equivalent to using predicted return.
    target_weight = (
        TARGET_WEIGHT
        if predicted_return >= ENTRY_THRESHOLD
        else 0.0
    )

    if ContextInfo.do_back_test:
        # QMT creates and manages the backtest account internally.
        order_target_percent(
            TRADE_CODE,
            target_weight,
            ContextInfo,
        )
    else:
        order_target_percent(
            TRADE_CODE,
            target_weight,
            ContextInfo,
            ACCOUNT_ID,
        )

    g["matched_signal_count"] += 1
    print(
        "[GoldStrategy] signal:",
        signal_date,
        "execution:",
        execution_date,
        "exit:",
        exit_date,
        "model:",
        model,
        "horizon:",
        horizon,
        "pred_return:",
        round(predicted_return, 6),
        "gold_close:",
        round(gold_close, 2),
        "entry_open:",
        round(entry_open, 2),
        "predicted_price:",
        round(predicted_gold_price, 2),
        "target_weight:",
        target_weight,
    )
    g["last_order_date"] = execution_date

    # Values shown on the QMT chart for diagnosis.
    ContextInfo.paint(
        "predicted_return",
        predicted_return,
        -1,
        0,
        "yellow",
        "noaxis",
    )
    ContextInfo.paint(
        "actual_gold_close",
        gold_close,
        -1,
        0,
        "white",
        "noaxis",
    )
    ContextInfo.paint(
        "predicted_gold_price",
        predicted_gold_price,
        -1,
        0,
        "magenta",
        "noaxis",
    )
    ContextInfo.paint(
        "target_weight",
        target_weight,
        -1,
        42,
        "cyan",
        "noaxis",
    )
