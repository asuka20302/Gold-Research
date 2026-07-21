# Optional QMT integration

This folder converts research outputs into a QMT-compatible signal file and
contains the corresponding long/flat Gold ETF example strategy.

From the repository root:

```powershell
python qmt/run_qmt_gold_pipeline.py
```

The command regenerates the main outputs and writes
`outputs/qmt_gold_signals.csv`. The QMT strategy reads that file by default.
When QMT runs the strategy from another location, set `QMT_GOLD_SIGNAL_FILE` to
the absolute CSV path or edit `SIGNAL_FILE` inside the QMT editor.

Important limitations:

- `predicted_gold_price` is an Au99.99 CNY/gram forecast, not the ETF share
  price.
- The stock-account example is long/flat because ordinary ETF positions cannot
  be shorted directly.
- Exact open-to-close execution requires QMT settings or intraday bars that can
  represent both the entry and the close exit.
- Keep `ENABLE_LIVE_ORDERS = False` until the backtest and account settings have
  been independently verified.
