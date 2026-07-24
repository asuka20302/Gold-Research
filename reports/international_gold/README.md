# International XAU/USD baseline

This report switches the prediction target from SGE `Au99.99` (CNY/gram) to
FXCM `XAUUSD` (USD/troy ounce).

## Data and target used in this run

- Provider provenance: recovered Tushare/FXCM factor cache.
- Available target field: mid close only.
- Horizon: one FXCM session.
- Target: `log(next close / signal close)`.
- Evaluation: 19 equal chronological blocks; Block 19 is common holdout.
- P&L status: not executable because the recovered cache has no next-session
  open and no bid/ask spread.

The raw Tushare `fx_daily` endpoint is the preferred source because it provides
bid and ask OHLC. The current token failed its smoke test, so this run did not
invent or approximate an open price.

## Feature and model result

Training-only, p-value-constrained AIC reduction retained seven factors:

1. one-session XAU/USD return;
2. five-session XAU/USD momentum;
3. 20-session XAU/USD volatility;
4. one-session oil return;
5. one-session real-yield change;
6. 10-year breakeven inflation;
7. GVZ level.

The five maintained candidates are OLS, Ridge, Lasso, Elastic Net, and
moving-block-bagged Huber. Their voting weights are inverse to growth MAE on
Blocks 11–18.

On Block 19, the zero-return baseline had the lowest growth MAE
(1.2775 percentage points). The best fitted candidates, Lasso and Elastic Net,
had 1.2781 percentage points; the weighted ensemble had 1.2807. Therefore this
first international close-to-close specification does **not** demonstrate
out-of-sample predictive skill over a zero-return forecast. The price lines can
still look visually close because every predicted next close starts from the
current close; the return-error comparison is the relevant diagnostic.

## Reproduce

```powershell
python international_gold_model.py --horizon 1
```

To require full bid/ask OHLC and prevent the close-only fallback:

```powershell
python international_gold_model.py --horizon 1 --refresh --target-mode open_to_close
```
