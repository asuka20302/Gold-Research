# International gold return forecasting research

The maintained pipeline forecasts international spot gold, represented by
FXCM `XAUUSD` in USD per troy ounce. It uses five linear/robust candidates,
training-only AIC reduction, moving-block bagging, amount-error voting weights,
and a common Block 19 holdout.

The earlier Shanghai Gold Exchange `Au99.99` experiment remains in the
repository as a legacy comparison. It is no longer the default target.

## Current model set

The maintained comparison contains five regressors:

1. Ordinary Least Squares
2. Ridge
3. Lasso
4. Elastic Net
5. Moving-block-bagged Huber regression

The pipeline also contains training-only AIC/BIC/t-test/ANOVA feature
selection, interaction terms, market-state diagnostics, PCA and loss-function
experiments, moving-block bagging, weighted ensembles, fixed-horizon comparison
for horizons 1 through 7, and amount-based error reports.

## International target and timing

The preferred input is Tushare `fx_daily` for `XAUUSD.FXCM`. Tushare supplies
GMT-dated bid and ask OHLC fields, so the executable horizon-one target is:

```text
log(next-session mid close / next-session mid open)
```

If that endpoint cannot be refreshed, the code recovers the project's existing
Tushare/FXCM mid-close cache. A close-only run instead forecasts:

```text
log(next-session mid close / signal-session mid close)
```

The close-only fallback is useful for model research but is not an executable
P&L test because it has no next-session open or bid/ask spread. The output
metadata records which target was actually used.

## Data

The international pipeline downloads or loads:

- FXCM XAU/USD bid/ask OHLC from Tushare, when permission is available;
- a recovered FXCM XAU/USD mid-close cache otherwise;
- USD/CNH;
- dollar-index proxies;
- crude oil;
- US equity and volatility proxies;
- nominal and real yields and breakeven inflation.

SGE prices, the Shanghai premium, and SHFE gold are not target inputs in the
new international run. Downloaded data is intentionally not committed. See
[`data/README.md`](data/README.md).

## Setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:TUSHARE_TOKEN="your_token"
```

In PyCharm, open the repository root, select the `.venv` interpreter, and set
`TUSHARE_TOKEN` in the Run/Debug configuration.

## Run the maintained international experiment

```powershell
python international_gold_model.py --horizon 1
```

Useful options:

```powershell
python international_gold_model.py --horizon 1 --refresh
python international_gold_model.py --horizon 1 --target-mode open_to_close
python international_gold_model.py --horizon 1 --target-mode close_to_close
```

The run uses Blocks 1–9 for AIC selection, Block 10 for model
hyperparameters, Blocks 11–18 for inverse-growth-MAE voting weights, and Block
19 once for the common final comparison. It reports errors in both growth
percentage points and USD per troy ounce; it does not optimize directional
accuracy.

Generated data and figures go to `data/` and `outputs/`; both are ignored by
Git. Compact result tables are committed under
[`reports/international_gold/`](reports/international_gold/).

## Main research scripts

| File | Purpose |
|---|---|
| `international_gold_data.py` | FXCM bid/ask downloader and close-cache recovery |
| `international_gold_model.py` | Maintained five-model XAU/USD experiment |
| `gold_model_comparison.py` | Legacy SGE Au99.99 chronological experiment |
| `compare_fixed_horizons.py` | Aggregate horizons 1–7 and apply eligibility rules |
| `statistical_feature_selection.py` | t-test, ANOVA, AIC and BIC reduction |
| `interaction_regime_features.py` | Interactions and 0–4 market-state labels |
| `compare_bagging_all_models.py` | Moving-block bagging comparison |
| `compare_pca_models.py` | PCA experiment |
| `compare_loss_functions.py` | Squared, Huber and absolute-loss comparison |
| `compare_weighted_ensemble.py` | Validation-selected model weighting |
| `huber_10fold_blocked_cv.py` | Requested 9-of-10 blocked Huber diagnostic |
| `huber_rolling_19block_cv.py` | Fair 19-block, fixed-window leakage-free Huber validation |
| `locked_block19_ensemble.py` | Five-model development and frozen-weight comparison on common Block 19 |
| `locked_block19_pnl.py` | Development-selected Bagged Huber position sizing and locked Block 19 P&L |
| `huber_loss_coefficient_report.py` | Analytic-gradient Huber loss minimization |
| `gold_PnL_ledger.py` | Standalone position and P&L ledger |
| `tools/export_public_reports.py` | Copy selected outputs with portable repository-relative paths |

## Validation warning

The 9-of-10 blocked Huber experiment is retained as a coefficient-stability diagnostic.
For folds 1–9, training includes observations later than the held-out block, so
it is **not** a leakage-free trading backtest. The maintained fair validation
uses 19 equal chronological blocks: each of the ten
Huber models trains on the immediately preceding nine blocks and tests the next
block. Every model receives 1,287 training rows and 143 test rows, with no later
dates in training.

For a common final comparison, `locked_block19_ensemble.py` uses Blocks 1–18
for rolling development predictions and voting weights, fits all five final
models on Blocks 10–18, and evaluates them on the same Block 19 observations.
Block 19 is computationally excluded from fitting and weighting. Because its
Huber outcomes were inspected in an earlier experiment, it is described as a
locked common holdout rather than a historically pristine test set.

The legacy P&L experiment is a theoretical direct-Au99.99 long/flat test,
not an ETF execution simulation. It selects the threshold and volatility sizing
only from rolling development predictions on Blocks 10–18, then freezes the
rule before calculating Block 19 P&L. Commission and slippage are explicit CLI
assumptions and must be replaced with the actual instrument schedule before use.

## Tests

```powershell
pytest
```

GitHub Actions runs the same tests on every push and pull request. Tests cover
target timing, interaction/regime construction, feature hierarchy, and the
analytic Huber gradient used by the numerical optimizer, plus the equal-size
past-only construction of the 19-block rolling validation.

## Legacy QMT integration

QMT files are isolated under [`qmt/`](qmt/). They are optional and do not affect
the research pipeline. They still map the old Chinese-gold signal to a Chinese
ETF and must not be treated as the execution layer for XAU/USD. See
[`qmt/README.md`](qmt/README.md) before running a QMT backtest.

## Documentation

- [`docs/Gold_Model_Project_Notes.docx`](docs/Gold_Model_Project_Notes.docx)
- [`docs/黄金量化预测项目完整流程.docx`](docs/黄金量化预测项目完整流程.docx)

The Chinese manual can be regenerated with:

```powershell
python tools/build_pipeline_manual_cn.py
```

Refresh the compact GitHub reports after rerunning experiments with:

```powershell
python tools/export_public_reports.py
```
