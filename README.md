# Gold return forecasting research

Reproducible research pipeline for forecasting executable Shanghai Gold
Exchange `Au99.99` returns with linear and robust regression models. The signal
is constructed after the current close, enters at the following session's open,
and predicts the close at a fixed horizon. The main research choice is horizon
one.

## Current model set

The maintained comparison contains five regressors:

1. Ordinary Least Squares
2. Ridge
3. Lasso
4. Elastic Net
5. Huber regression

The pipeline also contains training-only AIC/BIC/t-test/ANOVA feature
selection, interaction terms, market-state diagnostics, PCA and loss-function
experiments, moving-block bagging, weighted ensembles, fixed-horizon comparison
for horizons 1 through 7, and amount-based error reports.

## Prediction target

For horizon one, the target is the executable log return

```text
log(next-session close / next-session open)
```

Features are based on information available by the signal date close. External
markets are shifted before joining so later same-calendar-day observations are
not treated as known in Shanghai.

## Data

The project downloads and caches:

- SGE Au99.99 prices from Tushare;
- international gold;
- USD/CNH;
- dollar-index proxies;
- crude oil;
- US equity and volatility proxies;
- nominal and real yields and breakeven inflation;
- Shanghai gold premium and SHFE gold features.

External-market fallbacks use Yahoo Finance and AKShare when the applicable
Tushare endpoint returns no observations. Downloaded data is intentionally not
committed. See [`data/README.md`](data/README.md).

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

## Run the main experiment

```powershell
python gold_model_comparison.py --horizon 1
```

Useful options:

```powershell
python gold_model_comparison.py --horizon 1 --refresh
python gold_model_comparison.py --horizon 1 --skip-external-factors
python gold_model_comparison.py --horizon 1 --feature-selection aic
python gold_model_comparison.py --horizon 1 --ignore-event-exclusions
```

Generated data and figures go to `data/` and `outputs/`; both folders are
ignored by Git. A compact set of final, reviewable result tables is committed
under [`reports/`](reports/).

## Main research scripts

| File | Purpose |
|---|---|
| `gold_model_comparison.py` | Main five-model chronological experiment |
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

## Tests

```powershell
pytest
```

GitHub Actions runs the same tests on every push and pull request. Tests cover
target timing, interaction/regime construction, feature hierarchy, and the
analytic Huber gradient used by the numerical optimizer, plus the equal-size
past-only construction of the 19-block rolling validation.

## QMT integration

QMT files are isolated under [`qmt/`](qmt/). They are optional and do not affect
the research pipeline. See [`qmt/README.md`](qmt/README.md) before running a QMT
backtest.

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
