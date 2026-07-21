# Selected reproducibility reports

This folder contains compact result tables selected from the generated
`outputs/` directory. They document the current project state without
committing complete market-data caches, large prediction ledgers or serialized
models.

- `huber_loss_minimization_summary.csv`: numerical optimizer diagnostics.
- `huber_loss_minimizing_coefficients.csv`: final standardized and raw-unit
  coefficients.
- `huber_10fold_overall_metrics.csv`: combined blocked ten-fold amount errors.
- `huber_10fold_fold_metrics.csv`: per-period stability diagnostics.
- `huber_10fold_coefficient_summary.csv`: coefficient dispersion across folds.
- `selected_horizon.json`: validation-only fixed-horizon selection record.
- `eligible_horizon_models.csv`: candidates that passed the eligibility rules.

The all-data coefficient fit is for parameter estimation, not out-of-sample
performance. The blocked ten-fold results are diagnostic rather than a
leakage-free trading backtest.
