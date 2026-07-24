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
- `huber_rolling_19block_overall_metrics.csv`: combined leakage-free amount errors.
- `huber_rolling_19block_fold_metrics.csv`: equal-window period results.
- `huber_rolling_19block_coefficient_stability.csv`: coefficient sign and dispersion report.
- `huber_rolling_19block_metadata.json`: exact 19-block construction and safeguards.
- `locked_block19_metrics.csv`: common Block 19 amount errors for all five models and the ensemble.
- `locked_block19_weights.csv`: voting weights learned exclusively from Blocks 1–18.
- `locked_block19_metadata.json`: holdout-isolation rules and final selection record.
- `locked_block19_pnl_summary.csv`: net P&L and risk comparison for the frozen strategy and benchmarks.
- `locked_block19_cost_sensitivity.csv`: primary-rule results under zero, half, base and double costs.
- `locked_block19_development_stability.csv`: primary-rule results across development Blocks 10–18.
- `locked_block19_trading_rules.json`: timing, cost assumptions and development-selected rules.

The all-data coefficient fit is for parameter estimation, not out-of-sample
performance. The blocked ten-fold results are diagnostic rather than a
leakage-free trading backtest. Use the 19-block rolling reports as the current
fair out-of-sample Huber validation.
