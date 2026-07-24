# Local data cache

This directory is populated by the research scripts and is intentionally
excluded from Git because the files are downloaded, time-varying, and may be
large.

Set `TUSHARE_TOKEN`, then run:

```powershell
python international_gold_model.py --horizon 1 --refresh
```

The preferred target cache is raw `XAUUSD.FXCM` bid/ask OHLC from Tushare
`fx_daily`. If the current token cannot access that endpoint, the loader can
recover the previously cached FXCM mid-close sequence from
`data/factors/macro_factors_v3_*.csv`. That fallback changes the target to
close-to-close and must not be presented as an executable open-to-close P&L
test. Every run records the provider and target mode in
`reports/international_gold/metadata.json`.

`event_exclusions.csv` is committed because it changes the modelling sample.
It must contain `start_date` and `end_date` columns plus a documented rationale
and source for every window. These windows were defined after residual review,
so filtered test results are diagnostic and must not be represented as an
untouched out-of-sample evaluation. Do not remove a period merely because it
reduces model accuracy.

The international pipeline does not currently apply those event exclusions;
they were diagnosed on the old Chinese-gold target and cannot be transferred
without a new, predeclared international-gold study.
