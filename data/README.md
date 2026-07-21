# Local data cache

This directory is populated by the research scripts and is intentionally
excluded from Git because the files are downloaded, time-varying, and may be
large.

Set `TUSHARE_TOKEN`, then run:

```powershell
python gold_model_comparison.py --horizon 1
```

`event_exclusions.csv` is committed because it changes the modelling sample.
It must contain `start_date` and `end_date` columns plus a documented rationale
and source for every window. These windows were defined after residual review,
so filtered test results are diagnostic and must not be represented as an
untouched out-of-sample evaluation. Do not remove a period merely because it
reduces model accuracy.
