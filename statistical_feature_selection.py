"""Training-only statistical feature selection for linear return models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSStatistics:
    rss: float
    rank: int
    df_resid: int
    aic: float
    bic: float
    coefficients: pd.Series
    standard_errors: pd.Series
    t_statistics: pd.Series
    p_values: pd.Series


def _training_matrix(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute from the supplied training frame and add an intercept."""
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    if values.isna().any().any():
        missing = values.columns[values.isna().any()].tolist()
        raise ValueError(f"Training-only imputation failed for: {missing}")
    x = np.column_stack([np.ones(len(values)), values.to_numpy(float)])
    y = pd.to_numeric(frame["target"], errors="coerce").to_numpy(float)
    return x, y


def ols_statistics(
    frame: pd.DataFrame,
    features: list[str],
) -> OLSStatistics:
    """Calculate OLS t-tests, AIC, and BIC without external dependencies."""
    x, y = _training_matrix(frame, features)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    rss = float(residual @ residual)
    n = len(y)
    df_resid = max(n - int(rank), 1)
    residual_variance = rss / df_resid
    covariance = residual_variance * np.linalg.pinv(x.T @ x)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df_resid)
    names = ["intercept", *features]
    safe_rss = max(rss, np.finfo(float).tiny)
    aic = n * np.log(safe_rss / n) + 2 * int(rank)
    bic = n * np.log(safe_rss / n) + np.log(n) * int(rank)
    return OLSStatistics(
        rss=rss,
        rank=int(rank),
        df_resid=df_resid,
        aic=float(aic),
        bic=float(bic),
        coefficients=pd.Series(beta, index=names),
        standard_errors=pd.Series(standard_errors, index=names),
        t_statistics=pd.Series(t_values, index=names),
        p_values=pd.Series(p_values, index=names),
    )


def _feature_group(feature: str) -> str:
    if feature in {
        "return_1d", "momentum_5d", "momentum_20d", "volatility_5d",
        "volatility_20d", "price_range", "volume_change_5d", "ma_gap_20d",
    }:
        return "sge_technical"
    if feature.startswith("xauusd_"):
        return "international_gold"
    if feature.startswith("usdcnh_") or feature.startswith("dollar_"):
        return "currency"
    if "yield" in feature or "breakeven" in feature:
        return "rates_inflation"
    if feature.startswith("sp500_") or feature.startswith("oil_"):
        return "risk_assets_commodities"
    if "premium" in feature:
        return "relative_value"
    if feature.startswith("shfe_gold_"):
        return "shfe_futures"
    return "other"


def _partial_f_test(
    full: OLSStatistics,
    reduced: OLSStatistics,
) -> tuple[float, float, int]:
    restrictions = max(full.rank - reduced.rank, 0)
    if restrictions == 0 or full.rss <= 0:
        return np.nan, np.nan, restrictions
    numerator = max(reduced.rss - full.rss, 0.0) / restrictions
    denominator = full.rss / full.df_resid
    f_statistic = numerator / denominator if denominator > 0 else np.nan
    p_value = stats.f.sf(f_statistic, restrictions, full.df_resid)
    return float(f_statistic), float(p_value), restrictions


def t_test_backward(
    train: pd.DataFrame,
    features: list[str],
    alpha: float = 0.05,
) -> tuple[list[str], list[dict[str, object]]]:
    selected = list(features)
    audit: list[dict[str, object]] = []
    step = 0
    while len(selected) > 1:
        fitted = ols_statistics(train, selected)
        p_values = fitted.p_values.drop("intercept")
        worst = str(p_values.idxmax())
        worst_p = float(p_values.loc[worst])
        audit.append(
            {
                "method": "t_test",
                "step": step,
                "candidate": worst,
                "candidate_type": "feature",
                "statistic": float(fitted.t_statistics.loc[worst]),
                "p_value": worst_p,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "action": "remove" if worst_p > alpha else "stop",
            }
        )
        if worst_p <= alpha:
            break
        selected.remove(worst)
        step += 1
    return selected, audit


def anova_group_backward(
    train: pd.DataFrame,
    features: list[str],
    alpha: float = 0.05,
) -> tuple[list[str], list[dict[str, object]]]:
    selected = list(features)
    audit: list[dict[str, object]] = []
    step = 0
    while True:
        groups: dict[str, list[str]] = {}
        for feature in selected:
            groups.setdefault(_feature_group(feature), []).append(feature)
        if len(groups) <= 1:
            break
        full = ols_statistics(train, selected)
        tests: list[tuple[str, float, float, int]] = []
        for group, members in groups.items():
            reduced_features = [x for x in selected if x not in members]
            reduced = ols_statistics(train, reduced_features)
            f_stat, p_value, restrictions = _partial_f_test(full, reduced)
            tests.append((group, f_stat, p_value, restrictions))
        finite = [row for row in tests if np.isfinite(row[2])]
        if not finite:
            break
        group, f_stat, p_value, restrictions = max(
            finite, key=lambda row: row[2]
        )
        audit.append(
            {
                "method": "anova",
                "step": step,
                "candidate": group,
                "candidate_type": "feature_group",
                "statistic": f_stat,
                "p_value": p_value,
                "restrictions": restrictions,
                "aic": full.aic,
                "bic": full.bic,
                "action": "remove" if p_value > alpha else "stop",
            }
        )
        if p_value <= alpha:
            break
        selected = [x for x in selected if _feature_group(x) != group]
        step += 1
    return selected, audit


def information_criterion_backward(
    train: pd.DataFrame,
    features: list[str],
    criterion: str,
    alpha: float = 0.05,
    tolerance: float = 1e-9,
) -> tuple[list[str], list[dict[str, object]]]:
    """Remove statistically weak features when AIC/BIC also improves.

    A p-value above ``alpha`` means that the coefficient is not statistically
    distinguishable from zero; it does not prove that the population
    coefficient is exactly zero. Therefore p-value screening defines the
    removal candidates, while the information criterion decides whether a
    removal is accepted.
    """
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")
    selected = list(features)
    audit: list[dict[str, object]] = []
    step = 0
    while len(selected) > 1:
        current = ols_statistics(train, selected)
        current_value = getattr(current, criterion)
        feature_p_values = current.p_values.drop("intercept")
        removable = feature_p_values[
            feature_p_values > alpha
        ].index.tolist()
        if not removable:
            audit.append(
                {
                    "method": criterion,
                    "step": step,
                    "candidate": "",
                    "candidate_type": "feature",
                    "p_value": np.nan,
                    "criterion_before": current_value,
                    "criterion_after": current_value,
                    "criterion_change": 0.0,
                    "aic": current.aic,
                    "bic": current.bic,
                    "action": "stop_all_p_values_significant",
                }
            )
            break
        candidates: list[tuple[str, float, OLSStatistics]] = []
        for feature in removable:
            reduced_features = [x for x in selected if x != feature]
            reduced = ols_statistics(train, reduced_features)
            candidates.append((feature, getattr(reduced, criterion), reduced))
        feature, best_value, reduced = min(candidates, key=lambda row: row[1])
        improves = best_value < current_value - tolerance
        audit.append(
            {
                "method": criterion,
                "step": step,
                "candidate": feature,
                "candidate_type": "feature",
                "p_value": float(feature_p_values.loc[feature]),
                "criterion_before": current_value,
                "criterion_after": best_value,
                "criterion_change": best_value - current_value,
                "aic": reduced.aic,
                "bic": reduced.bic,
                "action": "remove" if improves else "stop",
            }
        )
        if not improves:
            break
        selected.remove(feature)
        step += 1
    return selected, audit


def build_feature_subsets(
    train: pd.DataFrame,
    active_features: list[str],
    alpha: float = 0.05,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Build all training-only subsets and coefficient-test diagnostics."""
    full = ols_statistics(train, active_features)
    coefficient_table = pd.DataFrame(
        {
            "feature": full.coefficients.index,
            "coefficient": full.coefficients.values,
            "standard_error": full.standard_errors.values,
            "t_statistic": full.t_statistics.values,
            "p_value": full.p_values.values,
            "significant_at_5pct": full.p_values.values < alpha,
        }
    )
    t_features, t_audit = t_test_backward(train, active_features, alpha)
    anova_features, anova_audit = anova_group_backward(
        train, active_features, alpha
    )
    aic_features, aic_audit = information_criterion_backward(
        train, active_features, "aic", alpha=alpha
    )
    bic_features, bic_audit = information_criterion_backward(
        train, active_features, "bic", alpha=alpha
    )
    subsets = {
        "t_test": t_features,
        "anova": anova_features,
        "aic": aic_features,
        "bic": bic_features,
    }
    audit = pd.DataFrame(t_audit + anova_audit + aic_audit + bic_audit)
    return subsets, coefficient_table, audit
