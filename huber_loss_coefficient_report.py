"""Fit and report the coefficients that minimize the current Huber loss.

This is a coefficient-estimation report, not an out-of-sample backtest.  It
fits the frozen horizon-one Huber specification to every available
event-filtered row, refines the numerical solution with L-BFGS-B, verifies the
gradient at the solution, and exports coefficients in standardized and raw
feature units.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.base import clone
from sklearn.linear_model._huber import _huber_loss_and_gradient

from advanced_experiment_utils import (
    PROJECT,
    amount_metrics,
    experiment_frames,
    selected_templates,
)


OUTPUT = PROJECT / "outputs" / "huber_loss_minimization"


def exact_loss_and_gradient(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Return scikit-learn's exact regularized Huber objective and gradient."""
    return _huber_loss_and_gradient(
        parameters,
        x,
        y,
        epsilon,
        alpha,
        sample_weight=np.ones(len(y), dtype=float),
    )


def normalized_loss_and_gradient(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Scale the complete objective by n to improve optimizer conditioning."""
    loss, gradient = exact_loss_and_gradient(
        parameters, x, y, epsilon, alpha
    )
    return float(loss / len(y)), np.asarray(gradient, dtype=float) / len(y)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    _, _, pretest, test = experiment_frames()
    all_data = pd.concat([pretest, test]).sort_index()
    if all_data.index.has_duplicates:
        raise RuntimeError("Duplicate dates found in the all-data estimation frame.")

    templates, feature_map = selected_templates()
    pipeline = clone(templates["Huber"])
    features = feature_map["Huber"]
    pipeline.fit(all_data[features], all_data["target"])

    imputed = pipeline.named_steps["imputer"].transform(all_data[features])
    x_scaled = np.asarray(
        pipeline.named_steps["scaler"].transform(imputed), dtype=float
    )
    y = all_data["target"].to_numpy(dtype=float)
    estimator = pipeline.named_steps["model"]
    epsilon = float(estimator.epsilon)
    alpha = float(estimator.alpha)

    initial_parameters = np.concatenate(
        [
            np.asarray(estimator.coef_, dtype=float).ravel(),
            [float(estimator.intercept_), float(estimator.scale_)],
        ]
    )
    initial_loss, initial_gradient = exact_loss_and_gradient(
        initial_parameters, x_scaled, y, epsilon, alpha
    )

    loss_trace: list[dict[str, float | int]] = []

    def callback(intermediate_result: optimize.OptimizeResult) -> None:
        value, gradient = exact_loss_and_gradient(
            np.asarray(intermediate_result.x), x_scaled, y, epsilon, alpha
        )
        loss_trace.append(
            {
                "refinement_iteration": len(loss_trace) + 1,
                "loss": float(value),
                "gradient_infinity_norm": float(np.max(np.abs(gradient))),
            }
        )

    bounds = [(None, None)] * (len(features) + 1) + [
        (10.0 * np.finfo(float).eps, None)
    ]
    result = optimize.minimize(
        normalized_loss_and_gradient,
        initial_parameters,
        args=(x_scaled, y, epsilon, alpha),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": 20_000,
            "maxls": 100,
            "ftol": 1e-16,
            "gtol": 1e-12,
        },
    )

    candidate_parameters = np.asarray(result.x, dtype=float)
    candidate_loss, candidate_gradient = exact_loss_and_gradient(
        candidate_parameters, x_scaled, y, epsilon, alpha
    )
    if candidate_loss <= initial_loss:
        final_parameters = candidate_parameters
        final_loss = float(candidate_loss)
        final_gradient = candidate_gradient
        refinement_used = True
    else:
        final_parameters = initial_parameters
        final_loss = float(initial_loss)
        final_gradient = initial_gradient
        refinement_used = False

    coefficients = final_parameters[: len(features)]
    intercept_scaled = float(final_parameters[-2])
    scale = float(final_parameters[-1])
    estimator.coef_ = coefficients.copy()
    estimator.intercept_ = intercept_scaled
    estimator.scale_ = scale
    residual = y - (x_scaled @ coefficients + intercept_scaled)
    estimator.outliers_ = np.abs(residual) > epsilon * scale

    scaler = pipeline.named_steps["scaler"]
    raw_coefficients = coefficients / np.asarray(scaler.scale_, dtype=float)
    raw_intercept = intercept_scaled - float(
        np.dot(raw_coefficients, np.asarray(scaler.mean_, dtype=float))
    )

    coefficient_report = pd.DataFrame(
        {
            "feature": features,
            "standardized_coefficient": coefficients,
            "absolute_standardized_coefficient": np.abs(coefficients),
            "raw_unit_coefficient": raw_coefficients,
            "feature_training_mean_after_imputation": scaler.mean_,
            "feature_training_scale": scaler.scale_,
        }
    )
    coefficient_report["absolute_importance_rank"] = coefficient_report[
        "absolute_standardized_coefficient"
    ].rank(method="min", ascending=False).astype(int)
    coefficient_report = coefficient_report.sort_values(
        "absolute_importance_rank"
    ).reset_index(drop=True)

    zero_parameters = np.concatenate(
        [np.zeros(len(features) + 1, dtype=float), [scale]]
    )
    zero_loss, _ = exact_loss_and_gradient(
        zero_parameters, x_scaled, y, epsilon, alpha
    )
    prediction = np.asarray(pipeline.predict(all_data[features]), dtype=float)
    amount = amount_metrics(
        all_data["target"],
        prediction,
        all_data["entry_open"],
        all_data["exit_close"],
    )

    summary = {
        "purpose": "all-data coefficient estimation; not out-of-sample evaluation",
        "rows": int(len(all_data)),
        "start_date": str(all_data.index.min().date()),
        "end_date": str(all_data.index.max().date()),
        "features": int(len(features)),
        "epsilon": epsilon,
        "alpha": alpha,
        "optimizer": (
            "L-BFGS-B with analytic Huber gradient; complete objective "
            "divided by n for numerical conditioning"
        ),
        "sklearn_initial_iterations": int(estimator.n_iter_),
        "refinement_iterations": int(result.nit),
        "refinement_success": bool(result.success),
        "refinement_message": str(result.message),
        "refinement_used": refinement_used,
        "initial_loss": float(initial_loss),
        "final_loss": final_loss,
        "loss_reduction_from_sklearn_solution": float(initial_loss - final_loss),
        "zero_coefficient_loss_at_final_scale": float(zero_loss),
        "loss_reduction_vs_zero_coefficient_pct": float(
            100.0 * (zero_loss - final_loss) / zero_loss
        ),
        "final_gradient_infinity_norm": float(np.max(np.abs(final_gradient))),
        "final_mean_gradient_infinity_norm": float(
            np.max(np.abs(final_gradient)) / len(y)
        ),
        "standardized_intercept": intercept_scaled,
        "raw_unit_intercept": raw_intercept,
        "estimated_residual_scale": scale,
        "outlier_rows": int(np.sum(estimator.outliers_)),
        "outlier_fraction": float(np.mean(estimator.outliers_)),
        **{key: float(value) for key, value in amount.items()},
    }

    coefficient_report.to_csv(
        OUTPUT / "loss_minimizing_huber_coefficients.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(
        OUTPUT / "loss_minimization_summary.csv", index=False
    )
    pd.DataFrame(loss_trace).to_csv(
        OUTPUT / "loss_refinement_trace.csv", index=False
    )
    with (OUTPUT / "loss_minimization_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    joblib.dump(pipeline, OUTPUT / "final_all_data_huber_model.joblib")

    print("\nLoss-minimizing Huber coefficients (standardized features):")
    print(
        coefficient_report[
            [
                "absolute_importance_rank",
                "feature",
                "standardized_coefficient",
                "raw_unit_coefficient",
            ]
        ].to_string(index=False)
    )
    print("\nOptimizer summary:")
    for key, value in summary.items():
        if key not in amount:
            print(f"  {key}: {value}")
    print(f"\nSaved reports to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
