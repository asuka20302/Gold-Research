import numpy as np

from huber_loss_coefficient_report import exact_loss_and_gradient


def test_analytic_huber_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 3))
    y = rng.normal(scale=0.4, size=40)
    parameters = np.array([0.15, -0.08, 0.04, 0.02, 0.7])
    epsilon = 1.35
    alpha = 0.001

    _, analytic = exact_loss_and_gradient(
        parameters, x, y, epsilon, alpha
    )
    step = 1e-6
    numeric = np.empty_like(parameters)
    for index in range(len(parameters)):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        upper_loss, _ = exact_loss_and_gradient(
            upper, x, y, epsilon, alpha
        )
        lower_loss, _ = exact_loss_and_gradient(
            lower, x, y, epsilon, alpha
        )
        numeric[index] = (upper_loss - lower_loss) / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)
