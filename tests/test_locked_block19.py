import numpy as np
import pandas as pd

from huber_rolling_19block_cv import equal_chronological_blocks
from locked_block19_ensemble import development_windows, inverse_mae_weights


def test_development_windows_never_use_block_19() -> None:
    dates = pd.bdate_range("2020-01-01", periods=95)
    frame = pd.DataFrame(
        {
            "target": np.linspace(-0.01, 0.01, 95),
            "exit_date": dates,
        },
        index=dates,
    )
    excluded, blocks = equal_chronological_blocks(frame, n_blocks=19)
    windows = development_windows(blocks)

    assert excluded.empty
    assert len(windows) == 9
    holdout = blocks[18]
    for score_block, training, scoring in windows:
        assert 10 <= score_block <= 18
        assert len(training) == 45
        assert len(scoring) == 5
        assert training.index.max() < scoring.index.min()
        assert scoring.index.max() < holdout.index.min()


def test_inverse_mae_weights_are_positive_and_sum_to_one() -> None:
    actual = pd.Series([0.0, 0.01, -0.01])
    predictions = pd.DataFrame(
        {
            "OLS": [0.0, 0.009, -0.009],
            "Ridge": [0.0, 0.008, -0.008],
            "Lasso": [0.0, 0.007, -0.007],
            "ElasticNet": [0.0, 0.006, -0.006],
            "BaggedHuber": [0.0, 0.005, -0.005],
        }
    )
    mae, weights = inverse_mae_weights(predictions, actual)

    assert (mae > 0).all()
    assert (weights > 0).all()
    assert np.isclose(weights.sum(), 1.0)
    assert weights["OLS"] > weights["BaggedHuber"]
