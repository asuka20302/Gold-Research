import numpy as np
import pandas as pd

from huber_rolling_19block_cv import (
    equal_chronological_blocks,
    rolling_windows,
)


def test_nineteen_blocks_create_ten_equal_past_only_windows() -> None:
    dates = pd.bdate_range("2020-01-01", periods=100)
    frame = pd.DataFrame({"value": np.arange(100)}, index=dates)

    excluded, blocks = equal_chronological_blocks(frame, n_blocks=19)
    windows = rolling_windows(blocks, train_blocks=9)

    assert len(excluded) == 5
    assert len(blocks) == 19
    assert {len(block) for block in blocks} == {5}
    assert len(windows) == 10
    for _, training_blocks, test_block in windows:
        training = pd.concat(training_blocks)
        assert len(training) == 45
        assert len(test_block) == 5
        assert training.index.max() < test_block.index.min()


def test_equal_block_builder_rejects_duplicate_dates() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    frame = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

    try:
        equal_chronological_blocks(frame, n_blocks=2)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("Duplicate dates should be rejected.")
