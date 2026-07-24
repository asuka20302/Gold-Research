import numpy as np
import pandas as pd

import international_gold_data as igd
import international_gold_model as igm


def test_recover_shifted_fxcm_close_restores_internal_dates(tmp_path):
    path = tmp_path / "factors.csv"
    dates = pd.bdate_range("2025-01-01", periods=205)
    stored_values = [np.nan, *np.arange(100.0, 304.0)]
    pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            # Old factor construction stored raw[i-1] at date[i].
            "xauusd_close_known": stored_values,
        }
    ).to_csv(path, index=False)

    recovered = igd.recover_shifted_fxcm_close(
        path, "20250101", "20261231"
    )

    assert recovered["trade_date"].iloc[0] == dates[1]
    assert recovered["trade_date"].iloc[-1] == dates[-2]
    assert recovered["mid_close"].iloc[:2].tolist() == [101.0, 102.0]


def synthetic_prices(rows: int = 600) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = 1_500.0 * np.exp(np.linspace(0.0, 0.25, rows))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "mid_open": close * 0.999,
            "mid_close": close,
        }
    )


def test_close_to_close_target_uses_strictly_future_close():
    prices = synthetic_prices()
    dataset = igm.build_dataset(
        prices,
        horizon=2,
        target_mode="close_to_close",
    )
    first_date = dataset.index[0]
    source = prices.set_index("trade_date")
    location = source.index.get_loc(first_date)
    expected_exit = source.index[location + 2]
    expected_target = np.log(
        source.iloc[location + 2]["mid_close"]
        / source.iloc[location]["mid_close"]
    )

    assert dataset.loc[first_date, "exit_date"] == expected_exit
    assert np.isclose(dataset.loc[first_date, "target"], expected_target)
    assert dataset.loc[first_date, "entry_date"] == first_date


def test_open_to_close_enters_next_session():
    prices = synthetic_prices()
    dataset = igm.build_dataset(
        prices,
        horizon=1,
        target_mode="open_to_close",
    )
    first_date = dataset.index[0]
    source = prices.set_index("trade_date")
    location = source.index.get_loc(first_date)
    next_date = source.index[location + 1]

    assert dataset.loc[first_date, "entry_date"] == next_date
    assert dataset.loc[first_date, "exit_date"] == next_date
    assert np.isclose(
        dataset.loc[first_date, "target"],
        np.log(
            source.loc[next_date, "mid_close"]
            / source.loc[next_date, "mid_open"]
        ),
    )


def test_equal_blocks_keep_block_19_last_and_equal():
    prices = synthetic_prices(700)
    dataset = igm.build_dataset(
        prices,
        horizon=1,
        target_mode="close_to_close",
    )
    excluded, blocks = igm.equal_chronological_blocks(dataset, n_blocks=19)

    assert len(blocks) == 19
    assert len({len(block) for block in blocks}) == 1
    assert excluded.index.max() < blocks[0].index.min()
    assert blocks[17].index.max() < blocks[18].index.min()
