import unittest
import numpy as np
import pandas as pd

from attention_sentiment_thesis.preprocessing.returns import construct_returns
from attention_sentiment_thesis.preprocessing.panel import lag_wpv_by_trading_observation
from attention_sentiment_thesis.attention.wpv import compute_raw_wpv
from attention_sentiment_thesis.attention.atv import compute_atv

class ReturnsAndAttentionTests(unittest.TestCase):
    def test_return_construction(self):
        frame = pd.DataFrame({
            "firm_id": ["a"] * 3,
            "trading_date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
            "open_price": [100.0, 101.0, 102.0],
            "close_price": [101.0, 103.0, 102.0],
        })
        out = construct_returns(frame)
        self.assertAlmostEqual(out.loc[1, "r_cc"], np.log(103.0 / 101.0))
        self.assertAlmostEqual(out.loc[0, "target_return"], out.loc[1, "r_cc"])

    def test_wpv_lag_crosses_nontrading_days(self):
        dates = pd.to_datetime([
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-09", "2024-01-10",
        ])
        raw = compute_raw_wpv(pd.DataFrame({
            "firm_id": ["a"] * 6, "trading_date": dates,
            "page_views": [10, 11, 12, 13, 20, 15],
        }), min_observations=3)
        lagged = lag_wpv_by_trading_observation(raw)
        self.assertEqual(lagged.loc[4, "trading_date"], pd.Timestamp("2024-01-09"))
        self.assertEqual(lagged.loc[4, "wpv_lag1"], raw.loc[3, "wpv_raw"])

    def test_atv_uses_prior_60_and_excludes_current(self):
        dates = pd.bdate_range("2024-01-01", periods=62)
        volumes = np.arange(1.0, 63.0)
        out = compute_atv(pd.DataFrame({
            "firm_id": ["a"] * 62, "trading_date": dates, "volume": volumes,
        }))
        expected = (61.0 - np.mean(np.arange(1.0, 61.0))) / np.std(
            np.arange(1.0, 61.0), ddof=1
        )
        self.assertAlmostEqual(out.loc[60, "atv"], expected)

    def test_wpv_benchmark_excludes_current(self):
        frame = pd.DataFrame({
            "firm_id": ["a"] * 5,
            "trading_date": pd.bdate_range("2024-01-01", periods=5),
            "page_views": [10.0, 10.0, 10.0, 10.0, 1000.0],
        })
        out = compute_raw_wpv(frame)
        self.assertAlmostEqual(out.loc[4, "wpv_raw"], np.log(1000.0) - np.log(10.0))

if __name__ == "__main__":
    unittest.main()
