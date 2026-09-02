import unittest

from attention_sentiment_thesis.spec import (
    CLARK_WEST_COMPARISONS, CONTROL_COLUMNS, FIRM_VARYING_REGRESSORS,
    FINAL_SPEC, MODEL_PREDICTORS,
)

class FinalSpecificationTests(unittest.TestCase):
    def test_final_predictor_sets(self):
        self.assertEqual(len(CONTROL_COLUMNS), 18)
        self.assertEqual(len(MODEL_PREDICTORS["M_ctrl"]), 18)
        self.assertEqual(len(MODEL_PREDICTORS["M_sent"]), 19)
        self.assertEqual(len(MODEL_PREDICTORS["M_att"]), 20)
        self.assertEqual(len(MODEL_PREDICTORS["M_full"]), 21)
        self.assertEqual(
            set(MODEL_PREDICTORS["M_full"]) - set(CONTROL_COLUMNS),
            {"sentiment", "wpv_lag1", "atv"},
        )
        self.assertEqual(
            FIRM_VARYING_REGRESSORS,
            ("sentiment", "wpv_lag1", "atv", "r_cc", "r_oc"),
        )
        self.assertEqual(len(CLARK_WEST_COMPARISONS), 6)
        self.assertEqual(
            CONTROL_COLUMNS[2:],
            (
                "market_excess_return", "smb", "hml", "rmw", "cma",
                "market_volatility", "jgb_10y", "jp_3m", "us_default_spread",
                "usdjpy_return", "eurjpy_return", "gbpjpy_return",
                "cpi_jp", "ip_jp", "m1_jp", "unemployment_jp",
            ),
        )

    def test_final_window_and_vocabulary(self):
        self.assertEqual(FINAL_SPEC.candidate_vocabulary_size, 88)
        self.assertEqual(FINAL_SPEC.baseline_oos_window, 504)
        self.assertEqual(FINAL_SPEC.oos_window_sensitivities, (378, 630))

if __name__ == "__main__":
    unittest.main()
