import json
from pathlib import Path
import unittest
import pandas as pd

from attention_sentiment_thesis.sentiment import (
    build_presence_matrix, aggregate_daily_sentiment,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_expression_events.json"

class SentimentTests(unittest.TestCase):
    def setUp(self):
        content = json.loads(FIXTURE.read_text())
        self.events = pd.DataFrame(content["events"])
        self.vocabulary = content["vocabulary"]

    def test_presence_and_daily_aggregation(self):
        matrix = build_presence_matrix(self.events, self.vocabulary)
        self.assertEqual(matrix.loc[0, "expr_001"], 1)
        polarities = pd.DataFrame({
            "scoring_year": [2024, 2024],
            "expression_id": ["expr_001", "expr_002"],
            "polarity": [0.3, -0.1],
        })
        daily = aggregate_daily_sentiment(self.events, polarities)
        # First event: 0.2; second: 0.3; ineligible event contributes nothing.
        self.assertEqual(len(daily), 1)
        self.assertAlmostEqual(daily.loc[0, "sentiment"], 0.5)

if __name__ == "__main__":
    unittest.main()

