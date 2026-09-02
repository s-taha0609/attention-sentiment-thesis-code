import json
from pathlib import Path
import unittest
import pandas as pd

import attention_sentiment_thesis
from attention_sentiment_thesis.schemas import SchemaError, validate_expression_events
from attention_sentiment_thesis.sentiment.build_dictionary import validate_candidate_vocabulary

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_expression_events.json"

class ImportsAndSchemaTests(unittest.TestCase):
    def test_import_smoke(self):
        self.assertEqual(attention_sentiment_thesis.__version__, "0.1.0")

    def test_provider_neutral_schema(self):
        content = json.loads(FIXTURE.read_text())
        events = validate_expression_events(pd.DataFrame(content["events"]))
        self.assertEqual(len(events), 3)
        self.assertEqual(len(validate_candidate_vocabulary(content["vocabulary"])), 88)

    def test_missing_schema_field_fails(self):
        with self.assertRaises(SchemaError):
            validate_expression_events(pd.DataFrame({"firm_id": ["firm_a"]}))

if __name__ == "__main__":
    unittest.main()

