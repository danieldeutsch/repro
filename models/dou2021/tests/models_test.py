import json
import unittest
from parameterized import parameterized

from repro.models.dou2021 import OracleGSumModel
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestDou2021Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_oracle_sentence_guided(self, device: int):
        model = OracleGSumModel(device=device)

        examples = self.examples["OracleSentenceGuided"]
        inputs = [
            {
                "document": example["document"],
                "reference": example["reference"]
            }
            for example in examples
        ]
        expected_summaries = [data["summary"] for data in examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

        # Rerun but without pre-tokenized documents and references
        inputs = [
            {
                "document": " ".join(example["document"]),
                "reference": " ".join(example["reference"])
            }
            for example in examples
        ]
        expected_summaries = [data["untok_summary"] for data in examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries
