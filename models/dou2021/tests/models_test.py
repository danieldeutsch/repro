import json
import unittest
from parameterized import parameterized

from repro.models.dou2021 import OracleSentenceGSumModel, SentenceGSumModel
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestDou2021Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_oracle_sentence_guided(self, device: int):
        model = OracleSentenceGSumModel(device=device)

        examples = self.examples["OracleSentenceGuided"]
        inputs = [
            {"document": example["document"], "reference": example["reference"]}
            for example in examples
        ]
        expected_summaries = [data["summary"] for data in examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

        # Rerun but without pre-tokenized documents and references
        inputs = [
            {
                "document": " ".join(example["document"]),
                "reference": " ".join(example["reference"]),
            }
            for example in examples
        ]
        expected_summaries = [data["untok_summary"] for data in examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

    @parameterized.expand(get_testing_device_parameters())
    def test_sentence_guided(self, device: int):
        model = SentenceGSumModel(device=device)

        examples = self.examples["SentenceGuided"]
        inputs = [{"document": example["document"]} for example in examples]
        expected_summaries = [data["summary"] for data in examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

        # We expect the same output if the documents are not sentence-tokenized
        inputs = [{"document": " ".join(example["document"])} for example in examples]
        expected_summaries = [data["summary"] for data in examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries
