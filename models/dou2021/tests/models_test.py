import json
import unittest
from parameterized import parameterized

from repro.models.dou2021 import OracleGSumModel
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestDou2021Models(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_output = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_oracle_sentence_guided(self, device: int):
        model = OracleGSumModel(device=device)
        expected_output = self.expected_output["OracleSentenceGuided"]
        expected_summaries = [inp["summary"] for inp in expected_output]
        summaries = model.predict_batch(expected_output)
        assert summaries == expected_summaries
