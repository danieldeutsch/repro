import json
import unittest
from parameterized import parameterized

from repro.models.zhang2020 import BERTScore
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestZhang2020Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))
        self.bertscore_examples = json.load(
            open(f"{FIXTURES_ROOT}/bertscore-unittests.json", "r")
        )

    @parameterized.expand(get_testing_device_parameters())
    def test_bertscore(self, device: int):
        model = BERTScore(device=device)
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected["macro"]
        expected_micro = self.expected["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_bertscore_unittests(self, device: int):
        # This tests the examples in the bert_score repo unit tests
        model = BERTScore(device=device)
        inputs = self.bertscore_examples["inputs"]
        expected_micro = self.bertscore_examples["micro"]
        _, actual_micro = model.predict_batch(inputs)

        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
