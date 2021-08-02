import json
import unittest
from parameterized import parameterized

from repro.models.zhao2019 import MoverScore, MoverScoreForSummarization
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestZhao2019Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_moverscore_regression(self, device: int):
        model = MoverScore(device=device)
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected["default"]["macro"]
        expected_micro = self.expected["default"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_moverscore_for_summarization_regression(self, device: int):
        model = MoverScoreForSummarization(device=device)
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected["summarization"]["macro"]
        expected_micro = self.expected["summarization"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_moverscore_for_summarization_invalid_kwargs(self):
        model = MoverScoreForSummarization()
        with self.assertRaises(Exception):
            model.predict(
                candidate="Candidate", references=["References"], use_stopwords=False
            )

        with self.assertRaises(Exception):
            model.predict_batch([], use_stopwords=False)
