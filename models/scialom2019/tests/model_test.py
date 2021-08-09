import json
import unittest

from repro.models.scialom2019 import SummaQA
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal

from . import FIXTURES_ROOT


class TestScialom2019Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))
        self.summaqa_examples = json.load(
            open(f"{FIXTURES_ROOT}/summaqa-examples.json", "r")
        )

    def test_summaqa_examples(self):
        # Tests the examples provided in the Github repo
        model = SummaQA()
        inputs = self.summaqa_examples["inputs"]
        expected_micro = self.summaqa_examples["outputs"]
        _, actual_micro = model.predict_batch(inputs)

        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_summaqa_regression(self):
        model = SummaQA()
        # We only take the first source since SummaQA only supports one of each
        inputs = [
            {
                "candidate": inp["candidate"],
                "sources": [inp["sources"][0]],
            }
            for inp in self.examples
        ]
        expected_macro = self.expected["macro"]
        expected_micro = self.expected["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
