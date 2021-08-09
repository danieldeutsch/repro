import json
import unittest

from repro.models.kane2020 import NUBIA
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal

from . import FIXTURES_ROOT


class TestKane2020(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))
        self.nubia_examples = json.load(
            open(f"{FIXTURES_ROOT}/nubia-examples.json", "r")
        )

    def test_nubia(self):
        # Tests the example from the Github repo. The "grammar_ref" and
        # "grammar_hyp" features aren't included in the documentation. The actual
        # score also changed, presumably because of the additional features
        model = NUBIA()
        inputs = self.nubia_examples["inputs"]
        expected_micro = self.nubia_examples["outputs"]
        _, actual_micro = model.predict_batch(inputs)

        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_nubia_regression(self):
        model = NUBIA()
        # Nubia only supports single references, so we take just the first one
        inputs = [
            {
                "candidate": example["candidate"],
                "references": [example["references"][0]],
            }
            for example in self.examples
        ]
        expected_macro = self.expected["macro"]
        expected_micro = self.expected["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-3)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-3)
