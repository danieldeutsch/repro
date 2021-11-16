import json
import unittest

from repro.models.denkowski2014 import METEOR
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal

from . import FIXTURES_ROOT


class TestDenkowski2014Models(unittest.TestCase):
    def setUp(self) -> None:
        self.multiling2011_examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.multiling2011_expected = json.load(
            open(f"{FIXTURES_ROOT}/expected.json", "r")
        )

    def test_meteor_regression(self):
        metric = METEOR()
        # All inputs must have the same number of references and the
        # minimum number in this dataset is 2, so we take only the
        # first 2 for each one
        inputs = [
            {
                "candidate": inp["candidate"],
                "references": inp["references"][:2],
            }
            for inp in self.multiling2011_examples
        ]
        expected_macro = self.multiling2011_expected["macro"]
        expected_micro = self.multiling2011_expected["micro"]

        actual_macro, actual_micro = metric.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
