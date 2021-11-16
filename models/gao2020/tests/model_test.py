import json
import unittest

from repro.models.gao2020 import SUPERT
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal

from . import FIXTURES_ROOT


class TestGao2020Models(unittest.TestCase):
    def setUp(self) -> None:
        # These examples were taken from the SUPERT repository
        self.topic1 = json.load(open(f"{FIXTURES_ROOT}/topic_1.json", "r"))

        self.multiling2011_examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.multiling2011_expected = json.load(
            open(f"{FIXTURES_ROOT}/expected.json", "r")
        )

    def test_supert_topic_1(self):
        # The expected scores weren't given in the original repo, so
        # this is a regression test
        metric = SUPERT()

        inputs = [
            {"sources": self.topic1["documents"], "candidate": candidate}
            for candidate in self.topic1["candidates"]
        ]
        expected_macro = self.topic1["macro"]
        expected_micro = self.topic1["micro"]

        actual_macro, actual_micro = metric.predict_batch(inputs)
        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_supert_multiling(self):
        # This is a regression test
        metric = SUPERT()

        inputs = [
            {"candidate": inp["candidate"], "sources": inp["sources"]}
            for inp in self.multiling2011_examples
        ]
        expected_macro = self.multiling2011_expected["macro"]
        expected_micro = self.multiling2011_expected["micro"]
        actual_macro, actual_micro = metric.predict_batch(inputs)
        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
