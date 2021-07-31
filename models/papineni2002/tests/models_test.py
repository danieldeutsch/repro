import json
import unittest

from repro.models.papineni2002 import SentBLEU
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal

from . import FIXTURES_ROOT


class TestPapineni2002Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

    def test_sent_bleu(self):
        model = SentBLEU()
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
