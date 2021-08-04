import json
import unittest
from parameterized import parameterized

from repro.models.kryscinski2019 import FactCC, FactCCX
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestKryscinski2019Models(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )

    @parameterized.expand(get_testing_device_parameters())
    def test_factcc_regression(self, device: int):
        model = FactCC(device=device)
        inputs = [
            {
                "candidate": example["candidate"],
                "sources": [example["sources"][0]],
            }
            for example in self.examples
        ]
        expected_macro = self.expected["factcc"]["macro"]
        expected_micro = self.expected["factcc"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_factccx_regression(self, device: int):
        model = FactCCX(device=device)
        inputs = [
            {
                "candidate": example["candidate"],
                "sources": [example["sources"][0]],
            }
            for example in self.examples
        ]
        expected_macro = self.expected["factccx"]["macro"]
        expected_micro = self.expected["factccx"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
