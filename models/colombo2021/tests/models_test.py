import json
import unittest
from parameterized import parameterized

from repro.models.colombo2021 import BaryScore, DepthScore, InfoLM
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestColombo2021Models(unittest.TestCase):
    def setUp(self) -> None:
        # Load the inputs
        self.inputs = []
        with open(f"{FIXTURES_ROOT}/refs.txt", "r") as f_refs:
            with open(f"{FIXTURES_ROOT}/hyps.txt", "r") as f_hyps:
                for reference, hyp in zip(f_refs, f_hyps):
                    self.inputs.append(
                        {"candidate": hyp.strip(), "references": [reference.strip()]}
                    )

        # Load the expected outputs
        with open(f"{FIXTURES_ROOT}/expected.json", "r") as f:
            self.expected = json.load(f)

    @parameterized.expand(get_testing_device_parameters())
    def test_infolm(self, device: int):
        model = InfoLM(device=device)
        expected_macro = self.expected["infolm"]["macro"]
        expected_micro = self.expected["infolm"]["micro"]
        actual_macro, actual_micro = model.predict_batch(self.inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_baryscore(self, device: int):
        model = BaryScore(device=device)
        device = "cpu" if device == -1 else "gpu"
        expected_macro = self.expected["baryscore"][device]["macro"]
        expected_micro = self.expected["baryscore"][device]["micro"]
        actual_macro, actual_micro = model.predict_batch(self.inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_depthscore(self, device: int):
        model = DepthScore(device=device)
        device = "cpu" if device == -1 else "gpu"
        expected_macro = self.expected["depthscore"][device]["macro"]
        expected_micro = self.expected["depthscore"][device]["micro"]
        actual_macro, actual_micro = model.predict_batch(self.inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
