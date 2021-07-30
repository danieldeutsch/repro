import json
import unittest
from parameterized import parameterized

from repro.models.sellam2020 import BLEURT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestSellam2020Models(unittest.TestCase):
    def setUp(self) -> None:
        self.bleurt_examples = json.load(
            open(f"{FIXTURES_ROOT}/bleurt-unittests.json", "r")
        )

    @parameterized.expand(get_testing_device_parameters())
    def test_bleurt_unittest_examples(self, device: int):
        # Tests the examples from the BLEURT repository unit tests
        model = BLEURT(model="bleurt/bleurt/test_checkpoint", device=device)
        inputs = self.bleurt_examples["inputs"]
        expected_macro = self.bleurt_examples["output"]["macro"]
        expected_micro = self.bleurt_examples["output"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
