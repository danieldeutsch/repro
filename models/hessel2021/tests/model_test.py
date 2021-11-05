import json
import unittest
from parameterized import parameterized

from repro.models.hessel2021 import CLIPScore
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestHessel2021Models(unittest.TestCase):
    def setUp(self) -> None:
        # These examples were taken from the CLIPScore readme
        self.good_examples = json.load(open(f"{FIXTURES_ROOT}/good.json", "r"))
        self.bad_examples = json.load(open(f"{FIXTURES_ROOT}/bad.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_clipscore_good_referenceless(self, device: int):
        device_str = "cpu" if device == -1 else "gpu"

        model = CLIPScore(device=device)
        inputs = [
            {
                "candidate": self.good_examples["candidates"][i],
                "image_file": f"{FIXTURES_ROOT}/{self.good_examples['image_files'][i]}",
            }
            for i in range(len(self.good_examples["candidates"]))
        ]

        expected_macro = self.good_examples["metrics"]["referenceless"][device_str][
            "macro"
        ]
        expected_micro = self.good_examples["metrics"]["referenceless"][device_str][
            "micro"
        ]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_clipscore_good_reference_based(self, device: int):
        device_str = "cpu" if device == -1 else "gpu"

        model = CLIPScore(device=device)
        inputs = [
            {
                "candidate": self.good_examples["candidates"][i],
                "image_file": f"{FIXTURES_ROOT}/{self.good_examples['image_files'][i]}",
                "references": self.good_examples["references"][i],
            }
            for i in range(len(self.good_examples["candidates"]))
        ]

        expected_macro = self.good_examples["metrics"]["reference_based"][device_str][
            "macro"
        ]
        expected_micro = self.good_examples["metrics"]["reference_based"][device_str][
            "micro"
        ]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_clipscore_bad_reference_based(self, device: int):
        device_str = "cpu" if device == -1 else "gpu"

        model = CLIPScore(device=device)
        inputs = [
            {
                "candidate": self.bad_examples["candidates"][i],
                "image_file": f"{FIXTURES_ROOT}/{self.bad_examples['image_files'][i]}",
                "references": self.bad_examples["references"][i],
            }
            for i in range(len(self.bad_examples["candidates"]))
        ]

        expected_macro = self.bad_examples["metrics"]["reference_based"][device_str][
            "macro"
        ]
        expected_micro = self.bad_examples["metrics"]["reference_based"][device_str][
            "micro"
        ]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
