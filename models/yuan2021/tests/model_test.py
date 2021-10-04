import json
import unittest
from parameterized import parameterized

from repro.models.yuan2021 import BARTScore
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


def _get_device_by_model_parameters():
    devices = get_testing_device_parameters()
    models = ["default", "cnn", "parabank"]
    return [(device, model) for device, in devices for model in models]


class TestYuan2021Model(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_bartscore_examples(self, device: int):
        # Tests the examples from their Github Repo
        model = BARTScore(device=device, model="cnn")
        expected = {"bartscore": -2.510652780532837}
        actual = model.predict("This is interesting.", ["This is fun."])
        assert_dicts_approx_equal(actual, expected, abs=1e-4)

        model = BARTScore(device=device, model="parabank")
        expected = {"bartscore": -2.336203098297119}
        actual = model.predict("This is interesting.", ["This is fun."])
        assert_dicts_approx_equal(actual, expected, abs=1e-4)

    @parameterized.expand(_get_device_by_model_parameters())
    def test_bartscore_regression(self, device: int, model_name: str):
        # Runs the metric on the MultiLing examples
        model = BARTScore(device=device, model=model_name)
        inputs = [
            {"candidate": example["candidate"], "references": example["references"]}
            for example in self.examples
        ]
        expected_macro = self.expected[model_name]["macro"]
        expected_micro = self.expected[model_name]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
