import json
import unittest
from parameterized import parameterized

from repro.models.goyal2020 import DAE
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


def _get_testing_parameters():
    devices = get_testing_device_parameters()
    # `None` will be used to indicate the default model
    models = [None, "dae_basic", "dae_w_syn", "dae_w_syn_hallu"]

    parameters = []
    for (device,) in devices:
        for model in models:
            parameters.append((device, model))
    return parameters


class TestGoyal2020Models(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )

    @parameterized.expand(_get_testing_parameters())
    def test_dae_regression(self, device: int, model_name: str):
        if model_name is None:
            # Use the default model, which is "dae_w_syn" and
            # set the model name for usage later on
            model = DAE(device=device)
            model_name = "dae_w_syn"
        else:
            model = DAE(model=model_name, device=device)

        # The inputs to DAE are single sentences, so we take the
        # first sentence of the candidate and first sentence of the
        # first source document as input
        inputs = [
            {
                "candidate": example["candidate"][0],
                "sources": [example["sources"][0][0]],
            }
            for example in self.examples
        ]
        expected_macro = self.expected[model_name]["macro"]
        expected_micro = self.expected[model_name]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
