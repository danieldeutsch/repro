import pytest
import unittest
from parameterized import parameterized

from repro.models.durmus2020 import FEQA
from repro.testing import get_testing_device_parameters


class TestDurmus2020Models(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_feqa_example(self, device: int):
        # These examples are taken from https://github.com/esdurmus/feqa/blob/master/run_feqa.ipynb
        model = FEQA(device=device)
        inputs = [
            {
                "candidate": "The world's oldest person died in 1898",
                "sources": [
                    "The world's oldest person has died a few weeks after celebrating her 117th birthday. Born on March 5, 1898, the greatgrandmother had lived through two world wars, the invention of the television and the first successful powered aeroplane."
                ],
            },
            {
                "candidate": "The world's oldest person died after her 117th birthday",
                "sources": [
                    "The world's oldest person has died a few weeks after celebrating her 117th birthday. Born on March 5, 1898, the greatgrandmother had lived through two world wars, the invention of the television and the first successful powered aeroplane."
                ],
            },
        ]

        _, actual_micro = model.predict_batch(inputs)
        assert len(actual_micro) == 2
        assert actual_micro[0]["feqa"] == pytest.approx(0.674074074074074, abs=1e-4)
        # This example is slightly off from the expected version value (https://github.com/esdurmus/feqa/blob/master/run_feqa.ipynb)
        # due to the different en_core_web_sm model. The original expected value was 0.8875
        assert actual_micro[1]["feqa"] == pytest.approx(0.85, abs=1e-4)
