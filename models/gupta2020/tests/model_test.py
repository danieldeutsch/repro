import json
import unittest
from parameterized import parameterized

from repro.models.gupta2020 import Gupta2020
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestGupta2020(unittest.TestCase):
    def setUp(self) -> None:
        # The examples were taken from the AllenNLP demo for this model
        self.expected_outputs = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_gupta2020(self, device: int):
        model = Gupta2020(device=device)
        predictions = model.predict_batch(self.expected_outputs)
        answers = [inp["answer"] for inp in self.expected_outputs]
        assert predictions == answers
