import json
import unittest
from parameterized import parameterized

from repro.models.gupta2020 import NeuralModuleNetwork
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestGupta2020(unittest.TestCase):
    def setUp(self) -> None:
        # The examples were taken from the AllenNLP demo for this model
        self.expected_outputs = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_neural_module_network(self, device: int):
        model = NeuralModuleNetwork(device=device)
        predictions = model.predict_batch(self.expected_outputs)
        answers = [inp["answer"] for inp in self.expected_outputs]
        assert predictions == answers
