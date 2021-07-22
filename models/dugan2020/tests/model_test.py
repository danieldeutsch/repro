import json
import unittest
from parameterized import parameterized

from repro.models.dugan2020 import RoFTRecipeGenerator
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestDugan2020(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_outputs = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_roft_recipe_generator(self, device: int):
        model = RoFTRecipeGenerator(device=device)
        predictions = model.predict_batch(self.expected_outputs)
        device = "cpu" if device == -1 else "gpu"
        recipes = [inp["recipe"][device] for inp in self.expected_outputs]
        assert predictions == recipes
