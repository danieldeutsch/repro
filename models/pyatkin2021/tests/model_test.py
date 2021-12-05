import json
import unittest
from parameterized import parameterized

from repro.models.pyatkin2021 import RoleQuestionGenerator
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestPyatkin2021Model(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_role_question_generator_regression(self, device: int):
        # Tests some debug examples from the original repository
        model = RoleQuestionGenerator(device=device)

        inputs = self.expected["inputs"]
        expected_outputs = self.expected["outputs"]
        actual_outputs = model.predict_batch(inputs)
        assert actual_outputs == expected_outputs
