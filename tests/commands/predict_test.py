import unittest
from typing import List

from repro.commands.predict import predict_with_model
from repro.models import Model


@Model.register("test-model", exist_ok=True)
class _Model(Model):
    """A simple model which reverses `input1` then concatenates `input2`."""

    def predict(self, input1: str, input2: int, *args, **kwargs):
        return self.predict_batch([{"input1": input1, "input2": input2}])[0]

    def predict_batch(self, inputs: List, *args, **kwargs):
        return [inp["input1"][::-1] + str(inp["input2"]) for inp in inputs]


class TestPredict(unittest.TestCase):
    def test_predict_with_model(self):
        """Tests to make sure the predictions are correct."""
        # Extra values in the input dicts are ignored
        model = _Model()
        instances = [
            {"input1": "abc", "input2": 1},
            {"input1": "def", "input2": 2},
            {"input1": "ghi", "input2": 3, "input3": "extra"},
        ]

        predictions = predict_with_model(model, instances)
        assert predictions == ["cba1", "fed2", "ihg3"]

    def test_predict_with_model_missing_input(self):
        """Tests to make sure prediction fails if there is a missing input."""
        model = _Model()
        instances = [
            {"input1": "abc", "input2": 1},
            {"input1": "def"},
        ]
        with self.assertRaises(Exception):
            predict_with_model(model, instances)
