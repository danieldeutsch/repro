import unittest
from typing import List

from repro.commands.predict import predict_with_model
from repro.models import Model


@Model.register("test-model", exist_ok=True)
class _Model(Model):
    """
    A simple model which reverses `input1` then concatenates `input2` or
    directly concatenates them, depending on the `mode` parameter
    Anything passed in **kwargs will also be added to each instance

    """

    def predict(self, input1: str, input2: int, **kwargs):
        return self.predict_batch([{"input1": input1, "input2": input2}], **kwargs)[0]

    def predict_batch(self, inputs: List, mode : str = "reverse", **kwargs):
        if mode == "reverse":
            return [inp["input1"][::-1] + str(inp["input2"]) for inp in inputs]
        elif mode == "concatenate":
            return [inp["input1"] + str(inp["input2"]) for inp in inputs]
        else:
            raise Exception(f"Unknown mode: {mode}")


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

    def test_predict_with_kwargs(self):
        model = _Model()
        instances = [
            {"input1": "abc", "input2": 1},
            {"input1": "def", "input2": 2},
            {"input1": "ghi", "input2": 3, "input3": "extra"},
        ]

        predictions = predict_with_model(model, instances, kwargs={"mode": "concatenate"})
        assert predictions == ["abc1", "def2", "ghi3"]

        predictions = predict_with_model(model, instances, kwargs='{"mode": "concatenate"}')
        assert predictions == ["abc1", "def2", "ghi3"]
