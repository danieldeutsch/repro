import unittest

from repro.common import util
from repro.models import Model


class TestUtil(unittest.TestCase):
    def test_flatten(self):
        assert util.flatten("text") == "text"
        assert util.flatten("text", separator="-") == "text"
        assert util.flatten(["1", "2"]) == "1 2"
        assert util.flatten(["1", "2"], separator="-") == "1-2"

    def test_load_model_no_required_args(self):
        @Model.register("test-model", exist_ok=True)
        class _Model(Model):
            def __init__(self, a: int = 2, b: int = 3):
                self.a = a
                self.b = b

        model = util.load_model("test-model")
        assert isinstance(model, _Model)
        assert model.a == 2
        assert model.b == 3

        model = util.load_model("test-model", '{"a": 4}')
        assert isinstance(model, _Model)
        assert model.a == 4
        assert model.b == 3

        model = util.load_model("test-model", {"a": 8})
        assert isinstance(model, _Model)
        assert model.a == 8
        assert model.b == 3

    def test_load_model_required_args(self):
        @Model.register("test-model", exist_ok=True)
        class _Model(Model):
            def __init__(self, a: int, b: int = 3):
                self.a = a
                self.b = b

        with self.assertRaises(Exception):
            util.load_model("test-model")

        model = util.load_model("test-model", '{"a": 4}')
        assert isinstance(model, _Model)
        assert model.a == 4
        assert model.b == 3
