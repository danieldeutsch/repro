import json
import unittest

from repro.models.sacrerouge import ROUGE

from . import FIXTURES_ROOT


class TestSacreROUGE(unittest.TestCase):
    def setUp(self) -> None:
        self.image = "sacrerouge"
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json"))

    def test_rouge(self):
        metric = ROUGE()
        examples = self.examples["rouge"]
        inputs = [
            {"summary": example["summary"], "references": [example["reference"]]}
            for example in examples["input"]
        ]
        expected = examples["output"]
        actual = metric.predict_batch(inputs)
        assert actual == expected
