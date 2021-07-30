import json
import unittest

from repro.models.sacrerouge import SRROUGE

from . import FIXTURES_ROOT


class TestSRROUGE(unittest.TestCase):
    def setUp(self) -> None:
        self.image = "sacrerouge"
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json"))

    def test_rouge(self):
        metric = SRROUGE()
        examples = self.examples["rouge"]
        inputs = [
            {"summary": example["summary"], "references": [example["reference"]]}
            for example in examples["input"]
        ]
        expected = examples["output"]
        actual = metric.predict_batch(inputs)
        assert actual == expected
