import json
import pytest
import unittest
from parameterized import parameterized

from repro.models.chen2020 import LERC, MOCHAEvaluationMetric
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestChen2020Models(unittest.TestCase):
    def setUp(self) -> None:
        # These examples were taken from the "qaeval" unit tests
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_lerc_regression(self, device: int):
        model = LERC(device=device)
        examples = self.examples["lerc"]
        inputs = [
            {
                "context": example["context"],
                "question": example["question"],
                "reference": example["reference"],
                "candidate": example["candidate"],
            }
            for example in examples
        ]
        expected = [example["score"] for example in examples]
        actual = model.predict_batch(inputs)
        assert expected == pytest.approx(actual, abs=1e-4)

    def test_mocha_eval_regression(self):
        metric = MOCHAEvaluationMetric()
        examples = self.examples["evaluation"]
        inputs = examples["inputs"]
        expected = examples["metrics"]
        actual = metric.predict_batch(inputs)
        assert actual == expected
