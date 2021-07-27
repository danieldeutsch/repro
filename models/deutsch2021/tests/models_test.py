import json
import pytest
import unittest
from parameterized import parameterized

from repro.models.deutsch2021 import (
    QAEvalQuestionAnsweringModel,
    QAEvalQuestionGenerationModel,
)
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestDeutsch2021Models(unittest.TestCase):
    def setUp(self) -> None:
        # These examples were taken from the "qaeval" unit tests
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_question_generation(self, device: int):
        model = QAEvalQuestionGenerationModel(device=device)
        examples = self.examples["generation"]
        inputs = [
            {
                "context": example["context"],
                "start": example["start"],
                "end": example["end"],
            }
            for example in examples
        ]
        expected = [example["question"] for example in examples]
        actual = model.predict_batch(inputs)
        assert actual == expected

    @parameterized.expand(get_testing_device_parameters())
    def test_question_answering(self, device: int):
        model = QAEvalQuestionAnsweringModel(device=device)
        examples = self.examples["answering"]
        inputs = [
            {
                "context": example["context"],
                "question": example["question"],
            }
            for example in examples
        ]
        expected_dicts = [
            {
                "prediction": example["answer"],
                "probability": example["probability"],
                "null_probability": example["null_probability"],
                "start": example["start"],
                "end": example["end"],
            }
            for example in examples
        ]
        expected_answers = [
            example["answer"]
            if example["probability"] > example["null_probability"]
            else None
            for example in examples
        ]
        # Check just the answers
        actual = model.predict_batch(inputs)
        assert actual == expected_answers

        # Check the full dict
        actual = model.predict_batch(inputs, return_dicts=True)
        for a, e in zip(actual, expected_dicts):
            assert a["prediction"] == e["prediction"]
            assert pytest.approx(a["null_probability"], e["null_probability"], abs=1e-8)
            assert pytest.approx(a["probability"], e["probability"], abs=1e-8)
            assert a["start"] == e["start"]
            assert a["end"] == e["end"]
