import json
import pytest
import unittest
from parameterized import parameterized

from repro.models.deutsch2021 import (
    QAEval,
    QAEvalQuestionAnsweringModel,
    QAEvalQuestionGenerationModel,
)
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestDeutsch2021Models(unittest.TestCase):
    def setUp(self) -> None:
        # These examples were taken from the "qaeval" unit tests
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json", "r"))

        self.multiling2011_examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )

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
            assert e["null_probability"] == pytest.approx(
                a["null_probability"], abs=1e-4
            )
            assert e["probability"] == pytest.approx(a["probability"], abs=1e-4)
            assert a["start"] == e["start"]
            assert a["end"] == e["end"]

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_qaeval(self, device: int):
        model = QAEval(device=device)
        inputs = self.multiling2011_examples
        expected_macro = self.examples["metric"]["macro"]
        expected_micro = self.examples["metric"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_qaeval_qa_pairs(self, device: int):
        model = QAEval(device=device)
        inputs = [
            {
                "candidate": "Dan walked to the bakery this morning.",
                "references": ["Dan went to buy scones earlier this morning."],
            },
            {
                "candidate": "He bought some scones today",
                "references": ["Dan went to buy scones earlier this morning."],
            },
        ]

        macro, micro, qa_pairs_lists = model.predict_batch(inputs, return_qa_pairs=True)

        assert len(qa_pairs_lists) == 2  # Number of inputs

        qa_pairs_list = qa_pairs_lists[0]
        assert len(qa_pairs_list) == 1  # Number of references
        qa_pairs = qa_pairs_list[0]
        assert len(qa_pairs) == 2
        assert (
            qa_pairs[0]["question"]["question"]
            == "Who went to buy scones earlier this morning?"
        )
        assert qa_pairs[0]["prediction"]["prediction"] == "Dan"
        assert qa_pairs[0]["prediction"]["start"] == 0
        assert qa_pairs[0]["prediction"]["end"] == 3
        assert qa_pairs[0]["prediction"]["is_answered"] == 1.0
        assert qa_pairs[0]["prediction"]["em"] == 1.0
        assert qa_pairs[0]["prediction"]["f1"] == 1.0
        self.assertAlmostEqual(
            qa_pairs[0]["prediction"]["lerc"], 5.035197734832764, places=4
        )
        assert (
            qa_pairs[1]["question"]["question"]
            == "What did Dan go to buy earlier this morning?"
        )
        assert qa_pairs[1]["prediction"]["prediction"] == "bakery"
        assert qa_pairs[1]["prediction"]["start"] == 18
        assert qa_pairs[1]["prediction"]["end"] == 24
        assert qa_pairs[1]["prediction"]["is_answered"] == 1.0
        assert qa_pairs[1]["prediction"]["em"] == 0.0
        assert qa_pairs[1]["prediction"]["f1"] == 0.0
        self.assertAlmostEqual(
            qa_pairs[1]["prediction"]["lerc"], 1.30755615234375, places=4
        )

        qa_pairs_list = qa_pairs_lists[1]
        assert len(qa_pairs_list) == 1  # Number of references
        qa_pairs = qa_pairs_list[0]
        assert len(qa_pairs) == 2
        assert (
            qa_pairs[0]["question"]["question"]
            == "Who went to buy scones earlier this morning?"
        )
        assert qa_pairs[0]["prediction"]["prediction"] == "He"
        assert qa_pairs[0]["prediction"]["start"] == 0
        assert qa_pairs[0]["prediction"]["end"] == 2
        assert qa_pairs[0]["prediction"]["is_answered"] == 0.0
        assert qa_pairs[0]["prediction"]["em"] == 0.0
        assert qa_pairs[0]["prediction"]["f1"] == 0.0
        assert qa_pairs[0]["prediction"]["lerc"] == 0.0
        assert (
            qa_pairs[1]["question"]["question"]
            == "What did Dan go to buy earlier this morning?"
        )
        assert qa_pairs[1]["prediction"]["prediction"] == "scones"
        assert qa_pairs[1]["prediction"]["start"] == 15
        assert qa_pairs[1]["prediction"]["end"] == 21
        assert qa_pairs[1]["prediction"]["is_answered"] == 1.0
        assert qa_pairs[1]["prediction"]["em"] == 1.0
        assert qa_pairs[1]["prediction"]["f1"] == 1.0
        self.assertAlmostEqual(
            qa_pairs[1]["prediction"]["lerc"], 4.984881401062012, places=4
        )
