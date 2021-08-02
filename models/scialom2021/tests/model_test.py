import json
import unittest
from parameterized import parameterized

from repro.models.scialom2021 import (
    QuestEval,
    QuestEvalForSummarization,
    QuestEvalForSimplification,
)
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestScialom2021Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))
        self.questeval_examples = json.load(
            open(f"{FIXTURES_ROOT}/questeval-unittests.json", "r")
        )

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_summarization_reference_and_source(self, device: int):
        model = QuestEvalForSummarization(device=device)
        # We only take the first reference and source since QuestEval only supports one of each
        inputs = [
            {
                "candidate": inp["candidate"],
                "references": [inp["references"][0]],
                "sources": [inp["sources"][0]],
            }
            for inp in self.examples
        ]
        expected_macro = self.expected["reference_and_source"]["macro"]
        expected_micro = self.expected["reference_and_source"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_summarization_reference_only(self, device: int):
        model = QuestEvalForSummarization(device=device)
        # We only take the first reference and source since QuestEval only supports one of each
        inputs = [
            {"candidate": inp["candidate"], "references": [inp["references"][0]]}
            for inp in self.examples
        ]
        expected_macro = self.expected["reference_only"]["macro"]
        expected_micro = self.expected["reference_only"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_summarization_source_only(self, device: int):
        model = QuestEvalForSummarization(device=device)
        # We only take the first reference and source since QuestEval only supports one of each
        inputs = [
            {"candidate": inp["candidate"], "sources": [inp["sources"][0]]}
            for inp in self.examples
        ]
        expected_macro = self.expected["source_only"]["macro"]
        expected_micro = self.expected["source_only"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_questeval_summarization_invalid_kwargs(self):
        model = QuestEvalForSummarization()
        with self.assertRaises(Exception):
            model.predict(candidate="Candidate", sources=["Source"], task="text2text")

        with self.assertRaises(Exception):
            model.predict(candidate="Candidate", sources=["Source"], do_weighter=False)

        with self.assertRaises(Exception):
            model.predict_batch([], task="text2text")

        with self.assertRaises(Exception):
            model.predict_batch([], do_weighter=False)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_unittests_reference_and_source(self, device: int):
        # Tests an example from the QuestEval repo with the reference and source.
        # The candidate and source and swapped in the repo
        model = QuestEval(device=device)
        inp = self.questeval_examples["summarization"]["reference_and_source"]["input"]
        expected = self.questeval_examples["summarization"]["reference_and_source"][
            "output"
        ]
        actual = model.predict(**inp)
        assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_unittests_reference_only(self, device: int):
        # Tests an example from the QuestEval repo with the reference only.
        # The candidate and source and swapped in the repo
        model = QuestEval(device=device)
        inp = self.questeval_examples["summarization"]["reference_only"]["input"]
        expected = self.questeval_examples["summarization"]["reference_only"]["output"]
        actual = model.predict(**inp)
        assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_unittests_source_only(self, device: int):
        # Tests an example from the QuestEval repo with the source only.
        # The candidate and source and swapped in the repo
        model = QuestEval(device=device)
        inp = self.questeval_examples["summarization"]["source_only"]["input"]
        expected = self.questeval_examples["summarization"]["source_only"]["output"]
        actual = model.predict(**inp)
        assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_simplification_reference_and_source_regression(
        self, device: int
    ):
        # Tests an example from the QuestEval repo but applied to simplification
        model = QuestEvalForSimplification(device=device)
        inp = self.questeval_examples["simplification"]["reference_and_source"]["input"]
        expected = self.questeval_examples["simplification"]["reference_and_source"][
            "output"
        ]
        actual = model.predict(**inp)
        assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_simplification_reference_only_regression(self, device: int):
        # Tests an example from the QuestEval repo but applied to simplification
        model = QuestEvalForSimplification(device=device)
        inp = self.questeval_examples["simplification"]["reference_only"]["input"]
        expected = self.questeval_examples["simplification"]["reference_only"]["output"]
        actual = model.predict(**inp)
        assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_questeval_simplification_source_only_regression(self, device: int):
        # Tests an example from the QuestEval repo but applied to simplification
        model = QuestEvalForSimplification(device=device)
        inp = self.questeval_examples["simplification"]["source_only"]["input"]
        expected = self.questeval_examples["simplification"]["source_only"]["output"]
        actual = model.predict(**inp)
        assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_questeval_simplification_invalid_kwargs(self):
        model = QuestEvalForSimplification()
        with self.assertRaises(Exception):
            model.predict(candidate="Candidate", sources=["Source"], task="text2text")

        with self.assertRaises(Exception):
            model.predict(candidate="Candidate", sources=["Source"], do_BERTScore=False)

        with self.assertRaises(Exception):
            model.predict_batch([], task="text2text")

        with self.assertRaises(Exception):
            model.predict_batch([], do_BERTScore=False)
