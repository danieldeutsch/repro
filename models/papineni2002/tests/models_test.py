import json
import pytest
import unittest

from repro.models.papineni2002 import BLEU, SentBLEU
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal

from . import FIXTURES_ROOT


class TestPapineni2002Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

    def test_sent_bleu(self):
        model = SentBLEU()
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected["sentbleu"]["macro"]
        expected_micro = self.expected["sentbleu"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    def test_bleu(self):
        model = BLEU()
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected["bleu"]["macro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(actual_micro) == 0

    def test_bleu_sacrebleu_tests(self):
        # Tests an example from the sacrebleu unit tests
        model = BLEU()
        inputs = [
            {
                "candidate": "The dog bit the man.",
                "references": ["The dog bit the man.", "The dog had bit the man."],
            },
            {
                "candidate": "It wasn't surprising.",
                "references": ["It was not unexpected.", "No one was surprised."],
            },
            {
                "candidate": "The man had just bitten him.",
                "references": ["The man bit him first.", "The man had bitten the dog."],
            },
        ]
        expected = 48.530827
        actual_macro, actual_micro = model.predict_batch(inputs)
        assert actual_macro["bleu"] == pytest.approx(expected, abs=1e-4)
        assert len(actual_micro) == 0

    def test_bleu_unequal_references_regression(self):
        # Tests an example when the candidates have an unequal number of references
        model = BLEU()
        inputs = [
            {
                "candidate": "The dog bit the man.",
                "references": ["The dog bit the man.", "The dog had bit the man."],
            },
            {
                "candidate": "It wasn't surprising.",
                "references": ["No one was surprised."],
            },
            {
                "candidate": "The man had just bitten him.",
                "references": [
                    "The man bit him first.",
                    "The man had bitten the dog.",
                    "The man bit the dog before.",
                ],
            },
        ]
        expected = 47.63997460581784
        actual_macro, actual_micro = model.predict_batch(inputs)
        assert actual_macro["bleu"] == pytest.approx(expected, abs=1e-4)
        assert len(actual_micro) == 0
