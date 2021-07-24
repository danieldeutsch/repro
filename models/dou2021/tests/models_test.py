import json
import unittest
from parameterized import parameterized

from repro.models.dou2021 import OracleSentenceGSumModel, SentenceGSumModel
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT
from ..src.commands import get_oracle_sentences, generate_summaries, sentence_split


class TestDou2021Models(unittest.TestCase):
    def setUp(self) -> None:
        self.image = "dou2021"
        # The documents and references were taken from https://github.com/icml-2020-nlp/semsim/tree/master/datasets.
        # The guidance field was taken from the provided outputs of the authors (except for the second example
        # in which the sentences were in a different order). The other fields are regression outputs
        # (i.e., we ran the code on those documents and references and saved the outputs).
        self.examples = json.load(open(f"{FIXTURES_ROOT}/examples.json", "r"))

    def test_sentence_split_regression(self):
        documents = [inp["document"] for inp in self.examples]
        expected = [inp["document_sentences"] for inp in self.examples]
        actual = sentence_split(self.image, documents)
        assert expected == actual

        references = [inp["reference"] for inp in self.examples]
        expected = [inp["reference_sentences"] for inp in self.examples]
        actual = sentence_split(self.image, references)
        assert expected == actual

    def test_get_oracle_sentences(self):
        documents = [inp["document_sentences"] for inp in self.examples]
        references = [inp["reference_sentences"] for inp in self.examples]

        expected = [inp["guidance"] for inp in self.examples]
        guidance = get_oracle_sentences(self.image, documents, references)
        assert guidance == expected

    @parameterized.expand(get_testing_device_parameters())
    def test_generate_summaries(self, device: int):
        model = "bart_sentence"
        batch_size = 16

        documents = [inp["document"] for inp in self.examples]
        guidance = [inp["guidance"] for inp in self.examples]
        expected = [inp["oracle_guided_summary"] for inp in self.examples]
        actual = generate_summaries(
            self.image, model, device, batch_size, documents, guidance
        )
        assert expected == actual

    @parameterized.expand(get_testing_device_parameters())
    def test_oracle_sentence_guided(self, device: int):
        model = OracleSentenceGSumModel(device=device)

        # First without pre-sentence-split documents and references
        inputs = [
            {"document": example["document"], "reference": example["reference"]}
            for example in self.examples
        ]
        expected_summaries = [data["oracle_guided_summary"] for data in self.examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

        # Now with pre-tokenized documents and references
        inputs = [
            {
                "document": example["document_sentences"],
                "reference": example["reference_sentences"],
            }
            for example in self.examples
        ]
        expected_summaries = [
            data["oracle_guided_summary_presplit"] for data in self.examples
        ]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

    @parameterized.expand(get_testing_device_parameters())
    def test_oracle_sentence_guided_with_guidance(self, device: int):
        model = OracleSentenceGSumModel(device=device)

        # First without pre-sentence-split documents and references
        inputs = [
            {"document": example["document"], "guidance": " ".join(example["guidance"])}
            for example in self.examples
        ]
        expected_summaries = [data["oracle_guided_summary"] for data in self.examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

        # Now with pre-tokenized documents and guidance
        inputs = [
            {
                "document": example["document_sentences"],
                "guidance": example["guidance"],
            }
            for example in self.examples
        ]
        expected_summaries = [
            data["oracle_guided_summary_presplit"] for data in self.examples
        ]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

    @parameterized.expand(get_testing_device_parameters())
    def test_sentence_guided(self, device: int):
        model = SentenceGSumModel(device=device)

        # Original document text
        inputs = [{"document": example["document"]} for example in self.examples]
        expected_summaries = [data["guided_summary"] for data in self.examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries

        # Pre-sentence-split documents
        inputs = [
            {"document": example["document_sentences"]} for example in self.examples
        ]
        expected_summaries = [data["guided_summary_presplit"] for data in self.examples]
        summaries = model.predict_batch(inputs)
        assert summaries == expected_summaries
