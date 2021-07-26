import json
import pytest
import unittest

from repro.data.dataset_readers import HuggingfaceDatasetsDatasetReader
from repro.data.dataset_readers.datasets import hf_dataset_exists_locally
from tests import FIXTURES_ROOT


class TestHuggingfaceDatasetsDatasetReader(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_output = json.load(
            open(f"{FIXTURES_ROOT}/hf-datasets/expected-output.json", "r")
        )

    @pytest.mark.skipif(
        not hf_dataset_exists_locally("cnn_dailymail"),
        reason="CNN/DailyMail dataset does not exist",
    )
    def test_cnn_dailymail(self):
        reader = HuggingfaceDatasetsDatasetReader("cnn_dailymail", "test")
        instances = reader.read()
        assert len(instances) == 11490

        expected_instances = self.expected_output["cnn_dailymail"]
        for actual, expected in zip(instances, expected_instances):
            assert actual == expected

    @pytest.mark.skipif(
        not hf_dataset_exists_locally("xsum"),
        reason="XSum dataset does not exist",
    )
    def test_xsum(self):
        reader = HuggingfaceDatasetsDatasetReader("xsum", "test")
        instances = reader.read()
        assert len(instances) == 11334

        expected_instances = self.expected_output["xsum"]
        for actual, expected in zip(instances, expected_instances):
            assert actual == expected

    @pytest.mark.skipif(
        not hf_dataset_exists_locally("scientific_papers", "arxiv"),
        reason="scientific_papers/arxiv dataset does not exist",
    )
    def test_arxiv(self):
        # We don't check for the parsing correctness because the documents are too long
        reader = HuggingfaceDatasetsDatasetReader("scientific_papers/arxiv", "test")
        instances = reader.read()
        assert len(instances) == 6440

    @pytest.mark.skipif(
        not hf_dataset_exists_locally("scientific_papers", "pubmed"),
        reason="scientific_papers/pubmed dataset does not exist",
    )
    def test_pubmed(self):
        # We don't check for the parsing correctness because the documents are too long
        reader = HuggingfaceDatasetsDatasetReader("scientific_papers/pubmed", "test")
        instances = reader.read()
        assert len(instances) == 6658

    @pytest.mark.skipif(
        not hf_dataset_exists_locally("squad_v2", "squad_v2"),
        reason="squad_v2 dataset does not exist",
    )
    def test_squad_v2(self):
        # squad_v2 only has validation, not test
        reader = HuggingfaceDatasetsDatasetReader("squad_v2", "validation")
        instances = reader.read()
        assert len(instances) == 11873

        expected_instances = self.expected_output["squad_v2"]
        for actual, expected in zip(instances, expected_instances):
            assert actual == expected
