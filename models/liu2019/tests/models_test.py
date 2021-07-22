import json
import unittest

from parameterized import parameterized

from repro.models.liu2019 import BertSumExt, BertSumExtAbs, TransformerAbs
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestLiu2019Models(unittest.TestCase):
    def setUp(self) -> None:
        # The example documents were taken from the PreSumm repo:
        # https://github.com/nlpyang/PreSumm/blob/dev/raw_data/temp.raw_src
        self.expected_output = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_bert_sum_ext_regression(self, device: int):
        model = BertSumExt(device=device)

        # Test when the document is pre-sentence split
        document = self.expected_output["BertSumExt"]["document"]
        expected_summary = self.expected_output["BertSumExt"]["summary"]
        summary = model.predict(document)
        assert summary == expected_summary

    @parameterized.expand(get_testing_device_parameters())
    def test_bert_sum_ext_abs_cnndm_regression(self, device: int):
        model = BertSumExtAbs(device=device)
        document = self.expected_output["BertSumExtAbs_CNNDM"]["document"]
        expected_summary = self.expected_output["BertSumExtAbs_CNNDM"]["summary"]
        summary = model.predict(document)
        assert summary == expected_summary

    @parameterized.expand(get_testing_device_parameters())
    def test_bert_sum_ext_abs_xsum_regression(self, device: int):
        model = BertSumExtAbs(model="bertsumextabs_xsum.pt", device=device)
        document = self.expected_output["BertSumExtAbs_XSum"]["document"]
        expected_summary = self.expected_output["BertSumExtAbs_XSum"]["summary"]
        summary = model.predict(document)
        assert summary == expected_summary

    @parameterized.expand(get_testing_device_parameters())
    def test_transformer_abs_regression(self, device: int):
        model = TransformerAbs(device=device)
        document = self.expected_output["TransformerAbs"]["document"]
        expected_summary = self.expected_output["TransformerAbs"]["summary"]
        summary = model.predict(document)
        assert summary == expected_summary
