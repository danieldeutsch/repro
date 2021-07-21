import json
import unittest

from parameterized import parameterized

from repro.models.lewis2020 import BART
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestLewis2020Models(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_output = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_bart_cnn(self, device: int):
        model = BART(device=device)
        document = self.expected_output["CNNDM"]["document"]
        expected_summary = self.expected_output["CNNDM"]["summary"]
        summary = model.predict(document)
        assert summary == expected_summary

    @parameterized.expand(get_testing_device_parameters())
    def test_bart_xsum(self, device: int):
        model = BART("bart.large.xsum", device=device)
        document = self.expected_output["XSum"]["document"]
        expected_summary = self.expected_output["XSum"]["summary"]
        summary = model.predict(document)
        assert summary == expected_summary
