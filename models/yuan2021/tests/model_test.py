import json
import pytest
import unittest
from parameterized import parameterized

from repro.models.yuan2021 import BARTScore
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestYuan2021Model(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        # self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

    @parameterized.expand(get_testing_device_parameters())
    def test_bartscore_examples(self, device: int):
        # Tests the examples from their Github Repo
        model = BARTScore(device=device)
        expected = {"bartscore": -2.510652780532837}
        actual = model.predict("This is interesting.", ["This is fun."])
        assert_dicts_approx_equal(actual, expected)

        model = BARTScore(device=device, model="parabank")
        expected = {"bartscore": -2.336203098297119}
        actual = model.predict("This is interesting.", ["This is fun."])
        assert_dicts_approx_equal(actual, expected)
