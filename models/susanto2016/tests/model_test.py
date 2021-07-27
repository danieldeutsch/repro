import json
import unittest
from parameterized import parameterized

from repro.models.susanto2016 import RNNTruecaser
from repro.testing import get_testing_device_parameters

from . import FIXTURES_ROOT


class TestSusanto2016(unittest.TestCase):
    def setUp(self) -> None:
        # The expected outputs are the sentences in this file. We will pass the .lower() version
        # to the model
        self.expected_outputs = json.load(open(f"{FIXTURES_ROOT}/expected-output.json"))

    @parameterized.expand(get_testing_device_parameters())
    def test_rnn_truecaser_en(self, device: int):
        model = RNNTruecaser("wiki-truecaser-model-en.tar.gz", device=device)
        inputs = [{"text": text.lower()} for text in self.expected_outputs["en"]]
        predictions = model.predict_batch(inputs)
        assert predictions == self.expected_outputs["en"]

    @parameterized.expand(get_testing_device_parameters())
    def test_rnn_truecaser_es(self, device: int):
        model = RNNTruecaser("wmt-truecaser-model-es.tar.gz", device=device)
        inputs = [{"text": text.lower()} for text in self.expected_outputs["es"]]
        predictions = model.predict_batch(inputs)
        assert predictions == self.expected_outputs["es"]

    @parameterized.expand(get_testing_device_parameters())
    def test_rnn_truecaser_de(self, device: int):
        model = RNNTruecaser("wmt-truecaser-model-de.tar.gz", device=device)
        inputs = [{"text": text.lower()} for text in self.expected_outputs["de"]]
        predictions = model.predict_batch(inputs)
        assert predictions == self.expected_outputs["de"]

    @parameterized.expand(get_testing_device_parameters())
    def test_rnn_truecaser_ru(self, device: int):
        model = RNNTruecaser("lrl-truecaser-model-ru.tar.gz", device=device)
        inputs = [{"text": text.lower()} for text in self.expected_outputs["ru"]]
        predictions = model.predict_batch(inputs)
        assert predictions == self.expected_outputs["ru"]
