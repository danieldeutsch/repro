import unittest
from parameterized import parameterized

from repro.models.kitaev2019 import Benepar
from repro.testing import get_testing_device_parameters


class TestKitaev2019(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_benepar_en3(self, device: int):
        # Tests the example from the Github repo
        model = Benepar(device=device)
        text = "The time for action is now."
        trees = model.predict(text)
        assert trees == [
            "(S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))"
        ]
