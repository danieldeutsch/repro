import unittest
from parameterized import parameterized

from repro.models.rei2020 import COMET
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters


class TestRei2020Models(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_comet_regression(self, device: int):
        # Tests the examples from the Github repo
        model = COMET(device=device)

        inputs = [
            {
                "sources": ["Dem Feuer konnte Einhalt geboten werden"],
                "candidate": "The fire could be stopped",
                "references": ["They were able to control the fire."],
            },
            {
                "sources": ["Schulen und Kindergärten wurden eröffnet."],
                "candidate": "Schools and kindergartens were open",
                "references": ["Schools and kindergartens opened"],
            },
        ]

        expected_macro = {"comet": 0.5529156997799873}
        expected_micro = [{"comet": 0.19016893208026886}, {"comet": 0.9156624674797058}]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_comet_src_regression(self, device: int):
        # Tests the examples from the Github repo
        model = COMET(device=device)

        inputs = [
            {
                "sources": ["Dem Feuer konnte Einhalt geboten werden"],
                "candidate": "The fire could be stopped",
            },
            {
                "sources": ["Schulen und Kindergärten wurden eröffnet."],
                "candidate": "Schools and kindergartens were open",
            },
        ]

        expected_macro = {"comet-src": 0.35479202680289745}
        expected_micro = [
            {"comet-src": 0.00831037387251854},
            {"comet-src": 0.7012736797332764},
        ]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
