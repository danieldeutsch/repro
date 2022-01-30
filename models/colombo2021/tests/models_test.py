import unittest
from parameterized import parameterized

from repro.models.colombo2021 import BaryScore, DepthScore, InfoLM
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters


class TestColombo2021Models(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_infolm(self, device: int):
        model = InfoLM(device=device, idf=True)
        inputs = [
            {
                "candidate": "I like my cakes very much",
                "references": ["I like my cakes very much"],
            },
            {
                "candidate": "I like my cakes very much",
                "references": ["I hate these cakes!"],
            },
        ]

        actual_macro, actual_micro = model.predict_batch(inputs)
        expected_macro = {
            "infolm": {
                "fisher_rao": 1.1136181355,
                "r_fisher_rao": 1.1136181355,
                "sim_fisher_rao": 1.1136181355,
            }
        }
        expected_micro = [
            {
                "infolm": {
                    "fisher_rao": 0.0,
                    "r_fisher_rao": 0.0,
                    "sim_fisher_rao": 0.0,
                }
            },
            {
                "infolm": {
                    "fisher_rao": 2.227236270904541,
                    "r_fisher_rao": 2.227236270904541,
                    "sim_fisher_rao": 2.227236270904541,
                }
            },
        ]

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_baryscore(self, device: int):
        model = BaryScore(device=device, idf=True)
        inputs = [
            {
                "candidate": "I like my cakes very much",
                "references": ["I like my cakes very much"],
            },
            {
                "candidate": "I like my cakes very much",
                "references": ["I hate these cakes!"],
            },
        ]

        actual_macro, actual_micro = model.predict_batch(inputs)
        expected_macro = {
            "baryscore": {
                "baryscore_W": 1.1136181355,
                "baryscore_SD": 1.1136181355,
            }
        }
        expected_micro = [
            {
                "baryscore": {
                    "baryscore_W": 0.0,
                    "baryscore_SD": 0.0,
                }
            },
            {
                "baryscore": {
                    "baryscore_W": 0.004107567128949505,
                    "baryscore_SD": 0.09756940805019898,
                }
            },
        ]

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_depthscore(self, device: int):
        model = DepthScore(device=device)
        inputs = [
            {
                "candidate": "I like my cakes very much",
                "references": ["I like my cakes very much"],
            },
            {
                "candidate": "I like my cakes very much",
                "references": ["I hate these cakes!"],
            },
        ]

        actual_macro, actual_micro = model.predict_batch(inputs)
        expected_macro = {
            "depthscore": {
                "depth_score": 0.05552341937,
            }
        }
        expected_micro = [
            {
                "depthscore": {
                    "depth_score": 0.0,
                }
            },
            {
                "depthscore": {
                    "depth_score": 0.11104683874186258,
                }
            },
        ]

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
