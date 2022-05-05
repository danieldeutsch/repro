import unittest
from parameterized import parameterized

from repro.models.krubinski2021 import MTEQA
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters


class TestKrubinski2021Models(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_mteqa_regression_short(self, device: int):
        model = MTEQA(device=device)
        candidate = "It is said that when Richard got sick, Salahuddin sent him some aloof, which was kept in the snow."
        reference = "It is said, when Richard got sick, Salahudin sent him few Plum fruit which were kept in the snow."
        expected = {
            "mteqa": {
                "f1": 0.21333333333333332,
                "em": 0.0,
                "chrf": 48.04483459121631,
                "bleu": 13.194715521231362,
            }
        }
        actual = model.predict(candidate, [reference])
        assert_dicts_approx_equal(actual, expected)

        # Include gen_from_out=True
        expected = {
            "mteqa": {
                "f1": 0.24,
                "em": 0.0,
                "chrf": 52.15993009975308,
                "bleu": 14.844054961385282,
            }
        }
        actual = model.predict(candidate, [reference], gen_from_out=True)
        assert_dicts_approx_equal(actual, expected)

    @parameterized.expand(get_testing_device_parameters())
    def test_mteqa_regression(self, device: int):
        model = MTEQA(device=device)
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
        expected_macro = {
            "mteqa": {
                "f1": 0.38888888888888884,
                "em": 0.3333333333333333,
                "chrf": 41.14592658372042,
                "bleu": 36.39899534309537,
            }
        }
        expected_micro = [
            {
                "mteqa": {
                    "f1": 0.9166666666666666,
                    "em": 0.75,
                    "chrf": 85.93155893536121,
                    "bleu": 84.1969860292861,
                }
            },
            {"mteqa": {"f1": 0.0, "em": 0.0, "chrf": 2.7027027027027026, "bleu": 0.0}},
            {
                "mteqa": {
                    "f1": 0.25,
                    "em": 0.25,
                    "chrf": 34.80351811309734,
                    "bleu": 25.00000000000001,
                }
            },
        ]

        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(actual_macro, expected_macro)
        assert len(actual_micro) == len(expected_micro)
        for actual, expected in zip(actual_micro, expected_micro):
            assert_dicts_approx_equal(actual, expected)
