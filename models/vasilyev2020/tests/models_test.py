import json
import unittest
from parameterized import parameterized

from repro.models.vasilyev2020 import BLANCHelp, BLANCTune
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters


class TestVasilyev2020Models(unittest.TestCase):
    def setUp(self) -> None:
        self.multiling2011_examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )

    @parameterized.expand(get_testing_device_parameters())
    def test_blanc_help(self, device: int):
        # Tests examples from the original repository. Some do not match the
        # exact values expected or even this Colab repository which re-runs
        # the examples from the repository
        # https://colab.research.google.com/drive/17pJ94L2kCL6QMBMflOm-H0ApBiOUWJ1H?usp=sharing
        # Further, there are differences between using GPU and CPU to compute the metric
        metric = BLANCHelp(device=device)

        document = "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
        summary = "Jack bought milk and honey."
        expected_score = {"blanc-help": 0.2222222222222222}
        actual_score = metric.predict([document], summary)
        assert_dicts_approx_equal(expected_score, actual_score, abs=1e-4)

        inputs = [
            {
                "sources": [
                    "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
                ],
                "candidate": "Jack bought milk and honey.",
            },
            {
                "sources": [
                    "As Jill started taking a walk in the park, she certainly noticed that the trees were extra green this year."
                ],
                "candidate": "Jill saw green trees in the park.",
            },
        ]
        expected_macro = {"blanc-help": 0.18253968253968253}
        expected_micro = [
            {"blanc-help": 0.2222222222222222},
            {"blanc-help": 0.14285714285714285},
        ]

        actual_macro, actual_micro = metric.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_blanc_tune(self, device: int):
        # See the note in `test_blanc_help`
        metric = BLANCTune(device=device, blanc_kwargs={"finetune_mask_evenly": False})

        document = "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
        summary = "Jack bought milk and honey."
        if device == -1:
            expected_score = {"blanc-tune": 0.1111111111111111}
        else:
            expected_score = {"blanc-tune": 0.3333333333333333}
        actual_score = metric.predict([document], summary)
        assert_dicts_approx_equal(expected_score, actual_score, abs=1e-4)

        inputs = [
            {
                "sources": [
                    "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
                ],
                "candidate": "Jack bought milk and honey.",
            },
            {
                "sources": [
                    "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
                ],
                "candidate": "Jack drove to the bazaar in a minivan",
            },
            {
                "sources": [
                    "As Jill started taking a walk in the park, she certainly noticed that the trees were extra green this year."
                ],
                "candidate": "Jill saw green trees in the park.",
            },
            {
                "sources": [
                    "As Jill started taking a walk in the park, she certainly noticed that the trees were extra green this year."
                ],
                "candidate": "The trees were green.",
            },
        ]
        if device == -1:
            expected_macro = {"blanc-tune": 0.11904761904761904}
            expected_micro = [
                {"blanc-tune": 0.1111111111111111},
                {"blanc-tune": 0.2222222222222222},
                {"blanc-tune": 0.14285714285714285},
                {"blanc-tune": 0.0},
            ]
        else:
            expected_macro = {"blanc-tune": 0.15674603174603174}
            expected_micro = [
                {"blanc-tune": 0.3333333333333333},
                {"blanc-tune": 0.2222222222222222},
                {"blanc-tune": 0.07142857142857142},
                {"blanc-tune": 0.0},
            ]

        actual_macro, actual_micro = metric.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
