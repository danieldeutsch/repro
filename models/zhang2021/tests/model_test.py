import json
import unittest
from parameterized import parameterized

from repro.models.zhang2021 import Lite3Pyramid
from repro.testing import FIXTURES_ROOT as REPRO_FIXTURES_ROOT
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters

from . import FIXTURES_ROOT


class TestZhang2021Models(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = json.load(
            open(f"{REPRO_FIXTURES_ROOT}/multiling2011/data.json", "r")
        )
        self.expected = json.load(open(f"{FIXTURES_ROOT}/expected.json", "r"))

        # Example data from https://github.com/ZhangShiyue/Lite2-3Pyramid/tree/main/data/REALSumm
        self.references = (
            open(f"{FIXTURES_ROOT}/references.txt", "r").read().splitlines()
        )
        self.stus = list(
            map(
                lambda line: line.split("\t"),
                open(f"{FIXTURES_ROOT}/STUs.txt", "r").read().splitlines(),
            )
        )
        self.summaries = open(f"{FIXTURES_ROOT}/summaries.txt", "r").read().splitlines()

    @parameterized.expand(get_testing_device_parameters())
    def test_extract_stus(self, device: int):
        # Ensures the example STUs from the repo are the ones extracted. We only
        # test the first 10 for time. We did notice slight differences on all 100.
        metric = Lite3Pyramid(device=device)

        num_inputs = 10
        input_references = self.references[:num_inputs]
        expected_stus = self.stus[:num_inputs]
        actual_stus = metric.extract_stus(input_references, True)
        assert actual_stus == expected_stus

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_predict_with_stus(self, device: int):
        # Tests to see whether using the summaries and STUs provided in the Github repo
        # reproduce the expected macro-level scores based on the Readme
        metric = Lite3Pyramid(device=device)

        inputs = [
            {"candidate": candidate, "units_list": [stus]}
            for candidate, stus in zip(self.summaries, self.stus)
        ]
        actual_macro, _ = metric.predict_batch(inputs)

        expected_macro = {
            "p2c": 0.4535250155726269,
            "l2c": 0.48114382512911924,
            "p3c": 0.38368004398714206,
            "l3c": 0.45765291326320745,
        }
        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_predict_with_references_regression(self, device: int):
        # Tests to see whether using the summaries and references provided in the Github repo
        # reproduce the expected macro-level scores based on the Readme. Currently, the
        # implementation does not reproduce the numbers, so this is a regression test to
        # see if they changed. We believe the STU extraction is different since that test
        # `test_extract_stus` does not reproduce the expected output for all 100 examples
        metric = Lite3Pyramid(device=device)

        inputs = [
            {"candidate": candidate, "references": [reference]}
            for candidate, reference in zip(self.summaries, self.references)
        ]
        actual_macro, _ = metric.predict_batch(inputs)

        expected_macro = {
            "p2c": 0.41426592423623565,
            "l2c": 0.4375990120990121,
            "p3c": 0.35243034715254057,
            "l3c": 0.41764121989121983,
        }
        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_predict_regression(self, device: int):
        # Runs the metric on the MultiLing summaries as a regression test
        model = Lite3Pyramid(device=device)
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected["default"]["macro"]
        expected_micro = self.expected["default"]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(
        get_testing_device_parameters(gpu_only=True), skip_on_empty=True
    )
    def test_predict_regression_generic_model(self, device: int):
        # Runs the metric on the MultiLing summaries as a regression test with a non-default model
        model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        model = Lite3Pyramid(device=device, model=model_name)
        inputs = [
            {"candidate": inp["candidate"], "references": inp["references"]}
            for inp in self.examples
        ]
        expected_macro = self.expected[model_name]["macro"]
        expected_micro = self.expected[model_name]["micro"]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro, abs=1e-4)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)
