import unittest
from parameterized import parameterized

from repro.models.thompson2020 import Prism, PrismSrc
from repro.testing import assert_dicts_approx_equal, get_testing_device_parameters


class TestThompson202Models(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_prism_references_example(self, device: int):
        # Tests the examples from the Github repo
        model = Prism(device=device)
        inputs = [
            {"candidate": "Hi world.", "references": ["Hello world."]},
            {"candidate": "This is a Test.", "references": ["This is a test."]},
        ]
        expected_macro = {"prism": -1.0184666}
        expected_micro = [{"prism": -1.4878583}, {"prism": -0.5490748}]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_prism_sources_example(self, device: int):
        # Tests the examples from the Github repo
        model = PrismSrc(device=device)
        inputs = [
            {"candidate": "Hi world.", "sources": ["Bonjour le monde."]},
            {"candidate": "This is a Test.", "sources": ["C'est un test."]},
        ]
        expected_macro = {"prism-src": -1.8306842}
        expected_micro = [{"prism-src": -2.462842}, {"prism-src": -1.1985264}]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_multi_reference(self, device: int):
        model = Prism(device=device)
        inputs = [
            {"candidate": "Hi world.", "references": ["Hello world."]},
            {"candidate": "Hi world.", "references": ["This is a Test."]},
            {
                "candidate": "Hi world.",
                "references": ["Hello world.", "This is a Test."],
            },
        ]
        expected_macro = {"prism": -3.9376096725463867}
        expected_micro = [
            {"prism": -1.4878592491149902},
            {"prism": -6.387360095977783},
            {"prism": -3.9376096725463867},
        ]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_multi_source(self, device: int):
        model = PrismSrc(device=device)
        inputs = [
            {"candidate": "Hi world.", "sources": ["Bonjour le monde."]},
            {"candidate": "Hi world.", "sources": ["C'est un test."]},
            {
                "candidate": "Hi world.",
                "sources": ["Bonjour le monde.", "C'est un test."],
            },
        ]
        expected_macro = {"prism-src": -5.0302382707595825}
        expected_micro = [
            {"prism-src": -2.4628407955169678},
            {"prism-src": -7.597635746002197},
            {"prism-src": -5.0302382707595825},
        ]
        actual_macro, actual_micro = model.predict_batch(inputs)

        assert_dicts_approx_equal(expected_macro, actual_macro)
        assert len(expected_micro) == len(actual_micro)
        for expected, actual in zip(expected_micro, actual_micro):
            assert_dicts_approx_equal(expected, actual, abs=1e-4)

    @parameterized.expand(get_testing_device_parameters())
    def test_prism_translation_regression(self, device: int):
        # Tests using Prism as a translation system using
        # the examples from https://github.com/thompsonb/prism/tree/master/translation
        model = Prism(device=device)
        inputs = [
            {"source": "Hi world."},
            {"source": "This is a Test."},
            {"source": "Some of my Best Friends are Linguists."},
        ]

        actual_outputs = model.translate_batch("fr", inputs)
        assert actual_outputs == [
            "Le Monde.",
            "C'est un test.",
            "Certains de mes meilleurs amis sont linguistes.",
        ]
