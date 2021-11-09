import unittest
from parameterized import parameterized

from repro.models.fitzgerald2018 import QASRLParser
from repro.testing import get_testing_device_parameters


class TestFitzGerald2018Models(unittest.TestCase):
    @parameterized.expand(get_testing_device_parameters())
    def test_qasrl_parser(self, device: int):
        model = QASRLParser(device=device)

        inputs = [
            {"sentence": "John went to the store."},
            {
                "sentence": "The man ate the burrito and threw the trash in the garbage bin."
            },
        ]
        expected_outputs = [
            {
                "words": ["John", "went", "to", "the", "store", "."],
                "verbs": [
                    {
                        "verb": "went",
                        "qa_pairs": [
                            {
                                "question": "Who went somewhere?",
                                "spans": [{"start": 0, "end": 0, "text": "John"}],
                            },
                            {
                                "question": "Where did someone go?",
                                "spans": [
                                    {"start": 2, "end": 4, "text": "to the store"},
                                    {"start": 3, "end": 4, "text": "the store"},
                                ],
                            },
                        ],
                        "index": 1,
                    }
                ],
            },
            {
                "words": [
                    "The",
                    "man",
                    "ate",
                    "the",
                    "burrito",
                    "and",
                    "threw",
                    "the",
                    "trash",
                    "in",
                    "the",
                    "garbage",
                    "bin",
                    ".",
                ],
                "verbs": [
                    {
                        "verb": "ate",
                        "qa_pairs": [
                            {
                                "question": "Who ate something?",
                                "spans": [{"start": 0, "end": 1, "text": "The man"}],
                            },
                            {
                                "question": "What did someone eat?",
                                "spans": [
                                    {"start": 3, "end": 4, "text": "the burrito"}
                                ],
                            },
                        ],
                        "index": 2,
                    },
                    {
                        "verb": "threw",
                        "qa_pairs": [
                            {
                                "question": "Who threw something?",
                                "spans": [{"start": 0, "end": 1, "text": "The man"}],
                            },
                            {
                                "question": "What did someone throw?",
                                "spans": [{"start": 7, "end": 8, "text": "the trash"}],
                            },
                            {
                                "question": "Where did someone throw something?",
                                "spans": [
                                    {
                                        "start": 9,
                                        "end": 12,
                                        "text": "in the garbage bin",
                                    }
                                ],
                            },
                        ],
                        "index": 6,
                    },
                ],
            },
        ]
        outputs = model.predict_batch(inputs)
        assert expected_outputs == outputs
