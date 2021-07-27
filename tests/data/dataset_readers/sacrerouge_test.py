import json
import unittest

from repro.common import TemporaryDirectory
from repro.data.dataset_readers import SacreROUGEDatasetReader


class TestSacreROUGEDatasetReader(unittest.TestCase):
    def test_sacrerouge_dataset_reader(self):
        with TemporaryDirectory() as temp:
            # Write the data
            input_file = f"{temp}/data.jsonl"
            with open(input_file, "w") as out:
                out.write(
                    json.dumps(
                        {
                            "instance_id": "1",
                            "summary": {"text": "The summary 1"},
                            "reference": {"text": "The reference"},
                        }
                    )
                    + "\n"
                )
                out.write(
                    json.dumps(
                        {
                            "instance_id": "2",
                            "summary": {"text": "The summary 2"},
                            "references": [
                                {"text": "The first reference"},
                                {"text": "The second reference"},
                            ],
                        }
                    )
                    + "\n"
                )

            reader = SacreROUGEDatasetReader()
            instances = reader.read(input_file)
            assert instances == [
                {
                    "instance_id": "1",
                    "summary": "The summary 1",
                    "references": ["The reference"],
                },
                {
                    "instance_id": "2",
                    "summary": "The summary 2",
                    "references": ["The first reference", "The second reference"],
                },
            ]
