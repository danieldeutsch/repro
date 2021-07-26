import json
import unittest

from repro.common import TemporaryDirectory
from repro.data.dataset_readers import JSONLinesDatasetReader


class TestJSONLinesDatasetReader(unittest.TestCase):
    def test_jsonl_dataset_reader(self):
        with TemporaryDirectory() as temp:
            input_file1 = f"{temp}/input1.jsonl"
            input_file2 = f"{temp}/input2.jsonl"

            instances = [
                {"instance_id": "1", "a": 1},
                {"instance_id": "2", "b": 2},
                {"instance_id": "3", "c": 3},
            ]

            with open(input_file1, "w") as out:
                out.write(json.dumps(instances[0]) + "\n")
                out.write(json.dumps(instances[1]) + "\n")

            with open(input_file2, "w") as out:
                out.write(json.dumps(instances[2]) + "\n")

            reader = JSONLinesDatasetReader()
            actual = reader.read(input_file1)
            assert actual == instances[:2]

            actual = reader.read(input_file1, input_file2)
            assert actual == instances
