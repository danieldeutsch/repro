import unittest

from repro.data.output_writers import DefaultOutputWriter
from repro.common import TemporaryDirectory
from repro.common.io import read_jsonl_file


class TestDefaultOutputWriter(unittest.TestCase):
    def test_default_output_writer(self):
        writer = DefaultOutputWriter()
        with TemporaryDirectory() as temp:
            output_file = f"{temp}/output.jsonl"

            instances = [
                {"instance_id": "1"},
                {"instance_id": "2"},
            ]
            predictions = ["a", "b"]
            writer.write(instances, predictions, output_file, "test-model")

            outputs = read_jsonl_file(output_file)
            assert outputs == [
                {"instance_id": "1", "model_id": "test-model", "prediction": "a"},
                {"instance_id": "2", "model_id": "test-model", "prediction": "b"},
            ]
