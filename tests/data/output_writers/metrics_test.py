import json
import unittest

from repro.data.output_writers import MetricsOutputWriter
from repro.common import TemporaryDirectory


class TestMetricsOutputWriter(unittest.TestCase):
    def test_default_output_writer(self):
        writer = MetricsOutputWriter()
        with TemporaryDirectory() as temp:
            output_file = f"{temp}/output.json"

            # Does not matter if the instances are empty or not
            instances = []
            predictions = {"a": 1, "b": 2}
            writer.write(instances, predictions, output_file, "test-model")

            output = json.load(open(output_file, "r"))
            assert output == predictions
