import json
import pytest
import unittest
from typing import List

from repro.models import Model, ParallelModel
from repro.models.zhang2020 import BERTScore
from repro.testing import (
    FIXTURES_ROOT,
    assert_dicts_approx_equal,
    get_testing_device_parameters,
)


class TestParallelModel(unittest.TestCase):
    def test_cpu_parallel(self):
        # Tests to ensure all of the inputs are batched, processed
        # and returned in the same order as they were input
        class _ProcessIDModel(Model):
            """A simple Model which returns the inputs"""

            def predict_batch(self, inputs: List, **kwargs) -> List:
                return inputs

        num_models = 10
        num_inputs = 100
        model = ParallelModel(_ProcessIDModel, num_models=num_models)

        inputs = [{"value": value} for value in range(num_inputs)]
        outputs_list = model.predict_batch(inputs)

        # Ensure all of the outputs are present and in the expected order
        assert len(outputs_list) == num_models
        expected = 0
        for output in outputs_list:
            for value in output:
                assert value["value"] == expected
                expected += 1
        assert expected == num_inputs

    @pytest.mark.skipif(
        len(get_testing_device_parameters(gpu_only=True)) < 2,
        reason="Test requres at least 2 available GPUs",
    )
    def test_gpu_parallel(self):
        # Ensures a serial GPU model returns the same output as the parallel model
        devices = [device for device, in get_testing_device_parameters(gpu_only=True)]
        parameters = [{"device": device} for device in devices]

        inputs = json.load(open(f"{FIXTURES_ROOT}/multiling2011/data.json", "r"))
        serial_model = BERTScore(device=devices[0])
        parallel_model = ParallelModel(BERTScore, parameters)

        _, serial_outputs = serial_model.predict_batch(inputs)
        parallel_outputs_list = parallel_model.predict_batch(inputs)

        # Ensure there was one process per GPU
        assert len(parallel_outputs_list) == len(devices)

        # Ensure the serial and parallel models output the same values
        index = 0
        for _, micro in parallel_outputs_list:
            for value in micro:
                assert_dicts_approx_equal(serial_outputs[index], value)
                index += 1
        assert index == len(serial_outputs)

    def test_input_validation(self):
        class _DummyModel(Model):
            pass

        # Must pass kwargs or number of models
        with self.assertRaises(ValueError):
            ParallelModel(_DummyModel)

        # Must not be empty
        with self.assertRaises(ValueError):
            ParallelModel(_DummyModel, [])

        # Must be positive
        with self.assertRaises(ValueError):
            ParallelModel(_DummyModel, num_models=0)

        # Only one is allowed
        with self.assertRaises(ValueError):
            ParallelModel(_DummyModel, [{}, {}], num_models=2)