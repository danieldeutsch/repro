import unittest

from repro.common import util
from repro.data.dataset_readers import DatasetReader
from repro.data.output_writers import OutputWriter
from repro.models import Model


class TestUtil(unittest.TestCase):
    def test_flatten(self):
        assert util.flatten("text") == "text"
        assert util.flatten("text", separator="-") == "text"
        assert util.flatten(["1", "2"]) == "1 2"
        assert util.flatten(["1", "2"], separator="-") == "1-2"

    def test_load_model_no_required_args(self):
        @Model.register("test-model", exist_ok=True)
        class _Model(Model):
            def __init__(self, a: int = 2, b: int = 3):
                self.a = a
                self.b = b

        model = util.load_model("test-model")
        assert isinstance(model, _Model)
        assert model.a == 2
        assert model.b == 3

        model = util.load_model("test-model", '{"a": 4}')
        assert isinstance(model, _Model)
        assert model.a == 4
        assert model.b == 3

        model = util.load_model("test-model", {"a": 8})
        assert isinstance(model, _Model)
        assert model.a == 8
        assert model.b == 3

    def test_load_model_required_args(self):
        @Model.register("test-model", exist_ok=True)
        class _Model(Model):
            def __init__(self, a: int, b: int = 3):
                self.a = a
                self.b = b

        with self.assertRaises(Exception):
            util.load_model("test-model")

        model = util.load_model("test-model", '{"a": 4}')
        assert isinstance(model, _Model)
        assert model.a == 4
        assert model.b == 3

    def test_load_dataset_reader_no_required_args(self):
        @DatasetReader.register("test-dataset-reader", exist_ok=True)
        class _DatasetReader(DatasetReader):
            def __init__(self, a: int = 2, b: int = 3):
                self.a = a
                self.b = b

        dataset_reader = util.load_dataset_reader("test-dataset-reader")
        assert isinstance(dataset_reader, _DatasetReader)
        assert dataset_reader.a == 2
        assert dataset_reader.b == 3

        dataset_reader = util.load_dataset_reader("test-dataset-reader", '{"a": 4}')
        assert isinstance(dataset_reader, _DatasetReader)
        assert dataset_reader.a == 4
        assert dataset_reader.b == 3

        dataset_reader = util.load_dataset_reader("test-dataset-reader", {"a": 8})
        assert isinstance(dataset_reader, _DatasetReader)
        assert dataset_reader.a == 8
        assert dataset_reader.b == 3

    def test_load_dataset_reader_required_args(self):
        @DatasetReader.register("test-dataset-reader", exist_ok=True)
        class _DatasetReader(DatasetReader):
            def __init__(self, a: int, b: int = 3):
                self.a = a
                self.b = b

        with self.assertRaises(Exception):
            util.load_dataset_reader("test-dataset-reader")

        dataset_reader = util.load_dataset_reader("test-dataset-reader", '{"a": 4}')
        assert isinstance(dataset_reader, _DatasetReader)
        assert dataset_reader.a == 4
        assert dataset_reader.b == 3

    def test_load_output_writer_no_required_args(self):
        @OutputWriter.register("test-output-writer", exist_ok=True)
        class _OutputWriter(OutputWriter):
            def __init__(self, a: int = 2, b: int = 3):
                self.a = a
                self.b = b

        writer = util.load_output_writer("test-output-writer")
        assert isinstance(writer, _OutputWriter)
        assert writer.a == 2
        assert writer.b == 3

        writer = util.load_output_writer("test-output-writer", '{"a": 4}')
        assert isinstance(writer, _OutputWriter)
        assert writer.a == 4
        assert writer.b == 3

        writer = util.load_output_writer("test-output-writer", {"a": 8})
        assert isinstance(writer, _OutputWriter)
        assert writer.a == 8
        assert writer.b == 3

    def test_load_output_writer_required_args(self):
        @OutputWriter.register("test-output-writer", exist_ok=True)
        class _OutputWriter(OutputWriter):
            def __init__(self, a: int, b: int = 3):
                self.a = a
                self.b = b

        with self.assertRaises(Exception):
            util.load_output_writer("test-output-writer")

        writer = util.load_output_writer("test-output-writer", '{"a": 4}')
        assert isinstance(writer, _OutputWriter)
        assert writer.a == 4
        assert writer.b == 3
