import datasets
from overrides import overrides
from typing import List

from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict


@DatasetReader.register("huggingface-datasets")
class HuggingfaceDatasetsDatasetReader(DatasetReader):
    def __init__(self, dataset_name: str, split: str):
        self.dataset_name = dataset_name
        self.split = split

    @overrides
    def _read(self) -> List[InstanceDict]:
        instances = []
        if self.dataset_name == "cnn_dailymail":
            dataset_splits = datasets.load_dataset(self.dataset_name, "3.0.0")
            split = dataset_splits[self.split]
            for instance in split:
                # The reference sentences are separated by \n, but the document
                # is not sentence split already
                reference = instance["highlights"].split("\n")
                instances.append(
                    {
                        "instance_id": instance["id"],
                        "document": instance["article"],
                        "reference": reference,
                    }
                )
        elif self.dataset_name == "xsum":
            dataset_splits = datasets.load_dataset(self.dataset_name, "1.2.0")
            split = dataset_splits[self.split]
            for instance in split:
                # The documents include \n characters to separate sentences. The
                # summaries are only one sentence, so it does not matter for them
                document = instance["document"].split("\n")
                instances.append(
                    {
                        "instance_id": instance["id"],
                        "document": document,
                        "reference": instance["summary"],
                    }
                )
        else:
            raise Exception(f"Unsupported dataset: {self.dataset_name}")

        return instances
