import datasets
from overrides import overrides
from typing import List

from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict

DEFAULT_VERSIONS = {
    "cnn_dailymail": "3.0.0",
    "xsum": "1.2.0"
}


@DatasetReader.register("huggingface-datasets")
class HuggingfaceDatasetsDatasetReader(DatasetReader):
    def __init__(self, dataset_name: str, split: str):
        self.dataset_name = dataset_name
        self.split = split

    @overrides
    def _read(self) -> List[InstanceDict]:
        # Split the version if it exists, taking the default if not
        parts = self.dataset_name.split("/")
        if len(parts) == 1:
            name = self.dataset_name
            if name not in DEFAULT_VERSIONS:
                raise Exception(f"Unknown default dataset version for dataset: {name}")
            version = DEFAULT_VERSIONS[name]
        elif len(parts) == 2:
            name = parts[0]
            version = parts[1]
        else:
            raise Exception(
                f"Unknown dataset format. Expected 0 or 1 '/': {self.dataset_name}"
            )

        dataset_splits = datasets.load_dataset(name, version)
        split = dataset_splits[self.split]

        instances = []
        if name == "cnn_dailymail":
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
        elif name == "xsum":
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
        elif name == "scientific_papers":
            for i, instance in enumerate(split):
                # There is no instance_id in `datasets`, so we make one up
                instance_id = f"{version}-{split}-{i}"

                # The articles and abstracts have paragraphs split by \n
                document = instance["article"].split("\n")
                summary = instance["abstract"].split("\n")

                # The section names are split by \n
                section_names = instance["section_names"].split("\n")

                instances.append(
                    {
                        "instance_id": instance_id,
                        "document": document,
                        "reference": summary,
                        "section_names": section_names,
                    }
                )
        else:
            raise Exception(f"Unsupported dataset: {self.dataset_name}")

        return instances
