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
        elif self.dataset_name.startswith("scientific_papers"):
            # Get either arxiv or pubmed from the name, e.g. "scientific_papers/arxiv"
            specific_dataset = self.dataset_name.split("/")[1]

            dataset_splits = datasets.load_dataset("scientific_papers", specific_dataset)
            split = dataset_splits[self.split]
            for i, instance in enumerate(split):
                # There is no instance_id in `datasets`, so we make one up
                instance_id = f"{specific_dataset}-{split}-{i}"

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
