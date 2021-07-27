import datasets
import logging
import os
from datasets.config import HF_DATASETS_CACHE
from overrides import overrides
from typing import List

from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict

DEFAULT_VERSIONS = {"cnn_dailymail": "3.0.0", "xsum": "1.2.0"}

logger = logging.getLogger(__name__)


def hf_dataset_exists_locally(name: str, version: str = None) -> bool:
    """
    Checks to see if a Huggingface `datasets` dataset exists locally in the cache.
    The logic checks to see if the directory exists where the data should be, but
    does not do any further verification.

    Parameters
    ----------
    name : str
        The name of the dataset, like "cnn_dailymail"
    version : str, default=None
        The version of the dataset, like "3.0.0". If `None`, then the
        default version is used if one exists.

    Returns
    -------
    bool
        True if the dataset exists, False otherwise.
    """
    if version is None:
        version = DEFAULT_VERSIONS[name]
    data_dir = os.path.join(HF_DATASETS_CACHE, name, version)
    return os.path.exists(data_dir)


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
                logger.warning(
                    f"Unknown default dataset version for dataset: {name}. Using `None`"
                )
                version = None
            else:
                version = DEFAULT_VERSIONS[name]
        elif len(parts) == 2:
            name = parts[0]
            version = parts[1]
        else:
            raise Exception(
                f"Unknown dataset format. Expected 0 or 1 '/': {self.dataset_name}"
            )

        dataset = datasets.load_dataset(name, version, split=self.split)

        instances = []
        if name == "cnn_dailymail":
            for instance in dataset:
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
        elif name == "mocha":
            for instance in dataset:
                instances.append(
                    {
                        "instance_id": instance["id"],
                        "constituent_dataset": instance["constituent_dataset"],
                        "context": instance["context"],
                        "question": instance["question"],
                        "reference": instance["reference"],
                        "candidate": instance["candidate"],
                        "score": instance["score"],
                        "metadata": instance["metadata"],
                    }
                )
        elif name == "scientific_papers":
            for i, instance in enumerate(dataset):
                # There is no instance_id in `datasets`, so we make one up
                instance_id = f"{version}-{self.split}-{i}"

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
        elif name == "squad_v2":
            for instance in dataset:
                instances.append(
                    {
                        "instance_id": instance["id"],
                        "context": instance["context"],
                        "question": instance["question"],
                        "answers": instance["answers"],
                    }
                )
        elif name == "xsum":
            for instance in dataset:
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
