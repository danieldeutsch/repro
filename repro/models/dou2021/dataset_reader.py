from overrides import overrides
from typing import List

from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict


@DatasetReader.register("dou2021")
class Dou2021DatasetReader(DatasetReader):
    @overrides
    def _read(
        self, source_file: str, target_file: str, guidance_file: str = None
    ) -> List[InstanceDict]:
        source = open(source_file, "r").read().splitlines()
        target = open(target_file, "r").read().splitlines()
        guidance = (
            open(guidance_file, "r").read().splitlines()
            if guidance_file is not None
            else None
        )

        instances = []
        for i, (document, reference) in enumerate(zip(source, target)):
            instance = {
                "instance_id": str(i),
                "document": document,
                "reference": reference,
            }
            if guidance is not None:
                instance["guidance"] = guidance[i]
            instances.append(instance)
        return instances
