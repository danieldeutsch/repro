import json
from overrides import overrides
from typing import List

from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict


@DatasetReader.register("sacrerouge")
class SacreROUGEDatasetReader(DatasetReader):
    @overrides
    def _read(self, *input_files: str) -> List[InstanceDict]:
        """
        Loads the instances from the `input_files`. Each of the instances should
        have an "instance_id" key with a value that uniquely identifies that instance
        plus a "summary" and "reference" or "references" field.

        Parameters
        ----------
        input_files : str
            The input files

        Returns
        -------
        List[InstanceDict]
            The instances
        """
        instances = []
        for input_file in input_files:
            with open(input_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    summary = data["summary"]["text"]
                    if "reference" in data:
                        references = [data["reference"]["text"]]
                    else:
                        references = [
                            reference["text"] for reference in data["references"]
                        ]
                    instances.append(
                        {
                            "instance_id": data["instance_id"],
                            "summary": summary,
                            "references": references,
                        }
                    )
        return instances
