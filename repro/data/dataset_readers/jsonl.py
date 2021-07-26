from overrides import overrides
from typing import List

from repro.common.io import read_jsonl_file
from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict


@DatasetReader.register("jsonl")
class JSONLinesDatasetReader(DatasetReader):
    @overrides
    def _read(self, *input_files: str) -> List[InstanceDict]:
        """
        Loads the instances from the `input_files`, where each line is expected to be
        a json-serialized object with an "instance_id" key.

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
            instances.extend(read_jsonl_file(input_file))
        return instances
