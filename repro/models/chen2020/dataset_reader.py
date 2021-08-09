from overrides import overrides
from typing import List

from repro.common.io import read_jsonl_file
from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict


@DatasetReader.register("chen2020-eval")
class Chen2020EvaluationDatasetReader(DatasetReader):
    @overrides
    def _read(self, *input_files: str) -> List[InstanceDict]:
        instances = []
        for input_file in input_files:
            predictions = read_jsonl_file(input_file)
            for prediction in predictions:
                instances.append(
                    {
                        "dataset": prediction["input"]["constituent_dataset"],
                        "source": prediction["input"]["metadata"]["source"],
                        "score": prediction["input"]["score"],
                        "prediction": prediction["prediction"],
                    }
                )
        return instances
