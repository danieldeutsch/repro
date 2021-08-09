from overrides import overrides
from typing import List

from repro.common.io import read_jsonl_file
from repro.data.dataset_readers import DatasetReader
from repro.data.types import InstanceDict
from repro.models.deutsch2021 import MODEL_NAME


@DatasetReader.register(f"{MODEL_NAME}-question-answering-eval")
class Deutsch2021QuestionAnsweringEvaluationDatasetReader(DatasetReader):
    @overrides
    def _read(self, *input_files: str) -> List[InstanceDict]:
        instances = []
        for input_file in input_files:
            predictions = read_jsonl_file(input_file)
            for prediction in predictions:
                instances.append(
                    {
                        "instance_id": prediction["instance_id"],
                        "prediction": prediction["prediction"]["prediction"],
                        "null_probability": prediction["prediction"][
                            "null_probability"
                        ],
                    }
                )
        return instances
