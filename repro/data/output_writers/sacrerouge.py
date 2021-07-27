import json
import os
from overrides import overrides
from typing import Any, List

from repro.data.output_writers import OutputWriter
from repro.data.types import InstanceDict


@OutputWriter.register("sacrerouge")
class SacreROUGEOutputWriter(OutputWriter):
    """
    Writes a jsonl file that is compatible for scoring with SacreROUGE. Each object
    includes an `instance_id`, `summarizer_id` (equal to the model name), `summarizer_type` (equal to "peer"),
    and `summary` (equal to the prediction). If the instance includes a reference, it will
    be included as well
    """

    def __init__(self):
        super().__init__(True)

    @overrides
    def _write(
        self,
        instances: List[InstanceDict],
        predictions: Any,
        output_file_or_dir: str,
        model_name: str,
        *args,
        **kwargs
    ) -> None:
        output_file = output_file_or_dir
        dirname = os.path.dirname(output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(output_file, "w") as out:
            for instance, prediction in zip(instances, predictions):
                output_dict = {
                    "instance_id": instance["instance_id"],
                    "summarizer_id": model_name,
                    "summarizer_type": "peer",
                    "summary": {"text": prediction},
                }
                if "reference" in instance:
                    output_dict["reference"] = {"text": instance["reference"]}
                elif "references" in instance:
                    output_dict["references"] = [
                        {"text": reference} for reference in instance["references"]
                    ]

                out.write(json.dumps(output_dict) + "\n")
