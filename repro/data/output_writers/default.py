import json
import os
from overrides import overrides
from typing import Any, List

from repro.data.output_writers import OutputWriter
from repro.data.types import InstanceDict


@OutputWriter.register("default")
class DefaultOutputWriter(OutputWriter):
    """
    Writes a jsonl file with keys for the `instance_id`, `model_id`, and `prediction`.
    """

    def __init__(self, include_input: bool = False):
        super().__init__(True)
        self.include_input = include_input

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
                data = {
                    "instance_id": instance["instance_id"],
                    "model_id": model_name,
                }
                if self.include_input:
                    data["input"] = instance
                data["prediction"] = prediction

                out.write(json.dumps(data) + "\n")
