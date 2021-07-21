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

    @overrides
    def write(
        self,
        instances: List[InstanceDict],
        predictions: List[Any],
        output_file: str,
        model_name: str,
        *args,
        **kwargs
    ) -> None:
        dirname = os.path.dirname(output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(output_file, "w") as out:
            for instance, prediction in zip(instances, predictions):
                out.write(
                    json.dumps(
                        {
                            "instance_id": instance["instance_id"],
                            "model_id": model_name,
                            "prediction": prediction,
                        }
                    )
                    + "\n"
                )
