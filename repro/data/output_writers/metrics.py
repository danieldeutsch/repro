import json
import os
from overrides import overrides
from typing import Any, List

from repro.data.output_writers import OutputWriter
from repro.data.types import InstanceDict


@OutputWriter.register("metrics")
class MetricsOutputWriter(OutputWriter):
    """
    Writes the json-serialized `predictions`, which are expected to be
    evaluation metrics.
    """

    def __init__(self):
        super().__init__(False)

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
            out.write(json.dumps(predictions, indent=2))
