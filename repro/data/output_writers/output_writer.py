from typing import Any, List

from repro.common import Registrable
from repro.data.types import InstanceDict


class OutputWriter(Registrable):
    def __init__(self, prediction_per_instance: bool) -> None:
        """
        Parameters
        ----------
        prediction_per_instance : bool
            If `True`, indicates there should be exactly one prediction per instances, and
            the `write()` method will verify this is the case. If `False`, no checking is done.
        """
        self.prediction_per_instance = prediction_per_instance

    def write(
        self,
        instances: List[InstanceDict],
        predictions: Any,
        output_file_or_dir: str,
        *args,
        **kwargs,
    ) -> None:
        if self.prediction_per_instance:
            if not isinstance(predictions, list):
                raise Exception(
                    f"`predictions` is expected to be a list if `self.prediction_per_instance` is `True`"
                )
            if len(instances) != len(predictions):
                raise Exception(
                    f"Number of instances {len(instances)} is not equal to the number "
                    f"of predictions {len(predictions)}"
                )
        self._write(instances, predictions, output_file_or_dir, *args, **kwargs)

    def _write(
        self,
        instances: List[InstanceDict],
        predictions: Any,
        output_file_or_dir: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Writes the results of the prediction to the `output_file_or_dir`. The `instances` and `predictions`
        are parallel lists such that `predictions[i]` is the prediction for `instances[i]`.

        Parameters
        ----------
        instances : List[InstanceDict]
            The instances that were used to get the predictions
        predictions : Any
            The predictions to save
        output_file_or_dir : str
            The output file or directory
        """
        raise NotImplementedError
