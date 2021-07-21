from typing import Any, List

from repro.common import Registrable
from repro.data.types import InstanceDict


class OutputWriter(Registrable):
    def write(
        self,
        instances: List[InstanceDict],
        predictions: List[Any],
        output_file: str,
        *args,
        **kwargs
    ) -> None:
        """
        Writes the results of the prediction to the `output_file`. The `instances` and `predictions`
        are parallel lists such that `predictions[i]` is the prediction for `instances[i]`.

        Parameters
        ----------
        instances : List[InstanceDict]
            The instances that were used to get the predictions
        predictions : List[Any]
            The predictions to save
        output_file : str
            The output file
        """
        raise NotImplementedError
