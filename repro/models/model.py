from typing import Any, Dict, List

from repro.common import Registrable


class Model(Registrable):
    def predict(self, *args, **kwargs):
        """
        Runs inference for the single input instance.
        """
        raise NotImplementedError

    def predict_batch(self, inputs: List[Dict[str, Any]], *args, **kwargs):
        """
        Runs inference for all of the instances in `inputs`. Each item in `inputs`
        should have a key which corresponds to a parameter of the `predict` method.
        For instance, if the signature of predict was:

            `def predict(input1, input2)`

        Then each item in `inputs` should be a dictionary with keys `"input1"` and `"input2"`.
        """
        raise NotImplementedError
