import json
import logging
import os
from typing import Dict, List, Union

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.data.types import MetricsType
from repro.models import Model
from repro.models.squad_v2 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(MODEL_NAME)
class SQuADv2Evaluation(Model):
    def __init__(self, image: str = DEFAULT_IMAGE):
        self.image = image

    def predict(
        self, instance_id: str, prediction: str, null_probability: float, **kwargs
    ) -> MetricsType:
        # We cannot evaluate just one instance
        raise NotImplementedError

    def predict_batch(
        self, inputs: List[Dict[str, Union[str, float]]], *args, **kwargs
    ) -> MetricsType:
        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_pred_file = f"{host_input_dir}/pred.json"
            host_na_prob_file = f"{host_input_dir}/na_prob.json"
            container_pred_file = f"{container_input_dir}/pred.json"
            container_na_probfile = f"{container_input_dir}/na_prob.json"

            predictions = {}
            na_probs = {}
            for inp in inputs:
                instance_id = inp["instance_id"]
                predictions[instance_id] = inp["prediction"]
                na_probs[instance_id] = inp["null_probability"]

            os.makedirs(host_input_dir)
            with open(host_pred_file, "w") as out:
                out.write(json.dumps(predictions, indent=2))

            with open(host_na_prob_file, "w") as out:
                out.write(json.dumps(na_probs, indent=2))

            host_output_file = f"{host_output_dir}/eval.json"
            container_output_file = f"{container_output_dir}/eval.json"

            command = (
                f"python evaluate-v2.0.py"
                f"  dev-v2.0.json"
                f"  {container_pred_file}"
                f"  --na-prob-file {container_na_probfile}"
                f"  --out-file {container_output_file}"
            )
            os.makedirs(host_output_dir)
            run_command(
                self.image, command, volume_map=volume_map, network_disabled=True
            )

            metrics = json.load(open(host_output_file, "r"))
            return metrics
