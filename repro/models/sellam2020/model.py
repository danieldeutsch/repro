import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.sellam2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-bleurt")
class BLEURT(Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        model: str = "bleurt-base-128",
        device: int = 0,
        batch_size: int = 16,
    ):
        self.image = image
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def predict(
        self,
        candidate: TextType,
        references: List[TextType],
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(
            f"Calculating BLEURT with model {self.model} and "
            f"image {self.image} on {len(inputs)} inputs."
        )

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # The each candidate and reference must be `str`, not `List[str]`
        candidates = [util.flatten(candidate) for candidate in candidates]
        references_list = [
            [util.flatten(reference) for reference in references]
            for references in references_list
        ]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"

            # BLEURT only runs with a single reference, so we write
            # the candidate with each of its references on its own. Later
            # the scores will be aggregated.
            with open(host_input_file, "w") as out:
                for candidate, references in zip(candidates, references_list):
                    for reference in references:
                        out.write(
                            json.dumps(
                                {
                                    "candidate": candidate,
                                    "reference": reference,
                                }
                            )
                            + "\n"
                        )

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            commands.append("cd bleurt")
            commands.append(
                f"python -m bleurt.score_files"
                f"  -sentence_pairs_file {container_input_file}"
                f"  -bleurt_checkpoint ../{self.model}"
                f"  -scores_file {container_output_file}"
                f"  -bleurt_batch_size {self.batch_size}"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            results = open(host_output_file, "r").read().splitlines()
            results = list(map(float, results))

            # Regroup by reference
            micro_metrics = []
            index = 0
            for references in references_list:
                scores = results[index : index + len(references)]
                index += len(references)
                micro_metrics.append(
                    {"bleurt": {"mean": np.mean(scores), "max": np.max(scores)}}
                )

            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
