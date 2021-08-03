import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model

logger = logging.getLogger(__name__)


@Model.register("sellam2020-bleurt")
class BLEURT(Model):
    def __init__(
        self,
        image: str = "sellam2020",
        model: str = "bleurt-base-128",
        device: int = 0,
        batch_size: int = 16,
    ):
        self.image = image
        self.model = model
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def _check_references(references_list: List[List[TextType]]) -> List[TextType]:
        single_references = []
        for references in references_list:
            if len(references) != 1:
                raise Exception(
                    f"BLEURT only supports single references. Found: {len(references)}"
                )
            single_references.append(references[0])
        return single_references

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

        # BLEURT only supports a single reference. Check to make sure only
        # one has been passed in
        references = self._check_references(references_list)

        # The each candidate and reference must be `str`, not `List[str]`
        candidates = [util.flatten(candidate) for candidate in candidates]
        references = [util.flatten(reference) for reference in references]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            with open(host_input_file, "w") as out:
                for candidate, reference in zip(candidates, references):
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

            micro = open(host_output_file, "r").read().splitlines()
            micro = list(map(float, micro))
            macro = np.mean(micro)

            macro = {"bleurt": macro}
            micro = [{"bleurt": score} for score in micro]
            return macro, micro
