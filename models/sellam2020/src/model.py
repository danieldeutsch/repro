import json
import logging
import numpy as np
import os
from typing import Any, Dict, List, Tuple, Union

from repro.common import TemporaryDirectory, util
from repro.common.docker import make_volume_map, run_command
from repro.data.types import TextType
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
    ) -> Dict[str, float]:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
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

        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_input_file = f"{host_input_dir}/input.jsonl"
            container_input_file = f"{container_input_dir}/input.jsonl"
            os.makedirs(host_input_dir)
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

            host_output_file = f"{host_output_dir}/output.jsonl"
            container_output_file = f"{container_output_dir}/output.jsonl"

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
            os.makedirs(host_output_dir)
            run_command(
                self.image,
                command,
                volume_map=volume_map,
                cuda=cuda,
                network_disabled=True,
            )

            micro = open(host_output_file, "r").read().splitlines()
            micro = list(map(float, micro))
            macro = np.mean(micro)

            macro = {"bleurt": macro}
            micro = [{"bleurt": score} for score in micro]
            return macro, micro