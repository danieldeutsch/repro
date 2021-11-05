import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType
from repro.models import Model
from repro.models.hessel2021 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-clipscore")
class CLIPScore(Model):
    def __init__(self, image: str = DEFAULT_IMAGE, device: int = 0) -> None:
        self.image = image
        self.device = device

    @staticmethod
    def _verify_all_or_no_references(references_list: List[List[str]]) -> bool:
        has_references = [references is not None for references in references_list]
        if any(has_references) and not all(has_references):
            raise Exception(f"If any input has references, all must.")
        return any(has_references)

    def predict(
        self,
        candidate: str,
        image_file: str,
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [
                {
                    "candidate": candidate,
                    "image_file": image_file,
                    "references": references,
                }
            ],
            **kwargs,
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[str, List[str]]]],
        **kwargs,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(
            f"Calculating CLIPScore for {len(inputs)} inputs with image {self.image}"
        )

        candidates = [inp["candidate"] for inp in inputs]
        image_files = [inp["image_file"] for inp in inputs]
        references_list = [
            inp["references"] if "references" in inp else None for inp in inputs
        ]

        # The references are optional. Make sure either all have references or none do
        has_references = self._verify_all_or_no_references(references_list)

        with DockerContainer(self.image) as backend:
            host_candidate_file = f"{backend.host_dir}/candidates.json"
            container_candidate_file = f"{backend.container_dir}/candidates.json"

            host_image_dir = f"{backend.host_dir}/images"
            container_image_dir = f"{backend.container_dir}/images"

            host_references_file = f"{backend.host_dir}/references.json"
            container_references_file = f"{backend.container_dir}/references.json"

            host_output_file = f"{backend.host_dir}/scores.json"
            container_output_file = f"{backend.container_dir}/scores.json"

            # Write all of the candidates
            with open(host_candidate_file, "w") as out:
                out.write(
                    json.dumps(
                        {str(i): candidate for i, candidate in enumerate(candidates)}
                    )
                )

            # Copy over all of the images
            os.makedirs(host_image_dir)
            for i, image_file in enumerate(image_files):
                # The extension already includes the "."
                _, extension = os.path.splitext(image_file)
                shutil.copy(image_file, f"{host_image_dir}/{i}{extension}")

            if has_references:
                with open(host_references_file, "w") as out:
                    out.write(
                        json.dumps(
                            {
                                str(i): references
                                for i, references in enumerate(references_list)
                            }
                        )
                    )

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            commands.append("cd clipscore")

            score_command = (
                f"python clipscore.py "
                f"  {container_candidate_file} "
                f"  {container_image_dir} "
                f"  --save_per_instance {container_output_file}"
                f"  --compute_other_ref_metrics 0"
            )
            if has_references:
                score_command += f" --references_json {container_references_file}"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            micro_dict = json.load(open(host_output_file, "r"))
            micro_metrics = [micro_dict[str(i)] for i in range(len(candidates))]
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
