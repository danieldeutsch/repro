import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.denkowski2014 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-meteor")
class METEOR(Model):
    def __init__(
        self, language: str = "en", norm: bool = True, image: str = DEFAULT_IMAGE
    ):
        self.language = language
        self.norm = norm
        self.image = image

    @staticmethod
    def _parse_output_file(file_path: str) -> List[Dict[str, float]]:
        lines = open(file_path, "r").read().splitlines()
        assert lines[11].startswith(
            "Segment 1 score"
        ), "Unexpected METEOR stdout format"

        scores = []
        for line in lines[11:]:
            if not line.startswith("Segment"):
                break
            columns = line.split()
            scores.append({"meteor": float(columns[3])})
        return scores

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
        logger.info(f"Calculating METEOR for {len(inputs)} inputs")

        references_list = [inp["references"] for inp in inputs]
        candidates = [inp["candidate"] for inp in inputs]

        # Ensure all have the same number of references
        num_references = len(references_list[0])
        for references in references_list:
            if len(references) != num_references:
                raise Exception(f"All inputs must have the same number of references")

        # Flatten the inputs
        references_list = [
            [util.flatten(reference) for reference in references]
            for references in references_list
        ]
        candidates = [util.flatten(candidate) for candidate in candidates]

        with DockerContainer(self.image) as backend:
            host_references_file = f"{backend.host_dir}/references.txt"
            container_references_file = f"{backend.container_dir}/references.txt"

            host_candidates_file = f"{backend.host_dir}/candidates.txt"
            container_candidates_file = f"{backend.container_dir}/candidates.txt"

            with open(host_references_file, "w") as out_references:
                with open(host_candidates_file, "w") as out_candidates:
                    # Each candidate is written once for the whole reference set. If there
                    # are N references, the reference file will be N times longer
                    for candidate, references in zip(candidates, references_list):
                        out_candidates.write(candidate + "\n")
                        for reference in references:
                            out_references.write(reference + "\n")

            host_output_file = f"{backend.host_dir}/output.txt"
            container_output_file = f"{backend.container_dir}/output.txt"

            commands = ["cd meteor-1.5"]
            score_command = (
                f"java -jar meteor-1.5.jar {container_candidates_file} {container_references_file}"
                f"  -r {num_references}"
                f"  -l {self.language}"
            )
            if self.norm:
                score_command += " -norm"
            score_command += f" > {container_output_file}"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=False,
                network_disabled=True,
            )

            # Load the scores
            micro_metrics = self._parse_output_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
