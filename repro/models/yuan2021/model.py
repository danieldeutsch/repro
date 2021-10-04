import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.yuan2021 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-bartscore")
class BARTScore(Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        model: str = "default",
        device: int = 0,
    ):
        assert model in ["default", "cnn", "parabank"]
        self.image = image
        self.model = model
        self.device = device

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
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
        batch_size: int = 4,
        **kwargs,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(
            f"Calculating BARTScore with image {self.image} on {len(inputs)} inputs."
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

            with open(host_input_file, "w") as out:
                for candidate, references in zip(candidates, references_list):
                    out.write(
                        json.dumps(
                            {
                                "candidate": candidate,
                                "references": references,
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
                predict_device = 0
            else:
                predict_device = -1

            commands.append("cd BARTScore")

            score_command = (
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --device {predict_device}"
                f"  --batch-size {batch_size}"
                f"  --model {self.model}"
                f"  --output-file {container_output_file}"
            )
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            micro_metrics = read_jsonl_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
