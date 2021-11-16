import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.vasilyev2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


class _BLANC(Model):
    def __init__(
        self,
        blanc_type: str,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        blanc_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.blanc_type = blanc_type
        self.image = image
        self.device = device
        self.blanc_kwargs = blanc_kwargs or {}

    def predict(self, sources: List[TextType], candidate: TextType) -> MetricsType:
        return self.predict_batch([{"sources": sources, "candidate": candidate}])[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating BLANC-{self.blanc_type} for {len(inputs)} inputs")

        sources_list = [inp["sources"] for inp in inputs]
        candidates = [inp["candidate"] for inp in inputs]

        # Ensure the texts are flattened
        sources_list = [
            [util.flatten(source) for source in sources] for sources in sources_list
        ]
        candidates = [util.flatten(candidate) for candidate in candidates]

        # Group the candidates by sources for more efficient processing
        (
            grouped_candidates_list,
            grouped_sources_list,
            group_mapping,
        ) = util.group_by_references(candidates, sources_list)

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"

            with open(host_input_file, "w") as out:
                for sources, candidates in zip(
                    grouped_sources_list, grouped_candidates_list
                ):
                    if len(sources) != 1:
                        raise Exception("BLANC only supports single-document summaries")
                    out.write(
                        json.dumps({"document": sources[0], "summaries": candidates})
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/output.json"
            container_output_file = f"{backend.container_dir}/output.json"

            commands = []
            # If there is a GPU, we restrict the visible devices to that GPU.
            # Then, the `process_device` is the ID of the GPU for the predict command. After
            # restricting the visible devices to `self.device`, that device now has ID 0.
            cuda = self.device != -1
            if cuda:
                process_device = 0
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            else:
                process_device = -1

            kwargs_str = json.dumps(self.blanc_kwargs)
            commands.append(
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --type {self.blanc_type}"
                f"  --device {process_device}"
                f"  --random-seed 123"
                f"  --kwargs '{kwargs_str}'"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            grouped_scores = json.load(open(host_output_file, "r"))
            micro_metrics = util.ungroup_values(grouped_scores, group_mapping)

            micro_metrics = [
                {f"blanc-{self.blanc_type}": score} for score in micro_metrics
            ]
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics


@Model.register(f"{MODEL_NAME}-blanc-help")
class BLANCHelp(_BLANC):
    def __init__(self, **kwargs):
        super().__init__("help", **kwargs)


@Model.register(f"{MODEL_NAME}-blanc-tune")
class BLANCTune(_BLANC):
    def __init__(self, **kwargs):
        super().__init__("tune", **kwargs)
