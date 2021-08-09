import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.scialom2019 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-summaqa")
class SummaQA(Model):
    def __init__(self, image: str = DEFAULT_IMAGE):
        self.image = image

    def predict(
        self,
        candidate: TextType,
        sources: List[TextType],
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources}],
            **kwargs,
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating SummaQA for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] for inp in inputs]

        # SummaQA only supports single sources
        sources = util.check_for_single_texts(sources_list)

        # Ensure all are strings
        candidates = [util.flatten(candidate) for candidate in candidates]
        sources = [util.flatten(source) for source in sources]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            with open(host_input_file, "w") as out:
                for candidate, source in zip(candidates, sources):
                    out.write(
                        json.dumps(
                            {
                                "candidate": candidate,
                                "source": source,
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            command = (
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --output-file {container_output_file}"
            )

            backend.run_command(command=command, network_disabled=True)
            micro_metrics = read_jsonl_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
