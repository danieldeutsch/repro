import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model

logger = logging.getLogger(__name__)


@Model.register("durmus2020-feqa")
class FEQA(Model):
    def __init__(self, image: str = "durmus2020", device: int = 0):
        self.image = image
        self.device = device

    @staticmethod
    def _check_sources(sources_list: List[List[TextType]]) -> List[TextType]:
        single_sources = []
        for sources in sources_list:
            if len(sources) != 1:
                raise Exception(
                    f"FEQA only supports single sources. Found: {len(sources)}"
                )
            single_sources.append(sources[0])
        return single_sources

    def predict(
        self,
        candidate: TextType,
        sources: List[TextType],
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources}], **kwargs
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
        batch_size: int = 16,
        **kwargs,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating FEQA for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] for inp in inputs]

        # FEQA only supports single source documents
        sources = self._check_sources(sources_list)

        # Ensure they are all type `str`
        candidates = [util.flatten(candidate) for candidate in candidates]
        sources = [util.flatten(source) for source in sources]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"
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

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
                score_device = 0
            else:
                score_device = -1

            commands.append("cd feqa")
            commands.append(
                f"python3 score.py"
                f"  --input-file {container_input_file}"
                f"  --cuda-device {score_device}"
                f"  --batch-size {batch_size}"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(command=command, cuda=cuda, network_disabled=True)

            micro_metrics = read_jsonl_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
