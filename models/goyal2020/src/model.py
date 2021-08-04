import json
import logging
from typing import Any, Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model

QAPairsType = List[List[Dict[str, Any]]]

logger = logging.getLogger(__name__)


@Model.register("goyal2020-dae")
class DAE(Model):
    def __init__(
        self,
        image: str = "goyal2020",
        model: str = "dae_w_syn_hallu",
        device: int = 0,
        sleep: int = 1,
    ):
        """
        Parameters
        ----------
        image : str, default="goyal2020"
            The name of the Docker image
        model : str, default="dae_w_syn_w_hallu"
            The name of the pre-trained DAE model to use
        device : int, default=0
            The ID of the GPU to use, -1 if CPU
        sleep : int, default=1
            The number of seconds to sleep while the Stanford CoreNLP server initializes
        """
        self.image = image
        self.model = model
        self.device = device
        self.sleep = sleep

    @staticmethod
    def _check_sources(sources_list: List[List[TextType]]) -> List[TextType]:
        single_sources = []
        for sources in sources_list:
            if len(sources) != 1:
                raise Exception(
                    f"DAE only supports single source documents. Found: {len(sources)}"
                )
            single_sources.append(sources[0])
        return single_sources

    def predict(
        self, candidate: TextType, sources: List[TextType], **kwargs
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources}], **kwargs
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
        **kwargs,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating DAE for {len(inputs)} inputs with model {self.model}")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] for inp in inputs]

        # DAE only accepts single source documents
        sources = self._check_sources(sources_list)

        # Make sure all are type `str`
        candidate = [util.flatten(candidate) for candidate in candidates]
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

            commands = []
            cuda = self.device != -1
            if cuda:
                score_device = 0
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            else:
                score_device = -1

            commands.append(
                f"sh run.sh"
                f"  {container_input_file}"
                f"  {container_output_file}"
                f"  {self.model}"
                f"  {score_device}"
                f"  {self.sleep}"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            micro_metrics = read_jsonl_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
