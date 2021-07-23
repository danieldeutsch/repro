import logging
import os
from overrides import overrides
from typing import Dict, List

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file, write_to_text_file
from repro.models import Model, TruecasingModel

logger = logging.getLogger(__name__)


@Model.register("susanto2016-truecaser")
class RNNTruecaser(TruecasingModel):
    def __init__(
        self,
        model: str,
        image: str = "susanto2016",
        device: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        model : str
            The name of the model, currently either "wiki-truecaser-model-en.tar.gz" for English,
            "wmt-truecaser-model-de.tar.gz" for German, "wmt-truecaser-model-es.tar.gz" for Spanish,
            or "lrl-truecaser-model-ru.tar.gz" for Russian.
        image : str, default="susanto2016"
            The name of the Docker image
        device : int, default=0
            The ID of the GPU, -1 if CPU
        """
        if model not in [
            "lrl-truecaser-model-ru.tar.gz",
            "wiki-truecaser-model-en.tar.gz",
            "wmt-truecaser-model-de.tar.gz",
            "wmt-truecaser-model-es.tar.gz",
        ]:
            raise Exception(f"Unknown model: {model}")

        self.model = model
        self.image = image
        self.device = device

    @overrides
    def predict_batch(self, inputs: List[Dict[str, str]], *args, **kwargs) -> List[str]:
        input_texts = [inp["text"] for inp in inputs]
        logger.info(f"Running truecasing on {len(inputs)} using image {self.image}")

        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_input_file = f"{host_input_dir}/input.txt"
            container_input_file = f"{container_input_dir}/input.txt"
            write_to_text_file(input_texts, host_input_file)

            # Run inference. The output_dir must exist before running
            # the docker command
            os.makedirs(host_output_dir)
            host_output_file = f"{host_output_dir}/output.txt"
            container_output_file = f"{container_output_dir}/output.txt"

            commands = []
            commands.append("cd pytorch-truecaser")

            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
                process_device = 0
            else:
                process_device = -1

            commands.append(
                f"allennlp predict"
                f"  ../{self.model}"
                f"  {container_input_file}"
                f"  --output-file {container_output_file}"
                f"  --include-package mylib "
                f"  --use-dataset-reader "
                f"  --predictor truecaser-predictor"
                f"  --cuda-device {process_device}"
                f"  --silent"
            )

            command = " && ".join(commands)
            run_command(
                self.image,
                command,
                volume_map=volume_map,
                network_disabled=True,
                cuda=cuda,
            )

            outputs = open(host_output_file, "r").read().splitlines()
            return outputs
