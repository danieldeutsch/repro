import json
import logging
from typing import Dict, List

from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.models import Model
from repro.models.kitaev2019 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-benepar")
class Benepar(Model):
    def __init__(
        self, image: str = DEFAULT_IMAGE, model: str = "benepar_en3", device: int = 0
    ):
        self.image = image
        self.model = model
        self.device = device

    def predict(self, text: str, **kwargs) -> List[str]:
        return self.predict_batch(
            [{"text": text}],
            **kwargs,
        )[0]

    def predict_batch(self, inputs: List[Dict[str, str]], **kwargs) -> List[List[str]]:
        logger.info(f"Parsing {len(inputs)} inputs with model {self.model}")

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            with open(host_input_file, "w") as out:
                for inp in inputs:
                    out.write(json.dumps({"text": inp["text"]}) + "\n")

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            commands.append(
                f"python parse.py"
                f"  --input-file {container_input_file}"
                f"  --model {self.model}"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(command=command, network_disabled=True, cuda=cuda)
            parses = [output["parses"] for output in read_jsonl_file(host_output_file)]
            return parses
