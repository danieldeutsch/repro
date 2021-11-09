import json
import logging
from typing import Dict, List

from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.models import Model
from repro.models.fitzgerald2018 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-qasrl-parser")
class QASRLParser(Model):
    def __init__(self, image: str = DEFAULT_IMAGE, device: int = 0) -> None:
        self.image = image
        self.device = device

    def predict(self, sentence: str, **kwargs) -> Dict:
        return self.predict_batch([{"sentence": sentence}], **kwargs)[0]

    def predict_batch(self, inputs: List[Dict[str, str]], **kwargs) -> List[Dict]:
        logger.info(f"Parsing {len(inputs)} inputs with image {self.image}")

        sentences = [inp["sentence"] for inp in inputs]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"

            host_output_file = f"{backend.host_dir}/scores.json"
            container_output_file = f"{backend.container_dir}/scores.json"

            with open(host_input_file, "w") as out:
                for sentence in sentences:
                    out.write(json.dumps({"sentence": sentence}) + "\n")

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
                predict_device = 0
            else:
                predict_device = -1

            commands.append("cd nrl-qasrl")
            commands.append(
                f"python -m allennlp.run predict "
                f"  ./data/qasrl_parser_elmo"
                f"  {container_input_file}"
                f"  --include-package nrl"
                f"  --predictor qasrl_parser"
                f"  --silent"
                f"  --cuda-device {predict_device}"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            outputs = read_jsonl_file(host_output_file)
            return outputs
