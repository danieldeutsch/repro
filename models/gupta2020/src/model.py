import json
import os
from overrides import overrides
from typing import Dict, List

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file
from repro.models import Model, QuestionAnsweringModel


@Model.register("gupta2020-nmn")
class NeuralModuleNetwork(QuestionAnsweringModel):
    def __init__(self, image: str = "gupta2020", device: int = 0) -> None:
        self.image = image
        self.device = device

    @overrides
    def predict_batch(self, inputs: List[Dict[str, str]], *args, **kwargs) -> List[str]:
        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_input_file = f"{host_input_dir}/input.jsonl"
            container_input_file = f"{container_input_dir}/input.jsonl"

            # Serialize the input to a file
            os.makedirs(host_input_dir)
            with open(host_input_file, "w") as out:
                for inp in inputs:
                    out.write(
                        json.dumps(
                            {"passage": inp["context"], "question": inp["question"]}
                        )
                        + "\n"
                    )

            # Run inference. The output_dir must exist before running
            # the docker command
            os.makedirs(host_output_dir)
            host_output_file = f"{host_output_dir}/output.jsonl"
            container_output_file = f"{container_output_dir}/output.jsonl"

            commands = []
            commands.append("cd nmn-drop")

            # If there is a GPU, we restrict the visible devices to that GPU.
            # Then, the `process_device` is the ID of the GPU for the predict command. After
            # restricting the visible devices to `self.device`, that device now has ID 0.
            cuda = self.device != -1
            if cuda:
                process_device = 0
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            else:
                process_device = -1

            commands.append(
                f"allennlp predict"
                f"  --output-file {container_output_file}"
                f"  --predictor drop_demo_predictor"
                f"  --include-package semqa"
                f"  --silent"
                f"  --batch-size 1"
                f"  --cuda-device {process_device}"
                f"  ../model.tar.gz"
                f"  {container_input_file}"
            )
            command = " && ".join(commands)

            run_command(
                self.image,
                command,
                volume_map=volume_map,
                network_disabled=True,
                cuda=cuda,
            )

            # The `drop_demo_predictor` does not separate outputs by \n. All
            # of the outputs are on the same line, so pass the `single_line=True`
            # flag to the loader
            results = read_jsonl_file(host_output_file, single_line=True)
            answers = [result["answer"] for result in results]
            return answers
