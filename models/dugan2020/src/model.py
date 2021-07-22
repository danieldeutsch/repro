import json
import logging
import os
from typing import Dict, List

from overrides import overrides

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file
from repro.models import Model, RecipeGenerationModel

logger = logging.getLogger(__name__)


@Model.register("dugan2020-roft-recipe")
class RoFTRecipeGenerator(RecipeGenerationModel):
    def __init__(
        self,
        image: str = "dugan2020",
        device: int = 0,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        max_length: int = 256,
        random_seed: int = 4,
    ) -> None:
        self.model = "gpt2-xl"
        self.image = image
        self.device = device
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.random_seed = random_seed

    @overrides
    def predict_batch(self, inputs: List[Dict[str, str]], *args, **kwargs) -> List[str]:
        logger.info(
            f"Generating recipes for {len(inputs)} inputs and Docker image {self.image}"
        )

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
                            {"name": inp["name"], "ingredients": inp["ingredients"]}
                        )
                        + "\n"
                    )

            # Run inference. The output_dir must exist before running
            # the docker command
            os.makedirs(host_output_dir)
            host_output_file = f"{host_output_dir}/output.jsonl"
            container_output_file = f"{container_output_dir}/output.jsonl"

            command = (
                f"python generate_recipes.py"
                f"  --input-file {container_input_file}"
                f"  --model-name {self.model}"
                f"  --top-p {self.top_p}"
                f"  --repetition-penalty {self.repetition_penalty}"
                f"  --max-length {self.max_length}"
                f"  --random-seed {self.random_seed}"
                f"  --output-file {container_output_file}"
            )

            cuda = self.device != -1
            if cuda:
                command = f"CUDA_VISIBLE_DEVICES={self.device} " + command

            run_command(
                self.image,
                command,
                volume_map=volume_map,
                network_disabled=True,
                cuda=cuda,
            )

            results = read_jsonl_file(host_output_file, single_line=False)
            recipes = [result["recipe"] for result in results]
            return recipes
