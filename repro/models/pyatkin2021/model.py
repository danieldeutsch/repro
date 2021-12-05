import json
import logging
from typing import Any, Dict, List

from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.models import Model
from repro.models.pyatkin2021 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-role-question-generator")
class RoleQuestionGenerator(Model):
    def __init__(self, image: str = DEFAULT_IMAGE, device: int = 0):
        self.image = image
        self.device = device

    def predict(
        self,
        sentence: str,
        token_index: int,
        lemma: str,
        pos: str,
        sense: int,
    ) -> Dict:
        return self.predict_batch(
            [
                {
                    "sentence": sentence,
                    "token_index": token_index,
                    "lemma": lemma,
                    "pos": pos,
                    "sense": sense,
                }
            ]
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Any]],
    ) -> List[Dict]:
        logger.info(f"Generating Role Questions for {len(inputs)} inputs")

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"

            with open(host_input_file, "w") as out:
                for i, inp in enumerate(inputs):
                    out.write(
                        json.dumps(
                            {
                                "id": i,
                                "sentence": inp["sentence"],
                                "target_idx": inp["token_index"],
                                "target_lemma": inp["lemma"],
                                "target_pos": inp["pos"],
                                "predicate_sense": inp["sense"],
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            commands = []
            cuda = self.device != -1
            if cuda:
                process_device = 0
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            else:
                process_device = -1

            commands.append("cd RoleQGeneration")
            commands.append(
                f"python predict_questions.py"
                f"  --infile {container_input_file}"
                f"  --outfile {container_output_file}"
                f"  --transformation_model_path ../question_transformation_grammar_corrected_who"
                f"  --device_number {process_device}"
                f"  --with_adjuncts"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            outputs = read_jsonl_file(host_output_file)
            return outputs
