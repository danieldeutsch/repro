import json
import logging
import os
from typing import Dict, List, Tuple, Union

from repro.common import TemporaryDirectory, util
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.zhang2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-bertscore")
class BERTScore(Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        model: str = None,
        device: int = 0,
        batch_size: int = 64,
        language: str = "en",
    ):
        self.image = image
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.language = language

    def predict(
        self,
        candidate: TextType,
        references: List[TextType],
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(
            f"Calculating BERTScore with image {self.image} on {len(inputs)} inputs."
        )

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # The each candidate and reference must be `str`, not `List[str]`
        candidates = [util.flatten(candidate) for candidate in candidates]
        references_list = [
            [util.flatten(reference) for reference in references]
            for references in references_list
        ]

        # The metric code will run if we provide it an empty input, but we want to
        # assign a value of 0 to that candidate. Remove the empty inputs here, then
        # add 0 values at the end
        empty_indices, candidates, references_list = util.remove_empty_inputs(
            candidates, references_list
        )

        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_input_file = f"{host_input_dir}/input.jsonl"
            container_input_file = f"{container_input_dir}/input.jsonl"
            os.makedirs(host_input_dir)
            with open(host_input_file, "w") as out:
                for candidate, references in zip(candidates, references_list):
                    out.write(
                        json.dumps(
                            {
                                "candidate": candidate,
                                "references": references,
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{host_output_dir}/output.jsonl"
            container_output_file = f"{container_output_dir}/output.jsonl"

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
                predict_device = 0
            else:
                predict_device = -1

            score_command = (
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --cuda-device {predict_device}"
                f"  --batch-size {self.batch_size}"
                f"  --output-file {container_output_file}"
            )
            if self.model is not None:
                score_command += f" --model-name {self.model}"
            if self.language is not None:
                score_command += f"  --language {self.language}"
            commands.append(score_command)

            command = " && ".join(commands)
            os.makedirs(host_output_dir)
            run_command(
                self.image,
                command,
                volume_map=volume_map,
                cuda=cuda,
            )

            micro_metrics = read_jsonl_file(host_output_file)
            micro_metrics = [{"bertscore": scores} for scores in micro_metrics]

            # Insert default metric values for inputs which were empty. All of the
            # scored inputs should have the same keys, so we can get a default set
            # of metrics from just the first set of metrics
            empty_value = util.get_default_dict(micro_metrics[0], default=0.0)
            micro_metrics = util.insert_empty_values(
                micro_metrics, empty_indices, empty_value
            )

            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
