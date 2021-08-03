import json
import logging
import os
from typing import Any, Dict, List, Tuple, Union

from repro.common import TemporaryDirectory, util
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model

logger = logging.getLogger(__name__)


@Model.register("papineni2002-sentbleu")
class SentBLEU(Model):
    def __init__(
        self,
        image: str = "papineni2002",
    ):
        self.image = image

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
        logger.info(f"Calculating SentBLEU for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # Ensure they are `str`
        candidates = [util.flatten(candidate) for candidate in candidates]
        references_list = [
            [util.flatten(reference) for reference in references]
            for references in references_list
        ]

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
                        json.dumps({"candidate": candidate, "references": references})
                        + "\n"
                    )

            host_output_file = f"{host_output_dir}/output.jsonl"
            container_output_file = f"{container_output_dir}/output.jsonl"

            command = (
                f"python sentbleu.py"
                f"  --input-file {container_input_file}"
                f"  --output-file {container_output_file}"
            )

            os.makedirs(host_output_dir)
            run_command(
                self.image, command, volume_map=volume_map, network_disabled=True
            )

            micro_metrics = read_jsonl_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics


@Model.register("papineni2002-bleu")
class BLEU(Model):
    def __init__(
        self,
        image: str = "papineni2002",
    ):
        self.image = image

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
        logger.info(f"Calculating BLEU for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # Ensure they are `str`
        candidates = [util.flatten(candidate) for candidate in candidates]
        references_list = [
            [util.flatten(reference) for reference in references]
            for references in references_list
        ]

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
                        json.dumps({"candidate": candidate, "references": references})
                        + "\n"
                    )

            host_output_file = f"{host_output_dir}/output.json"
            container_output_file = f"{container_output_dir}/output.json"

            command = (
                f"python bleu.py"
                f"  --input-file {container_input_file}"
                f"  --output-file {container_output_file}"
            )

            os.makedirs(host_output_dir)
            run_command(
                self.image, command, volume_map=volume_map, network_disabled=True
            )

            # BLEU is corpus-level, so there's no micro
            macro_metrics = json.load(open(host_output_file, "r"))
            micro_metrics = []
            return macro_metrics, micro_metrics
