import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.zhang2021 import DEFAULT_IMAGE, MODEL_NAME

METRICS = ["p2c", "l2c", "p3c", "l3c"]


logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-lite3pyramid")
class Lite3Pyramid(Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        model: str = None,
    ):
        self.image = image
        self.device = device
        self.model = model

    def extract_stus(
        self, texts: Union[str, List[str]], use_coref: bool
    ) -> Union[List[str], List[List[str]]]:
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]

        logger.info(f"Extracting STUs for {len(texts)} inputs")
        with DockerContainer(self.image) as backend:
            host_summaries_file = f"{backend.host_dir}/summaries.txt"
            host_ids_file = f"{backend.host_dir}/ids.txt"
            container_summaries_file = f"{backend.container_dir}/summaries.txt"
            container_ids_file = f"{backend.container_dir}/ids.txt"

            # Serialize the input data
            with open(host_summaries_file, "w") as out_summaries:
                with open(host_ids_file, "w") as out_ids:
                    for i, text in enumerate(texts):
                        out_summaries.write(text + "\n")
                        out_ids.write(str(i) + "\n")

            # Run the STU extraction
            host_output_dir = f"{backend.host_dir}/output"
            container_output_dir = f"{backend.container_dir}/output"

            commands = [f"mkdir -p {container_output_dir}", f"cd Lite2-3Pyramid"]
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            extract_command = (
                f"python score.py"
                f"  --extract_stus"
                f"  --reference {container_summaries_file}"
                f"  --doc_id {container_ids_file}"
                f"  --output_dir {container_output_dir}"
            )
            if use_coref:
                extract_command += " --use_coref"
            commands.append(extract_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            # Load the results
            outputs = []
            with open(f"{host_output_dir}/STUs.txt", "r") as f:
                for line in f:
                    stus = line.strip().split("\t")
                    outputs.append(stus)

            return outputs[0] if is_single_input else outputs

    def _assert_valid_inputs(self, references_list: List, units_lists: List) -> None:
        # The inputs must either have references xor units, and all input instances
        # must be the same
        has_references = [references is not None for references in references_list]
        has_units = [units_list is not None for units_list in units_lists]

        if not (all(has_references) or all(has_units)):
            raise Exception(f"Inputs must either all have references xor units.")
        if (all(has_references) and any(has_units)) or (
            any(has_references) and all(has_units)
        ):
            raise Exception(f"Inputs must either all have references xor units.")

    def _extract_stus_from_references(
        self, references_list: List[List[TextType]], use_coref: bool
    ) -> List[List[List[str]]]:
        flat_references = []
        reference_to_index = {}
        indices_list = []
        for references in references_list:
            indices_list.append([])
            for reference in references:
                reference = util.flatten(reference)
                if reference not in reference_to_index:
                    reference_to_index[reference] = len(flat_references)
                    flat_references.append(reference)
                indices_list[-1].append(reference_to_index[reference])

        # Extract the STUs
        stus_list = self.extract_stus(flat_references, use_coref)

        # Rematch the inputs
        outputs = []
        for indices in indices_list:
            outputs.append([])
            for index in indices:
                # Make a copy so the STUs aren't the same object in case any
                # downstream processing assumes they aren't
                outputs[-1].append(stus_list[index].copy())

        return outputs

    def predict(
        self,
        candidate: TextType,
        references: List[TextType] = None,
        units_list: List[List[str]] = None,
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [
                {
                    "candidate": candidate,
                    "references": references,
                    "units_list": units_list,
                }
            ],
            **kwargs,
        )[0]

    def predict_batch(
        self, inputs: List[Dict], use_coref: bool = False, **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Lite3Pyramid for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [
            inp["references"] if "references" in inp else None for inp in inputs
        ]
        units_lists = [
            inp["units_list"] if "units_list" in inp else None for inp in inputs
        ]

        self._assert_valid_inputs(references_list, units_lists)

        candidates = [util.flatten(candidate) for candidate in candidates]

        # If the inputs have references, we first need to extract their STUs
        has_references = any(references is not None for references in references_list)
        if has_references:
            units_lists = self._extract_stus_from_references(references_list, use_coref)

        with DockerContainer(self.image) as backend:
            host_candidates_file = f"{backend.host_dir}/candidates.txt"
            host_units_file = f"{backend.host_dir}/units.txt"
            container_candidates_file = f"{backend.container_dir}/candidates.txt"
            container_units_file = f"{backend.container_dir}/units.txt"

            # Serialize the input data
            with open(host_candidates_file, "w") as out_candidates:
                with open(host_units_file, "w") as out_units:
                    for candidate, units_list in zip(candidates, units_lists):
                        for units in units_list:
                            out_candidates.write(candidate + "\n")
                            out_units.write("\t".join(units) + "\n")

            host_output_file = f"{backend.host_dir}/scores.json"
            container_output_file = f"{backend.container_dir}/scores.json"

            # Run the scoring
            commands = [f"cd Lite2-3Pyramid"]
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            score_command = (
                f"python score.py"
                f"  --summary {container_candidates_file}"
                f"  --unit {container_units_file}"
                f"  --detail"
                f"  --output_file {container_output_file}"
            )
            if self.model is not None:
                score_command += f" --model {self.model}"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            scores = json.load(open(host_output_file, "r"))

            # There is a score for every candidate-reference pair, so we need
            # to group them in the event of multiple references. The `scores` dict
            # has a key for every metric which maps to a tuple with the macro
            # score as well as the micro scores
            grouped_micro_metrics = []
            index = 0
            for units_list in units_lists:
                grouped_micro_metrics.append([])
                for _ in units_list:
                    grouped_micro_metrics[-1].append(
                        {metric: scores[metric][1][index] for metric in METRICS}
                    )
                    index += 1

            micro_metrics = [
                util.average_dicts(group) for group in grouped_micro_metrics
            ]
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
