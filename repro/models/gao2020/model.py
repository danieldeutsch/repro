import json
import logging
import os
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.gao2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-supert")
class SUPERT(Model):
    def __init__(self, image: str = DEFAULT_IMAGE) -> None:
        self.image = image

    @staticmethod
    def _write_sources(sources: List[TextType], output_dir: str) -> None:
        os.makedirs(output_dir)
        for i, source in enumerate(sources):
            source = util.flatten(source)
            with open(f"{output_dir}/{i}.txt", "w") as out:
                out.write("<TEXT>\n")
                out.write(source + "\n")
                out.write("</TEXT>\n")

    @staticmethod
    def _write_candidates(candidates: List[TextType], output_dir: str) -> None:
        os.makedirs(output_dir)
        for i, candidate in enumerate(candidates):
            with open(f"{output_dir}/{i}", "w") as out:
                candidate = util.flatten(candidate)
                out.write(candidate)

    def predict(self, sources: List[TextType], candidate: TextType) -> MetricsType:
        return self.predict_batch([{"sources": sources, "candidate": candidate}])[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating SUPERT for {len(inputs)} inputs")

        sources_list = [inp["sources"] for inp in inputs]
        candidates = [inp["candidate"] for inp in inputs]

        # Group the candidates by the sources for more efficient processing.
        # This method groups by "references," but it will do the same thing
        (
            grouped_candidates_list,
            grouped_sources_list,
            group_mapping,
        ) = util.group_by_references(candidates, sources_list)

        with DockerContainer(self.image) as backend:
            host_input_dir = f"{backend.host_dir}/input"
            container_input_dir = f"{backend.container_dir}/input"

            # This could be optimized by grouping the candidates by identical sources, but
            # we haven't implemented that currently
            for i, (sources, candidates) in enumerate(
                zip(grouped_sources_list, grouped_candidates_list)
            ):
                output_dir = f"{host_input_dir}/{i}"
                self._write_sources(sources, f"{output_dir}/input_docs")
                self._write_candidates(candidates, f"{output_dir}/summaries")

            host_output_file = f"{backend.host_dir}/output.json"
            container_output_file = f"{backend.container_dir}/output.json"

            commands = ["cd SUPERT"]
            commands.append(
                f"python run_batch.py {container_input_dir} {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=False,
                network_disabled=True,
            )

            # The keys of this dict are strings, but they need to be indices to
            # undo the grouping, so we convert them
            str_scores_dict = json.load(open(host_output_file, "r"))
            int_scores_dict = {
                int(group_index): {
                    int(candidate_index): {"supert": value}
                    for candidate_index, value in str_scores_dict[group_index].items()
                }
                for group_index in str_scores_dict.keys()
            }

            micro_metrics = util.ungroup_values(int_scores_dict, group_mapping)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics
