import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import write_to_text_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.rei2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-comet")
class COMET(Model):
    def __init__(self, image: str = DEFAULT_IMAGE, device: int = 0):
        self.image = image
        self.device = device

    def predict(
        self,
        candidate: TextType,
        sources: List[TextType] = None,
        references: List[TextType] = None,
        batch_size: int = None,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources, "references": references}],
            batch_size=batch_size,
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
        batch_size: int = None,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating COMET for {len(inputs)} inputs")

        batch_size = batch_size or 8

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] if "sources" in inp else None for inp in inputs]
        references_list = [
            inp["references"] if "references" in inp else None for inp in inputs
        ]

        # If any input has a reference, they all must
        def _has_references(references: List[TextType]) -> bool:
            return references is not None and len(references) > 0

        has_references = any(
            _has_references(references) for references in references_list
        )
        if has_references:
            if not all(_has_references(references) for references in references_list):
                raise Exception(
                    f"COMET requires all or none of the inputs have references"
                )

        # COMET only supports single sources and references
        sources = util.check_for_single_texts(sources_list)
        if has_references:
            references = util.check_for_single_texts(references_list)

        # Ensure all are strings or None
        candidates = [util.flatten(candidate) for candidate in candidates]
        sources = [util.flatten(source) for source in sources]
        if has_references:
            references = [util.flatten(reference) for reference in references]

        with DockerContainer(self.image) as backend:
            host_src_file = f"{backend.host_dir}/src.txt"
            container_src_file = f"{backend.container_dir}/src.txt"
            write_to_text_file(sources, host_src_file)

            hyp_filename = f"hyp1.txt"
            host_hyp_file = f"{backend.host_dir}/{hyp_filename}"
            container_hyp_file = f"{backend.container_dir}/{hyp_filename}"
            write_to_text_file(candidates, host_hyp_file)

            host_ref_file = f"{backend.host_dir}/ref.txt"
            container_ref_file = f"{backend.container_dir}/ref.txt"
            if has_references:
                write_to_text_file(references, host_ref_file)

            host_output_file = f"{backend.host_dir}/output.json"
            container_output_file = f"{backend.container_dir}/output.json"

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
                num_gpus = 1
            else:
                num_gpus = 0

            score_command = (
                f"comet-score "
                f"-s {container_src_file} "
                f"-t {container_hyp_file} "
                f"--gpus {num_gpus} "
                f"--batch_size {batch_size} "
                f"--to_json {container_output_file}"
            )
            if has_references:
                score_command += f" -r {container_ref_file} --model wmt20-comet-da"
            else:
                score_command += " --model wmt20-comet-qe-da"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(command=command, cuda=cuda, network_disabled=False)

            output_dict = json.load(open(host_output_file, "r"))
            outputs = output_dict[container_hyp_file]

            metric = "comet" if has_references else "comet-src"

            micro = []
            for output in outputs:
                micro.append({metric: output["COMET"]})
            macro = util.average_dicts(micro)
            return macro, micro
