import json
import logging
import os
from typing import Dict, List, Union

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file, write_to_jsonl_file
from repro.data.types import MetricsType, SummaryType
from repro.models import Model
from repro.models.sacrerouge import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


def sentence_split(image: str, inputs: List[str]) -> List[List[str]]:
    """
    Run sentence splitting on the inputs.

    Parameters
    ----------
    image : str
        The name of the SacreROUGE Docker image to use.
    inputs : List[str]
        The texts which should be sentence-split.
    Returns
    -------
    List[List[str]]
        The sentence-split text.
    """
    with TemporaryDirectory() as temp:
        host_input_dir = f"{temp}/input"
        host_output_dir = f"{temp}/output"
        volume_map = make_volume_map(host_input_dir, host_output_dir)
        container_input_dir = volume_map[host_input_dir]
        container_output_dir = volume_map[host_output_dir]

        host_input_file = f"{host_input_dir}/input.txt"
        host_output_file = f"{host_output_dir}/output.txt"
        container_input_file = f"{container_input_dir}/input.txt"
        container_output_file = f"{container_output_dir}/output.txt"
        write_to_jsonl_file(inputs, "text", host_input_file)

        command = (
            f"python sentence_split.py {container_input_file} {container_output_file}"
        )
        os.makedirs(host_output_dir)
        run_command(image, command, volume_map=volume_map)

        outputs = read_jsonl_file(host_output_file)
        sentences = [output["sentences"] for output in outputs]
        return sentences


@Model.register(f"{MODEL_NAME}-rouge")
class SRROUGE(Model):
    def __init__(self, image: str = DEFAULT_IMAGE):
        self.image = image

    def _maybe_sentence_split(self, summaries: List[SummaryType]) -> List[List[str]]:
        if any(isinstance(summary, str) for summary in summaries):
            if not all(isinstance(summary, str) for summary in summaries):
                raise Exception(
                    f"Input summaries or references are mixed between strings and lists of strings. "
                    f"All must be of the same type"
                )
            return sentence_split(self.image, summaries)
        else:
            return summaries

    def _maybe_sentence_split_references(
        self, references_list: List[List[SummaryType]]
    ) -> List[List[List[str]]]:
        # Flatten the references into a single list so we can call `_maybe_sentence_split`, then
        # rearrange the output to be parallel to `references_list`
        flat_references = []
        for references in references_list:
            flat_references.extend(references)

        split_references = self._maybe_sentence_split(flat_references)
        split_references_list = []
        index = 0
        for references in references_list:
            split_references_list.append([])
            for _ in references:
                split_references_list[-1].append(split_references[index])
                index += 1
        return split_references_list

    def predict(
        self, summary: SummaryType, references: List[SummaryType], **kwargs
    ) -> MetricsType:
        return self.predict_batch(
            [{"summary": summary, "references": references}], **kwargs
        )

    def predict_batch(
        self, inputs: List[Dict[str, Union[str, List[str]]]], **kwargs
    ) -> MetricsType:
        logger.info(f"Calculating ROUGE for {len(inputs)} inputs")

        summaries = [inp["summary"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        summaries = self._maybe_sentence_split(summaries)
        references_list = self._maybe_sentence_split_references(references_list)

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
                for i, (summary, references) in enumerate(
                    zip(summaries, references_list)
                ):
                    summary = {"text": summary}
                    references = [{"text": reference} for reference in references]
                    out.write(
                        json.dumps(
                            {
                                "instance_id": str(i),
                                "summarizer_id": "repro",
                                "summarizer_type": "peer",
                                "summary": summary,
                                "references": references,
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{host_output_dir}/macro.json"
            container_output_file = f"{container_output_dir}/macro.json"
            command = (
                f"sacrerouge rouge evaluate"
                f"  --input-files {container_input_file}"
                f"  --compute_rouge_l true"
                f"  --dataset-reader reference-based"
                f"  --macro-output-json {container_output_file}"
                f"  --micro-output-jsonl {container_output_dir}/micro.jsonl"
            )
            os.makedirs(host_output_dir)
            run_command(
                self.image, command, volume_map=volume_map, network_disabled=True
            )

            scores = json.load(open(host_output_file, "r"))
            return scores["metrics"]
