import logging
import os
from typing import List

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file, write_to_jsonl_file

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
