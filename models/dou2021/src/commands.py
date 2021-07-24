import os
from typing import List

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import write_to_text_file
from repro.data.types import DocumentType, SummaryType


def sentence_split(image: str, inputs: List[str]) -> List[List[str]]:
    """
    Run sentence splitting on the inputs.

    Parameters
    ----------
    image : str
        The name of the Dou et al. (2021) Docker image to use.
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
        write_to_text_file(inputs, host_input_file)

        command = f"python sentence_tokenize.py {container_input_file} {container_output_file}"
        os.makedirs(host_output_dir)
        run_command(image, command, volume_map=volume_map)

        split = [
            line.split("<q>")
            for line in open(host_output_file, "r").read().splitlines()
        ]
        if len(split) != len(inputs):
            raise Exception(
                f"Sentence splitting returned {len(split)} instances but expected {len(inputs)}"
            )
        return split


def get_oracle_sentences(
    image: str, documents: List[List[str]], references: List[List[str]]
) -> List[List[str]]:
    """
    Extracts the oracle sentences from the documents based on the ROUGE
    score calculated against the reference summaries.

    Parameters
    ----------
    image : str
        The name of the Dou et al. (2021) Docker image to use.
    documents : List[List[str]]
        The sentence-split documents
    references : List[List[str]]
        The sentence-split reference summaries

    Returns
    -------
    List[List[str]]
        The oracle sentence-extractive summaries
    """
    with TemporaryDirectory() as temp:
        host_input_dir = f"{temp}/input"
        host_output_dir = f"{temp}/output"
        volume_map = make_volume_map(host_input_dir, host_output_dir)
        container_input_dir = volume_map[host_input_dir]
        container_output_dir = volume_map[host_output_dir]

        host_document_file = f"{host_input_dir}/documents.txt"
        host_reference_file = f"{host_input_dir}/references.txt"
        container_document_file = f"{container_input_dir}/documents.txt"
        container_reference_file = f"{container_input_dir}/references.txt"
        write_to_text_file(documents, host_document_file, separator="<q>")
        write_to_text_file(references, host_reference_file, separator="<q>")

        host_output_file = f"{host_output_dir}/guidance.txt"
        container_output_file = f"{container_output_dir}/guidance.txt"

        commands = [
            f"cd guided_summarization/scripts",
            f"python sents.py {container_document_file} {container_reference_file} {container_output_file}",
        ]
        command = " && ".join(commands)
        os.makedirs(host_output_dir)
        run_command(image, command, volume_map=volume_map)

        guidance = [
            line.split("<q>")
            for line in open(host_output_file, "r").read().splitlines()
        ]
        return guidance


def generate_summaries(
    image: str,
    model: str,
    device: int,
    batch_size: int,
    documents: List[DocumentType],
    guidance: List[SummaryType],
) -> List[str]:
    """
    Generates summaries using a pre-trained model and guidance signals.

    Parameters
    ----------
    image : str
        The name of the Dou et al. (2021) Docker image to use.
    model : str
        The name of the model to use
    device : int
        The GPU ID, -1 for CPU
    batch_size : int
        The batch size
    documents : List[DocumentType]
        The input documents
    guidance : List[SummaryType]
        The guidance signals

    Returns
    -------
    List[str]
        The summaries
    """
    with TemporaryDirectory() as temp:
        host_input_dir = f"{temp}/input"
        host_output_dir = f"{temp}/output"
        volume_map = make_volume_map(host_input_dir, host_output_dir)
        container_input_dir = volume_map[host_input_dir]
        container_output_dir = volume_map[host_output_dir]

        host_document_file = f"{host_input_dir}/documents.txt"
        host_guidance_file = f"{host_input_dir}/guidance.txt"
        container_document_file = f"{container_input_dir}/documents.txt"
        container_guidance_file = f"{container_input_dir}/guidance.txt"
        write_to_text_file(documents, host_document_file)
        write_to_text_file(guidance, host_guidance_file)

        host_output_file = f"{host_output_dir}/summaries.txt"
        container_output_file = f"{container_output_dir}/summaries.txt"

        commands = ["cd guided_summarization/bart"]
        inference_command = (
            f"python summarize.py"
            f"  {container_document_file}"
            f"  {container_guidance_file}"
            f"  {container_output_file}"
            f"  ../{model}"
            f"  model.pt"
            f"  ../{model}"
            f"  {batch_size}"
        )
        cuda = device != -1
        if cuda:
            inference_command = f"CUDA_VISIBLE_DEVICES={device} " + inference_command
        commands.append(inference_command)
        command = " && ".join(commands)

        os.makedirs(host_output_dir)
        run_command(image, command, volume_map=volume_map, cuda=cuda)

        summaries = open(host_output_file, "r").read().splitlines()
        return summaries
