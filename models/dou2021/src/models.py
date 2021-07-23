import logging
import os
from typing import Dict, List, Union

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import write_to_text_file
from repro.models import Model
from repro.models.model import DocumentType, SummaryType

logger = logging.getLogger(__name__)


def _get_prediction_command(
    model: str, document_file: str, guidance_file: str, output_file: str, device: int
) -> str:
    train_command = (
        f"python summarize.py"
        f"  {document_file}"
        f"  {guidance_file}"
        f"  {output_file}"
        f"  ../{model}"
        f"  model.pt"
        f"  ../{model}"
    )
    cuda = device != -1
    if cuda:
        train_command = f"CUDA_VISIBLE_DEVICES={device} " + train_command

    commands = [
        "cd guided_summarization/bart",
        train_command,
        "cd ../..",
    ]
    return " && ".join(commands)


@Model.register("dou2021-oracle-gsum")
class OracleGSumModel(Model):
    def __init__(self, image: str = "dou2021", device: int = 0) -> None:
        # The sentence-guided model is the only one available
        self.model = "bart_sentence"
        self.image = image
        self.device = device

    @staticmethod
    def _get_sentence_oracle_command(
        document_file: str, reference_file: str, output_file: str
    ) -> str:
        commands = [
            "cd guided_summarization/scripts",
            f"python sents.py {document_file} {reference_file} {output_file}",
            "cd ../.."
        ]
        return " && ".join(commands)

    @staticmethod
    def _get_de_sentence_tokenize_command(input_file: str, output_file: str) -> str:
        return f'sed "s/<q>/ /g" {input_file} > {output_file}'

    def predict(
        self, document: DocumentType, reference: SummaryType, *args, **kwargs
    ) -> SummaryType:
        return self.predict_batch([{"document": document, "reference": reference}])[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[DocumentType, SummaryType]]], *args, **kwargs
    ) -> List[SummaryType]:
        logger.info(
            f"Generating summaries for {len(inputs)} inputs and image {self.image}."
        )

        documents = [inp["document"] for inp in inputs]
        references = [inp["reference"] for inp in inputs]
        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            # Write the input documents and references. For extracting the guidance signal,
            # the input documents and references are expected to be sentence-split by "<q>",
            # but the original document is used for inference (the reference is not used).
            # Therefore we write two versions of the input documents
            host_document_file = f"{host_input_dir}/input.source"
            host_document_tok_file = f"{host_input_dir}/input.tokenized.source"
            host_reference_file = f"{host_input_dir}/input.ref"
            container_document_file = f"{container_input_dir}/input.source"
            container_document_tok_file = (
                f"{container_input_dir}/input.tokenized.source"
            )
            container_reference_file = f"{container_input_dir}/input.ref"
            write_to_text_file(documents, host_document_file)
            write_to_text_file(documents, host_document_tok_file, separator="<q>")
            write_to_text_file(references, host_reference_file, separator="<q>")

            # Run inference. The host output directory must already exist
            os.makedirs(host_output_dir)

            # Get the command to run extracting the oracle guidance. This uses
            # the sentence tokenized documents.
            container_guidance_tok_file = f"{container_output_dir}/input.tokenized.z"
            guidance_command = self._get_sentence_oracle_command(
                container_document_tok_file,
                container_reference_file,
                container_guidance_tok_file,
            )

            # The guidance file is sentence tokenized, but inference requires
            # the untokenized version. This command removes "<q>" from the text
            container_guidance_file = f"{container_output_dir}/input.z"
            detokenization_command = self._get_de_sentence_tokenize_command(
                container_guidance_tok_file, container_guidance_file
            )

            # Run inference with the original documents and de-tokenized guidance
            host_output_file = f"{host_output_dir}/input.output"
            container_output_file = f"{container_output_dir}/input.output"
            prediction_command = _get_prediction_command(
                self.model,
                container_document_file,
                container_guidance_file,
                container_output_file,
                self.device,
            )

            command = " && ".join(
                [guidance_command, detokenization_command, prediction_command]
            )
            run_command(self.image, command, volume_map=volume_map, cuda=True)

            summaries = open(host_output_file, "r").read().splitlines()
            return summaries
