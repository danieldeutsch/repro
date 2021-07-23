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
    model: str, document_file: str, guidance_file: str, output_file: str, device: int, batch_size: int
) -> str:
    train_command = (
        f"python summarize.py"
        f"  {document_file}"
        f"  {guidance_file}"
        f"  {output_file}"
        f"  ../{model}"
        f"  model.pt"
        f"  ../{model}"
        f"  {batch_size}"
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


@Model.register("dou2021-oracle-sentence-gsum")
class OracleSentenceGSumModel(Model):
    def __init__(self, image: str = "dou2021", device: int = 0, batch_size: int = 16) -> None:
        # The sentence-guided model is the only one available
        self.model = "bart_sentence"
        self.image = image
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def _get_sentence_tokenize_command(input_file: str, output_file: str) -> str:
        return f"python sentence_tokenize.py {input_file} {output_file}"

    @staticmethod
    def _get_sentence_oracle_command(
        document_file: str, reference_file: str, output_file: str
    ) -> str:
        commands = [
            "cd guided_summarization/scripts",
            f"python sents.py {document_file} {reference_file} {output_file}",
            "cd ../..",
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

            commands = []

            # Write the input documents and references. For extracting the guidance signal,
            # the input documents and references are expected to be sentence-split by "<q>",
            # but the original document is used for inference (the reference is not used).
            # Therefore we write two versions of the input documents. If either the documents
            # or references aren't already sentence-split, we have to do that as well
            host_document_file = f"{host_input_dir}/input.source"
            host_document_tok_file = f"{host_input_dir}/input.tokenized.source"
            container_document_file = f"{container_input_dir}/input.source"
            write_to_text_file(documents, host_document_file)

            if any(isinstance(document, str) for document in documents):
                # Sentence splitting needs to be run
                if any(isinstance(document, list) for document in documents):
                    logger.warning(
                        "`documents` contains both sentence-split and un-sentence-split documents. "
                        "The sentence-split boundaries will be ignored and sentence splitting will "
                        "be run again."
                    )

                container_document_tok_file = (
                    f"{container_output_dir}/input.tokenized.source"
                )
                commands.append(
                    self._get_sentence_tokenize_command(
                        container_document_file, container_document_tok_file
                    )
                )
            else:
                container_document_tok_file = (
                    f"{container_input_dir}/input.tokenized.source"
                )
                write_to_text_file(documents, host_document_tok_file, separator="<q>")

            if any(isinstance(reference, str) for reference in references):
                # Sentence splitting needs to be run
                if any(isinstance(reference, list) for reference in references):
                    logger.warning(
                        "`references` contains both sentence-split and un-sentence-split references. "
                        "The sentence-split boundaries will be ignored and sentence splitting will "
                        "be run again."
                    )

                host_reference_file = f"{host_input_dir}/input.ref"
                container_reference_file = f"{container_input_dir}/input.ref"
                write_to_text_file(references, host_reference_file)

                container_reference_tok_file = (
                    f"{container_output_dir}/input.tokenized.ref"
                )
                commands.append(
                    self._get_sentence_tokenize_command(
                        container_reference_file, container_reference_tok_file
                    )
                )
            else:
                host_reference_tok_file = f"{host_input_dir}/input.tokenized.ref"
                container_reference_tok_file = (
                    f"{container_input_dir}/input.tokenized.ref"
                )
                write_to_text_file(references, host_reference_tok_file, separator="<q>")

            # Get the command to run extracting the oracle guidance. This uses
            # the sentence tokenized documents.
            container_guidance_tok_file = f"{container_output_dir}/input.tokenized.z"
            commands.append(
                self._get_sentence_oracle_command(
                    container_document_tok_file,
                    container_reference_tok_file,
                    container_guidance_tok_file,
                )
            )

            # The guidance file is sentence tokenized, but inference requires
            # the untokenized version. This command removes "<q>" from the text
            container_guidance_file = f"{container_output_dir}/input.z"
            commands.append(
                self._get_de_sentence_tokenize_command(
                    container_guidance_tok_file, container_guidance_file
                )
            )

            # Run inference with the original documents and de-tokenized guidance
            host_output_file = f"{host_output_dir}/input.output"
            container_output_file = f"{container_output_dir}/input.output"
            commands.append(
                _get_prediction_command(
                    self.model,
                    container_document_file,
                    container_guidance_file,
                    container_output_file,
                    self.device,
                    self.batch_size
                )
            )

            command = " && ".join(commands)
            cuda = self.device != -1
            os.makedirs(host_output_dir)
            run_command(self.image, command, volume_map=volume_map, cuda=cuda)

            summaries = open(host_output_file, "r").read().splitlines()
            return summaries
