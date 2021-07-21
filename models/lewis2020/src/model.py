import logging
import os
from typing import Dict, List

from overrides import overrides

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import write_to_text_file
from repro.models import Model, SingleDocumentSummarizationModel
from repro.models.model import DocumentType, SummaryType

logger = logging.getLogger(__name__)


@Model.register("lewis2020-bart")
class BART(SingleDocumentSummarizationModel):
    def __init__(
        self,
        pretrained_model: str = "bart.large.cnn",
        batch_size: int = None,
        image: str = "lewis2020",
        device: int = 0,
    ) -> None:
        if pretrained_model not in ["bart.large.cnn", "bart.large.xsum"]:
            raise Exception(f"Unknown pretrained model: {pretrained_model}")
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.image = image
        self.device = device

    @overrides
    def predict_batch(
        self, inputs: List[Dict[str, DocumentType]], *args, **kwargs
    ) -> List[SummaryType]:
        documents = [inp["document"] for inp in inputs]
        logger.info(
            f"Predicting summaries for {len(documents)} documents with Docker image {self.image}"
        )

        with TemporaryDirectory() as temp:
            input_dir = f"{temp}/input"
            output_dir = f"{temp}/output"
            volume_map = make_volume_map(input_dir, output_dir)

            host_input_file = f"{input_dir}/documents.txt"
            container_input_file = f"{volume_map[input_dir]}/documents.txt"
            write_to_text_file(documents, host_input_file)

            # Run inference. The output_dir must exist before running
            # the docker command
            os.makedirs(output_dir)
            host_output_file = f"{output_dir}/summaries.txt"
            container_output_file = f"{volume_map[output_dir]}/summaries.txt"
            command = (
                f"cd fairseq && "
                f"CUDA_VISIBLE_DEVICES={self.device} "
                f"python examples/bart/summarize.py"
                f"  --model-dir ../{self.pretrained_model}"
                f"  --model-file model.pt"
                f"  --src {container_input_file}"
                f"  --out {container_output_file}"
            )

            if self.batch_size is not None:
                command += f" --bsz {self.batch_size}"

            if self.pretrained_model == "bart.large.xsum":
                command += " --xsum-kwargs"

            cuda = self.device != -1
            run_command(self.image, command, volume_map=volume_map, cuda=cuda)

            # Load the output summaries
            summaries = open(host_output_file, "r").read().splitlines()
            return summaries
