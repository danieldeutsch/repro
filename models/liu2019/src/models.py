import logging
import os
from glob import glob
from typing import Dict, List

from overrides import overrides

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import write_to_text_file
from repro.models import Model, SingleDocumentSummarizationModel
from repro.models.model import DocumentType, SummaryType

logger = logging.getLogger(__name__)


class _Liu2019Model(SingleDocumentSummarizationModel):
    """
    A wrapper around the TransformerAbs, BertSumExt, and BertSumExtAbs models proposed
    in `Liu & Lapata (2019) <https://arxiv.org/abs/1908.08345>`.
    """

    def __init__(
        self,
        model: str,
        task: str,
        image: str = "liu2019",
        device: int = 0,
        min_length: int = None,
        max_length: int = None,
        alpha: float = None,
    ) -> None:
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model
        task : str
            The task, either "ext" for an extractive model or "abs" for an abstractive model
        image : str, default="liu2019"
            The name of the Docker image
        device : int, default=0
            The ID of the GPU to use, -1 if CPU
        min_length : int, default=None
            The minimum length of the summary. If `None`, defaults to the original code
        max_length : int, default=None
            The maximum length of the summary. If `None`, defaults to the original code
        alpha : float, default=None
            The alpha length penalty parameter. If `None`, defaults to the original code
        """
        self.model = model
        self.task = task
        self.image = image
        self.device = device
        self.min_length = min_length
        self.max_length = max_length
        self.alpha = alpha

    @staticmethod
    def _load_results(output_dir: str) -> List[SummaryType]:
        # The summaries are written to the output_dir in a file with the
        # extension ".candidate". Here, we find that file and load it
        file_paths = glob(f"{output_dir}/*.candidate")
        if len(file_paths) != 1:
            raise Exception(f"Found {len(file_paths)} output files. Expected 1.")
        return open(file_paths[0], "r").read().splitlines()

    @staticmethod
    def _sentence_split_summaries(summaries: List[str]) -> List[List[str]]:
        return [summary.split("<q>") for summary in summaries]

    @overrides
    def predict_batch(
        self, inputs: List[Dict[str, DocumentType]], *args, **kwargs
    ) -> List[SummaryType]:
        documents = [inp["document"] for inp in inputs]
        logger.info(
            f"Predicting summaries for {len(documents)} documents "
            f"with pretrained model {self.model}, task {self.task} "
            f"and Docker image {self.image}."
        )

        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_input_file = f"{host_input_dir}/documents.txt"
            container_input_file = f"{container_input_dir}/documents.txt"
            write_to_text_file(documents, host_input_file)

            # Run inference. The output_dir must exist before running the docker command
            os.makedirs(host_output_dir)
            container_tokenized_file = f"{container_output_dir}/tokenized.txt"
            container_output_prefix = f"{container_output_dir}/out"

            commands = []
            commands.append(
                f"python preprocess.py"
                f"  --input-file {container_input_file}"
                f"  --output-file {container_tokenized_file}"
            )
            commands.append("cd PreSumm/src")

            # The train.py script sets the CUDA_VISIBLE_DEVICES environment variable,
            # so we do not do it here
            train_command = (
                f"python train.py"
                f"  -task {self.task}"
                f"  -mode test_text"
                f"  -test_from ../../{self.model}"
                f"  -text_src {container_tokenized_file}"
                f"  -result_path {container_output_prefix}"
                f"  -visible_gpus {self.device}"
            )
            if self.max_length is not None:
                train_command += f" -max_length {self.max_length}"
            if self.min_length is not None:
                train_command += f" -min_length {self.min_length}"
            if self.alpha is not None:
                train_command += f" -alpha {self.alpha}"
            commands.append(train_command)

            command = " && ".join(commands)

            cuda = self.device != -1
            run_command(
                self.image,
                command,
                volume_map=volume_map,
                network_disabled=True,
                cuda=cuda,
            )

            # Load the output summaries
            summaries = self._load_results(host_output_dir)

            # The <q> token splits sentences in each of the summaries.
            # Split them into sentences
            summaries = self._sentence_split_summaries(summaries)
            return summaries


@Model.register("liu2019-bertsumext")
class BertSumExt(_Liu2019Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(model="bertsumext_cnndm.pt", task="ext", **kwargs)


@Model.register("liu2019-bertsumextabs")
class BertSumExtAbs(_Liu2019Model):
    def __init__(self, model: str = "bertsumextabs_cnndm.pt", **kwargs) -> None:
        # Default values are taken from the Readme in the original repo
        if model == "bertsumextabs_cnndm.pt":
            min_length = kwargs.pop("min_length", 50)
            max_length = kwargs.pop("max_length", 200)
            alpha = kwargs.pop("alpha", 0.95)
        elif model == "bertsumextabs_xsum.pt":
            min_length = kwargs.pop("min_length", 20)
            max_length = kwargs.pop("max_length", 100)
            alpha = kwargs.pop("alpha", 0.9)
        else:
            raise Exception(f"Unknown pretrained model: {model}")

        super().__init__(
            model=model,
            task="abs",
            min_length=min_length,
            max_length=max_length,
            alpha=alpha,
            **kwargs,
        )


@Model.register("liu2019-transformerabs")
class TransformerAbs(_Liu2019Model):
    def __init__(self, **kwargs) -> None:
        # Default values are taken from the Readme in the original repo
        min_length = kwargs.pop("min_length", 50)
        max_length = kwargs.pop("max_length", 200)
        alpha = kwargs.pop("alpha", 0.95)

        super().__init__(
            model="transformerabs_cnndm.pt",
            task="abs",
            min_length=min_length,
            max_length=max_length,
            alpha=alpha,
            **kwargs,
        )
