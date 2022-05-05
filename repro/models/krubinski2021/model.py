import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType
from repro.models import Model
from repro.models.krubinski2021 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-mteqa")
class MTEQA(Model):
    def __init__(self, image: str = DEFAULT_IMAGE, device: int = 0):
        self.image = image
        self.device = device

    def predict(self, candidate: str, references: List[str], **kwargs) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[str, List[str]]]],
        gen_from_out: bool = False,
        **kwargs,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        """

        Parameters
        ----------
        inputs
        gen_from_out
        kwargs

        Returns
        -------

        """
        logger.info(f"Calculating MTEQA for {len(inputs)} inputs with model")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        with DockerContainer(self.image) as backend:
            host_candidate_file = f"{backend.host_dir}/candidates.txt"
            container_candidate_file = f"{backend.container_dir}/candidates.txt"

            host_reference_file = f"{backend.host_dir}/references.txt"
            container_reference_file = f"{backend.container_dir}/references.txt"

            indices = []
            with open(host_candidate_file, "w") as out_cand:
                with open(host_reference_file, "w") as out_ref:
                    for i, (candidate, references) in enumerate(
                        zip(candidates, references_list)
                    ):
                        for reference in references:
                            out_cand.write(candidate + "\n")
                            out_ref.write(reference + "\n")
                            indices.append(i)

            host_output_file = f"{backend.host_dir}/output.tsv"
            container_output_file = f"{backend.container_dir}/output.tsv"

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            # Only English is supported
            commands.append("cd MTEQA")
            score_command = (
                f"python mteqa_score.py"
                f"  --reference {container_reference_file}"
                f"  --hypothesis {container_candidate_file}"
                f"  --lang en"
                f"  --verbose"
            )
            if not cuda:
                score_command += " --cpu"
            if gen_from_out:
                score_command += " --gen_from_out"
            score_command += f" > {container_output_file}"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            # Load the data in the output tsv
            with open(host_output_file, "r") as f:
                lines = f.read().splitlines()

            # Format validation
            header = lines[0].strip().split("\t")
            if (
                header[2] != "F1"
                or header[3] != "EM"
                or header[4] != "chrf"
                or header[5] != "bleu"
            ):
                raise Exception(f"Unexpected header format: {header}")
            if len(lines) - 1 != len(indices):
                raise Exception(
                    f"Incorrect number of output scores. Expected {len(indices)}, found {len(lines) - 1}"
                )

            # Group by input candidate index
            index_to_metrics = defaultdict(list)
            for index, line in zip(indices, lines[1:]):
                cols = line.strip().split("\t")
                index_to_metrics[index].append(
                    {
                        "mteqa": {
                            "f1": float(cols[2]),
                            "em": float(cols[3]),
                            "chrf": float(cols[4]),
                            "bleu": float(cols[5]),
                        }
                    }
                )

            # Average and return
            micro = []
            for i in range(len(candidates)):
                micro.append(util.average_dicts(index_to_metrics[i]))
            macro = util.average_dicts(micro)
            return macro, micro
