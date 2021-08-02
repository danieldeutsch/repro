import json
import logging
from typing import Dict, List, Optional, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import TextType
from repro.models import Model

logger = logging.getLogger(__name__)


@Model.register("zhao2019-moverscore")
class MoverScore(Model):
    def __init__(self, image: str = "zhao2019", device: int = 0):
        self.image = image
        self.device = device

    def predict(
        self,
        candidate: TextType,
        references: List[TextType],
        **kwargs,
    ) -> Dict[str, float]:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
        batch_size: Optional[int] = None,
        use_stopwords: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        logger.info(
            f"Calculating MoverScore with image {self.image} on {len(inputs)} inputs."
        )

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # Each candidate and reference must be `str`, not `List[str]`
        candidates = [util.flatten(candidate) for candidate in candidates]
        references_list = [
            [util.flatten(reference) for reference in references]
            for references in references_list
        ]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            with open(host_input_file, "w") as out:
                for candidate, references in zip(candidates, references_list):
                    out.write(
                        json.dumps(
                            {
                                "candidate": candidate,
                                "references": references,
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            score_command = (
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --use-stopwords {str(use_stopwords).lower()}"
                f"  --output-file {container_output_file}"
            )
            if batch_size is not None:
                score_command += f" --batch-size {batch_size}"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(command=command, cuda=cuda)

            micro = read_jsonl_file(host_output_file)
            macro = util.average_dicts(micro)
            return macro, micro


@Model.register("zhao2019-moverscore-summarization")
class MoverScoreForSummarization(MoverScore):
    def predict_batch(
        self, *args, **kwargs
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        if "use_stopwords" in kwargs:
            if kwargs["use_stopwords"] != True:
                raise Exception(
                    f"`use_stopwords` must be `True` for `MoverScoreForSummarization`"
                )
        else:
            kwargs["use_stopwords"] = True
        return super().predict_batch(*args, **kwargs)
