import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.kryscinski2019 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


class _Kryscinski2019Model(Model):
    def __init__(
        self,
        is_factccx: bool,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        batch_size: int = 8,
    ):
        self.is_factccx = is_factccx
        if is_factccx:
            self.name = "FactCCX"
            self.model = "factccx-checkpoint"
            self.model_type = "pbert"
        else:
            self.name = "FactCC"
            self.model = "factcc-checkpoint"
            self.model_type = "bert"
        self.image = image
        self.device = device
        self.batch_size = batch_size

    def predict(
        self, candidate: TextType, sources: List[TextType], **kwargs
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources}], **kwargs
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[TextType, List[TextType]]]],
        **kwargs,
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating {self.name} for {len(inputs)} inputs with model")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] for inp in inputs]

        # FactCC only accepts single source documents
        sources = util.check_for_single_texts(sources_list)

        # Make sure all are type `str`
        candidates = [util.flatten(candidate) for candidate in candidates]
        sources = [util.flatten(source) for source in sources]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/data-dev.jsonl"
            with open(host_input_file, "w") as out:
                for i, (candidate, source) in enumerate(zip(candidates, sources)):
                    out.write(
                        json.dumps(
                            {
                                "id": str(i),
                                "claim": candidate,
                                "text": source,
                                "label": "CORRECT",  # dummy label
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            commands.append("cd factCC/modeling")

            # The run.py code uses wandb for logging, but the code will fail
            # if we run it without first disabling wandb
            commands.append("wandb disabled")

            score_command = (
                f"python run_test.py"
                f"  --task_name factcc_annotated"
                f"  --do_test"
                f"  --eval_all_checkpoints"
                f"  --do_lower_case"
                f"  --overwrite_cache"
                f"  --max_seq_length 512"
                f"  --per_gpu_eval_batch_size {self.batch_size}"
                f"  --model_type {self.model_type}"
                f"  --model_name_or_path bert-base-uncased"
                f"  --data_dir {backend.container_dir}"
                f"  --output_dir ../../{self.model}"
                f"  --output-file {container_output_file}"
            )
            if not cuda:
                score_command += " --no_cuda"
            commands.append(score_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            micro_metrics = read_jsonl_file(host_output_file)
            micro_metrics = [{self.name.lower(): metrics} for metrics in micro_metrics]
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics


@Model.register(f"{MODEL_NAME}-factcc")
class FactCC(_Kryscinski2019Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(is_factccx=False, **kwargs)


@Model.register(f"{MODEL_NAME}-factccx")
class FactCCX(_Kryscinski2019Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(is_factccx=True, **kwargs)
