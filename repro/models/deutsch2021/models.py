import json
import logging
from typing import Any, Dict, List, Tuple, Union

from overrides import overrides

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model, QuestionAnsweringModel, QuestionGenerationModel
from repro.models.deutsch2021 import DEFAULT_IMAGE, MODEL_NAME

QAPairsType = List[List[Dict[str, Any]]]

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-qaeval")
class QAEval(Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        generation_batch_size: int = 8,
        answering_batch_size: int = 8,
        lerc_batch_size: int = 8,
    ):
        self.image = image
        self.device = device
        self.generation_batch_size = generation_batch_size
        self.answering_batch_size = answering_batch_size
        self.lerc_batch_size = lerc_batch_size

    def predict(
        self, candidate: TextType, references: List[TextType], **kwargs
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}], **kwargs
        )[0]

    def predict_batch(
        self,
        inputs: List[Dict[str, Union[str, List[str]]]],
        return_qa_pairs=False,
        **kwargs,
    ) -> Union[
        Tuple[MetricsType, List[MetricsType]],
        Tuple[MetricsType, List[MetricsType], List[QAPairsType]],
    ]:
        logger.info(f"Calculating QAEval for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

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

            commands = []
            cuda = self.device != -1
            if cuda:
                predict_device = 0
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            else:
                predict_device = -1

            kwargs = {
                "cuda_device": predict_device,
                "generation_batch_size": self.generation_batch_size,
                "answering_batch_size": self.answering_batch_size,
                "use_lerc": True,
                "lerc_batch_size": self.lerc_batch_size,
            }
            kwargs_str = json.dumps(kwargs)
            commands.append(
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --kwargs '{kwargs_str}'"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            results = read_jsonl_file(host_output_file)
            micro_metrics = [result["metrics"] for result in results]
            macro_metrics = util.average_dicts(micro_metrics)

            if return_qa_pairs:
                qa_pairs = [result["qa_pairs"] for result in results]
                return macro_metrics, micro_metrics, qa_pairs
            else:
                return macro_metrics, micro_metrics


@Model.register(f"{MODEL_NAME}-question-generation")
class QAEvalQuestionGenerationModel(QuestionGenerationModel):
    def __init__(
        self, image: str = DEFAULT_IMAGE, device: int = 0, batch_size: int = 8
    ) -> None:
        self.image = image
        self.device = device
        self.batch_size = batch_size

    @overrides
    def predict_batch(self, inputs: List[Dict[str, str]], **kwargs) -> List[str]:
        logger.info(f"Generating questions for {len(inputs)} inputs")

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"
            with open(host_input_file, "w") as out:
                for inp in inputs:
                    out.write(
                        json.dumps(
                            {
                                "context": inp["context"],
                                "start": inp["start"],
                                "end": inp["end"],
                            }
                        )
                        + "\n"
                    )

            cuda = self.device != -1
            predict_device = 0 if cuda else -1
            command = (
                f"python generate_questions.py"
                f"  --input-file {container_input_file}"
                f"  --model-file models/generation/model.tar.gz"
                f"  --cuda-device {predict_device}"
                f"  --batch-size {self.batch_size}"
                f"  --output-file {container_output_file}"
            )
            if cuda:
                command = f"CUDA_VISIBLE_DEVICES={self.device} " + command

            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            questions = [
                output["question"] for output in read_jsonl_file(host_output_file)
            ]
            return questions


@Model.register(f"{MODEL_NAME}-question-answering")
class QAEvalQuestionAnsweringModel(QuestionAnsweringModel):
    def __init__(
        self, image: str = DEFAULT_IMAGE, device: int = 0, batch_size: int = 8
    ) -> None:
        self.image = image
        self.device = device
        self.batch_size = batch_size

    @overrides
    def predict_batch(
        self, inputs: List[Dict[str, str]], return_dicts: bool = False, **kwargs
    ) -> List[str]:
        logger.info(f"Answering for {len(inputs)} inputs")

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"
            with open(host_input_file, "w") as out:
                for inp in inputs:
                    out.write(
                        json.dumps(
                            {
                                "context": inp["context"],
                                "question": inp["question"],
                            }
                        )
                        + "\n"
                    )

            cuda = self.device != -1
            predict_device = 0 if cuda else -1
            command = (
                f"python answer_questions.py"
                f"  --input-file {container_input_file}"
                f"  --model-dir models/answering"
                f"  --cuda-device {predict_device}"
                f"  --batch-size {self.batch_size}"
                f"  --output-file {container_output_file}"
            )
            if cuda:
                command = f"CUDA_VISIBLE_DEVICES={self.device} " + command

            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            outputs = read_jsonl_file(host_output_file)
            if not return_dicts:
                outputs = [
                    output["prediction"]
                    if output["probability"] > output["null_probability"]
                    else None
                    for output in outputs
                ]
            return outputs
