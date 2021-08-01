import json
import logging
import os
from typing import Dict, List, Union

from overrides import overrides

from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import SummaryType
from repro.models import Model, QuestionAnsweringModel, QuestionGenerationModel

logger = logging.getLogger(__name__)


@Model.register("deutsch2021-qaeval")
class QAEval(Model):
    def __init__(
        self,
        image: str = "deutsch2021",
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
        self, summary: SummaryType, references: List[SummaryType], **kwargs
    ) -> Dict[str, float]:
        return self.predict_batch(
            [{"summary": summary, "references": references}], **kwargs
        )

    def predict_batch(
        self, inputs: List[Dict[str, Union[str, List[str]]]], **kwargs
    ) -> Dict[str, float]:
        logger.info(f"Calculating QAEval for {len(inputs)} inputs")

        summaries = [inp["summary"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            with open(host_input_file, "w") as out:
                for i, (summary, references) in enumerate(
                    zip(summaries, references_list)
                ):
                    summary = {"text": summary}
                    references = [{"text": reference} for reference in references]
                    out.write(
                        json.dumps(
                            {
                                "instance_id": str(i),
                                "summarizer_id": "repro",
                                "summarizer_type": "peer",
                                "summary": summary,
                                "references": references,
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/macro.json"
            container_output_file = f"{backend.container_dir}/macro.json"

            commands = []
            cuda = self.device != -1
            if cuda:
                predict_device = 0
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
            else:
                predict_device = -1

            commands.append(
                f"sacrerouge qa-eval evaluate"
                f"  --input-files {container_input_file}"
                f"  --dataset-reader reference-based"
                f"  --use_lerc true"
                f"  --generation_batch_size {self.generation_batch_size}"
                f"  --answering_batch_size {self.answering_batch_size}"
                f"  --lerc_batch_size {self.lerc_batch_size}"
                f"  --cuda_device {predict_device}"
                f"  --macro-output-json {container_output_file}"
                f"  --micro-output-jsonl {backend.container_dir}/micro.jsonl"
            )

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=True,
            )

            scores = json.load(open(host_output_file, "r"))
            return scores["metrics"]


@Model.register("deutsch2021-question-generation")
class QAEvalQuestionGenerationModel(QuestionGenerationModel):
    def __init__(
        self, image: str = "deutsch2021", device: int = 0, batch_size: int = 8
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
                f"  --model-file /root/.sacrerouge/metrics/qaeval/models/generation/model.tar.gz"
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


@Model.register("deutsch2021-question-answering")
class QAEvalQuestionAnsweringModel(QuestionAnsweringModel):
    def __init__(
        self, image: str = "deutsch2021", device: int = 0, batch_size: int = 8
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
                f"  --model-dir /root/.sacrerouge/metrics/qaeval/models/answering/model"
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
