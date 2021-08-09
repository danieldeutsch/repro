import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.scialom2021 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-questeval")
class QuestEval(Model):
    def __init__(self, image: str = DEFAULT_IMAGE, device: int = 0):
        self.image = image
        self.device = device

    @staticmethod
    def _check_single_text(texts_list: List[List[TextType]]) -> List[TextType]:
        single_texts = []
        for texts in texts_list:
            if texts is None:
                single_texts.append(None)
            else:
                if len(texts) != 1:
                    raise Exception(
                        f"QuestEval only supports single sources and references. Found: {len(texts)}"
                    )
                single_texts.append(texts[0])
        return single_texts

    def predict(
        self,
        candidate: TextType,
        sources: List[TextType] = None,
        references: List[TextType] = None,
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources, "references": references}],
            **kwargs,
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating QuestEval for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] if "sources" in inp else None for inp in inputs]
        references_list = [
            inp["references"] if "references" in inp else None for inp in inputs
        ]

        # QuestEval only supports single sources and references
        sources = self._check_single_text(sources_list)
        references = self._check_single_text(references_list)

        # Ensure all are strings or None
        candidates = [util.flatten(candidate) for candidate in candidates]
        sources = [
            util.flatten(source) if source is not None else None for source in sources
        ]
        references = [
            util.flatten(reference) if reference is not None else None
            for reference in references
        ]

        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.jsonl"
            container_input_file = f"{backend.container_dir}/input.jsonl"
            with open(host_input_file, "w") as out:
                for candidate, source, reference in zip(
                    candidates, sources, references
                ):
                    out.write(
                        json.dumps(
                            {
                                "candidate": candidate,
                                "source": source,
                                "reference": reference,
                            }
                        )
                        + "\n"
                    )

            host_output_file = f"{backend.host_dir}/output.jsonl"
            container_output_file = f"{backend.container_dir}/output.jsonl"

            kwargs_str = json.dumps(kwargs)
            if "'" in kwargs_str:
                raise Exception(
                    "Character `'` is currently not supported in values of `kwargs`"
                )

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")
                score_device = 0
            else:
                score_device = -1

            commands.append(
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --kwargs '{kwargs_str}'"
                f"  --cuda-device {score_device}"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(command=command, cuda=cuda, network_disabled=True)
            results = read_jsonl_file(host_output_file)
            micro_metrics = [{"questeval": metrics["scores"]} for metrics in results]
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics


@Model.register(f"{MODEL_NAME}-questeval-summarization")
class QuestEvalForSummarization(QuestEval):
    @staticmethod
    def _check_and_update_kwargs(kwargs: Dict):
        if "task" in kwargs:
            if kwargs["task"] != "summarization":
                raise Exception(f'kwarg `task` must be equal to "summarization"')
        else:
            kwargs["task"] = "summarization"

        if "do_weighter" in kwargs:
            if kwargs["do_weighter"] != True:
                raise Exception(f"kwarg `do_weighter` must be equal to `True`")
        else:
            kwargs["do_weighter"] = True

    def predict(self, *args, **kwargs) -> MetricsType:
        self._check_and_update_kwargs(kwargs)
        return super().predict(*args, **kwargs)

    def predict_batch(self, *args, **kwargs) -> Tuple[MetricsType, List[MetricsType]]:
        self._check_and_update_kwargs(kwargs)
        return super().predict_batch(*args, **kwargs)


@Model.register(f"{MODEL_NAME}-questeval-simplification")
class QuestEvalForSimplification(QuestEval):
    @staticmethod
    def _check_and_update_kwargs(kwargs: Dict):
        if "task" in kwargs:
            if kwargs["task"] != "text_simplification":
                raise Exception(f'kwarg `task` must be equal to "text_simplification"')
        else:
            kwargs["task"] = "text_simplification"

        if "do_BERTScore" in kwargs:
            if kwargs["do_BERTScore"] != True:
                raise Exception(f"kwarg `do_BERTScore` must be equal to `True`")
        else:
            kwargs["do_BERTScore"] = True

    def predict(self, *args, **kwargs) -> MetricsType:
        self._check_and_update_kwargs(kwargs)
        return super().predict(*args, **kwargs)

    def predict_batch(self, *args, **kwargs) -> Tuple[MetricsType, List[MetricsType]]:
        self._check_and_update_kwargs(kwargs)
        return super().predict_batch(*args, **kwargs)
