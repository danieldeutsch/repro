import json
import logging
from typing import Dict, List, Tuple, Union

from repro.common import util
from repro.common.docker import DockerContainer
from repro.common.io import read_jsonl_file
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.thompson2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


class _Prism(Model):
    def __init__(
        self, image: str = DEFAULT_IMAGE, device: int = 0, language: str = "en"
    ):
        self.image = image
        self.device = device
        self.language = language

    @staticmethod
    def _check_single_text(texts_list: List[List[TextType]]) -> List[TextType]:
        single_texts = []
        for texts in texts_list:
            if texts is None:
                single_texts.append(None)
            else:
                if len(texts) != 1:
                    raise Exception(
                        f"Prism only supports single sources and references. Found: {len(texts)}"
                    )
                single_texts.append(texts[0])
        return single_texts

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating Prism for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] if "sources" in inp else None for inp in inputs]
        references_list = [
            inp["references"] if "references" in inp else None for inp in inputs
        ]

        # Prism only supports single sources and references
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

        # Prism only supports having references xor sources. Ensure this is true
        has_references = all(reference is not None for reference in references)
        has_sources = all(source is not None for source in sources)
        if has_references and has_sources or (not has_references and not has_sources):
            raise Exception(
                f"Prism supports having either input references xor sources, not both or neither."
            )

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

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            commands.append("cd prism")
            commands.append(
                f"python score.py"
                f"  --input-file {container_input_file}"
                f"  --language {self.language}"
                f"  --output-file {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(command=command, cuda=cuda, network_disabled=True)
            micro_metrics = read_jsonl_file(host_output_file)
            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics

    def translate(self, language: str, source: str, batch_size: int = None) -> str:
        return self.translate_batch(
            language, [{"source": source}], batch_size=batch_size
        )[0]

    def translate_batch(
        self, language: str, inputs: List[Dict[str, str]], batch_size: int = None
    ) -> List[str]:
        logger.info(f"Translating {len(inputs)} inputs into {language}")

        batch_size = batch_size or 32

        sources = [inp["source"] for inp in inputs]
        with DockerContainer(self.image) as backend:
            host_input_file = f"{backend.host_dir}/input.src"
            container_input_file = f"{backend.container_dir}/input.src"
            with open(host_input_file, "w") as out:
                for source in sources:
                    if "\n" in source:
                        raise Exception(f"Input sources must not contain '\\n'")
                    out.write(source + "\n")

            host_output_file = f"{backend.host_dir}/output.tgt"
            container_output_file = f"{backend.container_dir}/output.tgt"

            cuda = self.device != -1
            commands = []
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            commands.append(
                f"sh translate.sh {container_input_file} {language} {batch_size} {container_output_file}"
            )

            command = " && ".join(commands)
            backend.run_command(command=command, cuda=cuda, network_disabled=True)

            outputs = open(host_output_file, "r").read().splitlines()
            return outputs


@Model.register(f"{MODEL_NAME}-prism")
class Prism(_Prism):
    def predict(
        self,
        candidate: TextType,
        references: List[TextType] = None,
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "references": references}],
            **kwargs,
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating Prism for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        # Create an input for each reference
        indices = []
        unrolled_inputs = []
        for i, (candidate, references) in enumerate(zip(candidates, references_list)):
            for reference in references:
                indices.append(i)
                unrolled_inputs.append(
                    {"candidate": candidate, "references": [reference]}
                )

        # Score using the base class
        _, unrolled_micro = super().predict_batch(unrolled_inputs)

        # Re-aggregate based on reference group
        micro = util.aggregate_metrics_by_group(indices, unrolled_micro)
        macro = util.average_dicts(micro)

        return macro, micro


@Model.register(f"{MODEL_NAME}-prism-src")
class PrismSrc(_Prism):
    def predict(
        self,
        candidate: TextType,
        sources: List[TextType] = None,
        **kwargs,
    ) -> MetricsType:
        return self.predict_batch(
            [{"candidate": candidate, "sources": sources}],
            **kwargs,
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[TextType, List[TextType]]]], **kwargs
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Calculating Prism for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        sources_list = [inp["sources"] for inp in inputs]

        # Create an input for each reference
        indices = []
        unrolled_inputs = []
        for i, (candidate, sources) in enumerate(zip(candidates, sources_list)):
            for source in sources:
                indices.append(i)
                unrolled_inputs.append({"candidate": candidate, "sources": [source]})

        # Score using the base class
        _, unrolled_micro = super().predict_batch(unrolled_inputs)

        # Re-aggregate based on source group
        micro = util.aggregate_metrics_by_group(indices, unrolled_micro)
        macro = util.average_dicts(micro)

        return macro, micro
