import json
import logging
from typing import Any, Dict, List, Tuple

from overrides import overrides

from repro.common import util
from repro.common.docker import DockerContainer
from repro.data.types import MetricsType, TextType
from repro.models import Model
from repro.models.colombo2021 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


class _Colombo2021Model(Model):
    def __init__(self, name: str, image: str, device: int, model: str, batch_size: int):
        self.name = name
        self.image = image
        self.device = device
        self.model = model
        self.batch_size = batch_size

    def _get_predict_options(self) -> str:
        """Returns the command line options for the metric"""
        raise NotImplementedError

    def predict(self, candidate: TextType, references: List[TextType]) -> MetricsType:
        return self.predict_batch([{"candidate": candidate, "references": references}])[
            0
        ]

    @overrides
    def predict_batch(
        self, inputs: List[Dict[str, Any]]
    ) -> Tuple[MetricsType, List[MetricsType]]:
        logger.info(f"Predicting scores for {len(inputs)} inputs")

        candidates = [inp["candidate"] for inp in inputs]
        references_list = [inp["references"] for inp in inputs]

        with DockerContainer(self.image) as backend:
            host_ref_file = f"{backend.host_dir}/ref.txt"
            container_ref_file = f"{backend.container_dir}/ref.txt"

            host_cand_file = f"{backend.host_dir}/cand.txt"
            container_cand_file = f"{backend.container_dir}/cand.txt"

            with open(host_ref_file, "w") as out_ref:
                with open(host_cand_file, "w") as out_cand:
                    for candidate, references in zip(candidates, references_list):
                        # Write the candidate once per reference
                        for reference in references:
                            out_ref.write(reference + "\n")
                            out_cand.write(candidate + "\n")

            host_out_file = f"{backend.host_dir}/out.json"
            container_out_file = f"{backend.container_dir}/out.json"

            commands = []
            cuda = self.device != -1
            if cuda:
                commands.append(f"export CUDA_VISIBLE_DEVICES={self.device}")

            commands.append("cd nlg_eval_via_simi_measures")

            predict_command = (
                f"python score_cli.py"
                f"  --ref {container_ref_file}"
                f"  --cand {container_cand_file}"
                f"  --metric_name {self.name}"
                f"  --output_file {container_out_file}"
            )
            if self.model is not None:
                predict_command += f" --model {self.model}"
            if self.batch_size is not None:
                predict_command += f" --batch_size {self.batch_size}"

            predict_command += " " + self._get_predict_options()
            commands.append(predict_command)

            command = " && ".join(commands)
            backend.run_command(
                command=command,
                cuda=cuda,
                network_disabled=False,
            )

            with open(host_out_file, "r") as f:
                unrolled_micro_metrics = json.load(f)

            # Average over references
            micro_metrics = []
            index = 0
            for references in references_list:
                metrics = []
                for _ in references:
                    metrics.append(unrolled_micro_metrics[index])
                    index += 1
                micro_metrics.append(util.average_dicts(metrics))

            macro_metrics = util.average_dicts(micro_metrics)
            return macro_metrics, micro_metrics


@Model.register(f"{MODEL_NAME}-infolm")
class InfoLM(_Colombo2021Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        model: str = None,
        batch_size: int = 8,
        idf: bool = False,
        measure: str = None,
        temperature: float = None,
        beta: float = None,
        alpha: float = None,
    ) -> None:
        super().__init__("infolm", image, device, model, batch_size)
        self.idf = idf
        self.measure = measure
        self.temperature = temperature
        self.beta = beta
        self.alpha = alpha

    @overrides
    def _get_predict_options(self) -> str:
        options = []
        if self.idf:
            options.append("--idf")
        if self.measure is not None:
            options.append(f"--measure_to_use {self.measure}")
        if self.temperature is not None:
            options.append(f"--temperature {self.temperature}")
        if self.beta is not None:
            options.append(f"--beta {self.beta}")
        if self.alpha is not None:
            options.append(f"--alpha {self.alpha}")
        return " ".join(options)


@Model.register(f"{MODEL_NAME}-baryscore")
class BaryScore(_Colombo2021Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        model: str = None,
        batch_size: int = 8,
        last_layers: int = None,
        sinkhorn_ref: float = None,
        idf: bool = False,
    ) -> None:
        super().__init__("baryscore", image, device, model, batch_size)
        self.last_layers = last_layers
        self.sinkhorn_ref = sinkhorn_ref
        self.idf = idf

    @overrides
    def _get_predict_options(self) -> str:
        options = []
        if self.last_layers is not None:
            options.append(f"--last_layers {self.last_layers}")
        if self.sinkhorn_ref is not None:
            options.append(f"--sinkhorn_ref {self.sinkhorn_ref}")
        if self.idf:
            options.append("--idf")
        return " ".join(options)


@Model.register(f"{MODEL_NAME}-depthscore")
class DepthScore(_Colombo2021Model):
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        device: int = 0,
        model: str = None,
        batch_size: int = 8,
        layers_to_consider: int = None,
        considered_measure: str = None,
        p: float = None,
        eps: float = None,
        n_alpha: float = None,
    ) -> None:
        super().__init__("depthscore", image, device, model, batch_size)
        self.layers_to_consider = layers_to_consider
        self.considered_measure = considered_measure
        self.p = p
        self.eps = eps
        self.n_alpha = n_alpha

    @overrides
    def _get_predict_options(self) -> str:
        options = []
        if self.layers_to_consider is not None:
            options.append(f"--layers_to_consider {self.layers_to_consider}")
        if self.considered_measure is not None:
            options.append(f"--considered_measure {self.considered_measure}")
        if self.p is not None:
            options.append(f"--p {self.p}")
        if self.eps is not None:
            options.append(f"--eps {self.eps}")
        if self.n_alpha is not None:
            options.append(f"--n_alpha {self.n_alpha}")
        return " ".join(options)
