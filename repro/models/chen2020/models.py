import json
import logging
import os
from typing import Any, Dict, List, Union

from overrides import overrides

from repro.common import TemporaryDirectory
from repro.common.docker import make_volume_map, run_command
from repro.common.io import read_jsonl_file
from repro.models import Model
from repro.models.chen2020 import DEFAULT_IMAGE, MODEL_NAME

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-lerc")
class LERC(Model):
    def __init__(
        self, image: str = DEFAULT_IMAGE, device: int = 0, batch_size: int = 8
    ) -> None:
        self.image = image
        self.device = device
        self.batch_size = batch_size
        self.model = "lerc-2020-11-18.tar.gz"

    @overrides
    def predict(
        self, context: str, question: str, reference: str, candidate: str, **kwargs
    ) -> float:
        return self.predict_batch(
            [
                {
                    "context": context,
                    "question": question,
                    "reference": reference,
                    "candidate": candidate,
                }
            ],
            **kwargs,
        )[0]

    @overrides
    def predict_batch(self, inputs: List[Dict[str, Any]], **kwargs) -> List[float]:
        logger.info(
            f"Predicting scores for {len(inputs)} inputs, image {self.image}, and model {self.model}"
        )

        with TemporaryDirectory() as temp:
            host_input_dir = f"{temp}/input"
            host_output_dir = f"{temp}/output"
            volume_map = make_volume_map(host_input_dir, host_output_dir)
            container_input_dir = volume_map[host_input_dir]
            container_output_dir = volume_map[host_output_dir]

            host_input_file = f"{host_input_dir}/input.jsonl"
            host_output_file = f"{host_output_dir}/output.jsonl"
            container_input_file = f"{container_input_dir}/input.jsonl"
            container_output_file = f"{container_output_dir}/output.jsonl"

            os.makedirs(host_input_dir)
            with open(host_input_file, "w") as out:
                for inp in inputs:
                    out.write(
                        json.dumps(
                            {
                                "context": inp["context"],
                                "question": inp["question"],
                                "reference": inp["reference"],
                                "candidate": inp["candidate"],
                            }
                        )
                        + "\n"
                    )

            cuda = self.device != -1
            predict_device = 0 if cuda else -1
            commands = ["cd MOCHA"]
            predict_command = (
                f"python predict.py"
                f"  --input-file {container_input_file}"
                f"  --model-path ../{self.model}"
                f"  --batch-size {self.batch_size}"
                f"  --cuda-device {predict_device}"
                f"  --output-file {container_output_file}"
            )
            if cuda:
                predict_command = (
                    f"CUDA_VISIBLE_DEVICES={self.device} " + predict_command
                )
            commands.append(predict_command)

            command = " && ".join(commands)
            os.makedirs(host_output_dir)
            run_command(
                self.image,
                command,
                volume_map=volume_map,
                cuda=cuda,
                network_disabled=False,
            )

            outputs = read_jsonl_file(host_output_file)
            scores = [output["pred_score"] for output in outputs]
            return scores


@Model.register(f"{MODEL_NAME}-eval")
class MOCHAEvaluationMetric(Model):
    def __init__(self, image: str = DEFAULT_IMAGE) -> None:
        self.image = image

    @overrides
    def predict(
        self,
        dataset: str,
        source: str,
        score: float,
        prediction: float,
        **kwargs,
    ) -> Dict[str, float]:
        # You can't compute this metric with just one example. We define
        # this method so the "predict" command can identify the required arguments.
        raise NotImplementedError

    @overrides
    def predict_batch(
        self, inputs: List[Dict[str, Union[str, float]]], **kwargs
    ) -> Dict[str, float]:
        logger.info(f"Evaluating {len(inputs)} inputs")
        with TemporaryDirectory() as temp:
            # The output file is hard-coded to be relative to the prediction file,
            # so we don't need an output directory
            host_input_dir = f"{temp}/input"
            volume_map = make_volume_map(host_input_dir)
            container_input_dir = volume_map[host_input_dir]

            host_annotations_file = f"{host_input_dir}/annotations.jsonl"
            host_predictions_file = f"{host_input_dir}/predictions.jsonl"
            host_output_file = f"{host_input_dir}/predictions.jsonl.corrs"
            container_annotations_file = f"{container_input_dir}/annotations.jsonl"
            container_predictions_file = f"{container_input_dir}/predictions.jsonl"

            annotations = {}
            predictions = {}
            for i, inp in enumerate(inputs):
                dataset = inp["dataset"]
                if dataset not in annotations:
                    annotations[dataset] = {}
                    predictions[dataset] = {}

                instance_id = str(i)
                annotations[dataset][instance_id] = {
                    "score": inp["score"],
                    "metadata": {"source": inp["source"]},
                }
                predictions[dataset][instance_id] = {
                    "pred_score": inp["prediction"],
                    "metadata": {"source": inp["source"]},
                }

            os.makedirs(host_input_dir)
            with open(host_annotations_file, "w") as out:
                out.write(json.dumps(annotations))
            with open(host_predictions_file, "w") as out:
                out.write(json.dumps(predictions))

            command = (
                f"cd MOCHA && "
                f"python evaluate_mocha_preds.py"
                f"  --annotations {container_annotations_file}"
                f"  --predictions {container_predictions_file}"
            )
            run_command(
                self.image, command, volume_map=volume_map, network_disabled=True
            )

            metrics = json.load(open(host_output_file, "r"))
            return metrics
