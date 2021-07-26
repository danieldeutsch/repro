import argparse
import inspect
import json
from overrides import overrides
from typing import Any, Dict, List, Union

from repro.commands.subcommand import RootSubcommand
from repro.commands.util import load_dataset_reader, load_model, load_output_writer
from repro.common.logging import prepare_global_logging
from repro.data.dataset_readers import HuggingfaceDatasetsDatasetReader
from repro.data.types import InstanceDict
from repro.models import Model


def predict_with_model(
    model: Model,
    instances: List[InstanceDict],
    kwargs: Union[str, Dict[str, Any]] = None,
) -> Any:
    # Deserialize kwargs if necessary
    kwargs = kwargs or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"`kwargs` is expected to be a dictionary or json-serialized dictionary: {kwargs}"
        )

    # Find the required arguments for the model's `predict` function
    parameters = inspect.signature(model.predict).parameters
    required_args = set()
    for key, value in parameters.items():
        if value.kind == inspect.Parameter.POSITIONAL_ONLY:
            required_args.add(key)

    # Ensure all of the instances have at least those arguments
    for i, instance in enumerate(instances):
        for arg in required_args:
            if arg not in instance:
                raise Exception(f"Instance {i} missing required argument {arg}")

    # Pass all of the instances, including with other, non-required arguments
    predictions = model.predict_batch(instances, **kwargs)
    return predictions


@RootSubcommand.register("predict")
class PredictSubcommand(RootSubcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = "Predict using a model"
        self.parser = parser.add_parser(
            "predict", description=description, help=description
        )
        self.parser.add_argument(
            "--model-name", required=True, help="The name of the model to predict with"
        )
        self.parser.add_argument(
            "--model-kwargs",
            required=False,
            help="A serialized json object which will be deserialized and passed as "
            "**kwargs to the model constructor",
        )
        self.parser.add_argument(
            "--predict-kwargs",
            required=False,
            help="A serialized json object which will be deserialized and passed as "
            "**kwargs to the `Model.predict_batch` function",
        )
        self.parser.add_argument(
            "--dataset-name",
            required=False,
            help="The name of the Huggingface `datasets` library dataset to predict on",
        )
        self.parser.add_argument(
            "--split",
            required=False,
            help="The split of the Huggingface `datasets` dataset to use",
        )
        self.parser.add_argument(
            "--input-files",
            required=False,
            nargs="+",
            help="The input file(s) to pass to the dataset reader",
        )
        self.parser.add_argument(
            "--dataset-reader",
            required=False,
            help="The name of the dataset reader to use",
        )
        self.parser.add_argument(
            "--dataset-reader-kwargs",
            required=False,
            help="A serialized json object which will be deserialized and passed as "
            "**kwargs to the dataset reader constructor",
        )
        self.parser.add_argument(
            "--output",
            required=True,
            help="The name of the output file or directory that the model predictions will be written to",
        )
        self.parser.add_argument(
            "--output-writer",
            required=False,
            default="default",
            help="The name of the class to use to write the predictions to the output file",
        )
        self.parser.add_argument(
            "--output-writer-kwargs",
            required=False,
            help="A serialized json object which will be deserialized and passed as "
            "**kwargs to the output writer constructor",
        )
        self.parser.add_argument(
            "--log-file",
            required=False,
            help="The file where the log should be written",
        )
        self.parser.add_argument(
            "--silent",
            required=False,
            action="store_true",
            help="Indicates the log should not be written to stdout",
        )
        self.parser.set_defaults(func=self.run)

    @staticmethod
    def _check_args(args):
        # --dataset-name and --input-files are exclusive-or
        if (args.dataset_name is not None) is (args.input_files is not None):
            raise ValueError(
                "Exactly one of --dataset-name or --input-files must be set"
            )

        # Both --dataset-name and --split must either be set or not set
        if (args.dataset_name is not None) is not (args.split is not None):
            raise ValueError(
                "Parameters --dataset-name and --split must either both be set "
                "or neither be set"
            )

        # Both --input-files and --dataset-reader must either be set or not set
        if (args.input_files is not None) is not (args.dataset_reader is not None):
            raise ValueError(
                "Parameters --input-files and --dataset-reader must either both be set "
                "or neither be set"
            )

        # If --dataset-reader-kwargs is passed, --dataset-reader must also be used
        if args.dataset_reader_kwargs is not None:
            if args.dataset_reader is None:
                raise ValueError(
                    "Parameter --dataset-reader must be used if --dataset-reader-kwargs "
                    "is also used"
                )

    @overrides
    def run(self, args):
        self._check_args(args)
        prepare_global_logging(args.log_file, args.silent)

        model = load_model(args.model_name, args.model_kwargs)

        if args.dataset_name is not None:
            dataset_reader = HuggingfaceDatasetsDatasetReader(
                args.dataset_name, args.split
            )
            instances = dataset_reader.read()
        else:
            dataset_reader = load_dataset_reader(
                args.dataset_reader, args.dataset_reader_kwargs
            )
            instances = dataset_reader.read(*args.input_files)

        predictions = predict_with_model(model, instances, args.predict_kwargs)

        output_writer = load_output_writer(
            args.output_writer, args.output_writer_kwargs
        )
        output_writer.write(
            instances, predictions, args.output, model_name=args.model_name
        )
