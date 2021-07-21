import argparse
import inspect
from overrides import overrides
from typing import List

from repro.commands.subcommand import RootSubcommand
from repro.common.logging import prepare_global_logging
from repro.common.util import load_dataset_reader, load_model, load_output_writer
from repro.data.dataset_readers import HuggingfaceDatasetsDatasetReader
from repro.data.types import InstanceDict
from repro.models import Model


def predict_with_model(model: Model, instances: List[InstanceDict]) -> List:
    # Find the required arguments for the model's `predict` function, then
    # select all of those fields from the `instances` and pass them through
    # the `predict_batch` function
    args = inspect.getfullargspec(model.predict)
    required_args = args.args
    required_args.remove("self")

    inputs = []
    for instance in instances:
        inputs.append({arg: instance[arg] for arg in required_args})

    predictions = model.predict_batch(inputs)
    if len(predictions) != len(instances):
        raise Exception(
            f"Model returned {len(predictions)} predictions for {len(instances)} instances"
        )
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
            "--model-args",
            required=False,
            help="A serialized json object which will be deserialized and passed as "
            "**kwargs to the model constructor",
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
            "--dataset-reader-args",
            required=False,
            help="A serialized json object which will be deserialized and passed as "
            "**kwargs to the dataset reader constructor",
        )
        self.parser.add_argument(
            "--output-file",
            required=True,
            help="The name of the output jsonl file that the model predictions will be written to",
        )
        self.parser.add_argument(
            "--output-writer",
            required=False,
            default="default",
            help="The name of the class to use to write the predictions to the output file",
        )
        self.parser.add_argument(
            "--output-writer-args",
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

        # If --dataset-reader-args is passed, --dataset-reader must also be used
        if args.dataset_reader_args is not None:
            if args.dataset_reader is None:
                raise ValueError(
                    "Parameter --dataset-reader must be used if --dataset-reader-args "
                    "is also used"
                )

    @overrides
    def run(self, args):
        self._check_args(args)
        prepare_global_logging(args.log_file, args.silent)

        model = load_model(args.model_name, args.model_args)

        if args.dataset_name is not None:
            dataset_reader = HuggingfaceDatasetsDatasetReader(
                args.dataset_name, args.split
            )
            instances = dataset_reader.read()
        else:
            dataset_reader = load_dataset_reader(
                args.dataset_reader, args.dataset_reader_args
            )
            instances = dataset_reader.read(*args.input_files)

        predictions = predict_with_model(model, instances)

        output_writer = load_output_writer(args.output_writer, args.output_writer_args)
        output_writer.write(
            instances, predictions, args.output_file, model_name=args.model_name
        )
