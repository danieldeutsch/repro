import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image
from repro.models.lewis2020 import DEFAULT_IMAGE, MODEL_NAME


@SetupSubcommand.register(MODEL_NAME)
class Lewis2020SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__(f"{MODELS_ROOT}/{MODEL_NAME}", DEFAULT_IMAGE)

    @overrides
    def add_subparser(self, model: str, parser: argparse._SubParsersAction):
        description = f'Build a Docker image for model "{model}"'
        self.parser = parser.add_parser(
            model, description=description, help=description
        )
        self.parser.add_argument(
            "--image-name",
            default=DEFAULT_IMAGE,
            help="The name of the image to build",
        )
        self.parser.add_argument(
            "--not-cnndm",
            action="store_true",
            help="Indicates the model trained on CNN/DM model should not be downloaded",
        )
        self.parser.add_argument(
            "--not-xsum",
            action="store_true",
            help="Indicates the model trained on XSum should not be downloaded",
        )
        self.parser.add_argument(
            "--silent",
            action="store_true",
            help="Silences the output from the build command",
        )
        self.parser.set_defaults(subfunc=self.run)

    @overrides
    def run(self, args):
        build_args = {
            "CNNDM": "false" if args.not_cnndm else "true",
            "XSUM": "false" if args.not_xsum else "true",
        }
        build_image(
            self.root, args.image_name, build_args=build_args, silent=args.silent
        )
