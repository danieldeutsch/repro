import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image

from .metadata import DEFAULT_IMAGE, MODEL_NAME


@SetupSubcommand.register(MODEL_NAME)
class Zhang2020SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__(MODEL_NAME, DEFAULT_IMAGE, f"{MODELS_ROOT}/{MODEL_NAME}")

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = f'Build the docker image "{self.image}"'
        self.parser = parser.add_parser(
            self.model, description=description, help=description
        )
        self.parser.add_argument(
            "--models",
            nargs="+",
            help="Indicates which models should be downloaded",
            default=["roberta-large"],
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
            "MODELS": " ".join(args.models),
        }
        build_image(self.root, self.image, build_args=build_args, silent=args.silent)
