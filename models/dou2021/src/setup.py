import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image


@SetupSubcommand.register("dou2021")
class Dou2021SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("dou2021", f"{MODELS_ROOT}/dou2021")

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = f'Build the docker image "{self.image}"'
        self.parser = parser.add_parser(
            self.image, description=description, help=description
        )
        self.parser.add_argument(
            "--sentence-guided",
            action="store_true",
            help="Indicates the sentence-guided model should be downloaded",
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
            "SENTENCE_GUIDED": "true" if args.sentence_guided else "false",
        }
        build_image(self.root, self.image, build_args=build_args, silent=args.silent)
