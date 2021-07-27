import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image


@SetupSubcommand.register("deutsch2021")
class Deutsch2021SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("deutsch2021", f"{MODELS_ROOT}/deutsch2021")

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = f'Build the docker image "{self.image}"'
        self.parser = parser.add_parser(
            self.image, description=description, help=description
        )
        self.parser.add_argument(
            "--question-generation",
            action="store_true",
            help="Indicates the question generation model",
        )
        self.parser.add_argument(
            "--question-answering",
            action="store_true",
            help="Indicates the question answering model",
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
            "QG": "true" if args.question_generation else "false",
            "QA": "true" if args.question_answering else "false",
        }
        build_image(self.root, self.image, build_args=build_args, silent=args.silent)
