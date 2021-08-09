import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image
from repro.models.sellam2020 import DEFAULT_IMAGE, MODEL_NAME


@SetupSubcommand.register(MODEL_NAME)
class Sellam2020SetupSubcommand(BuildDockerImageSubcommand):
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
        # The default models
        self.parser.add_argument(
            "--not-tiny-128",
            action="store_true",
            help="Indicates the BLEURT-Tiny-128 model should *not* be downloaded",
        )
        self.parser.add_argument(
            "--not-base-128",
            action="store_true",
            help="Indicates the BLEURT-Base-128 model should *not* be downloaded",
        )

        # Optional models
        self.parser.add_argument(
            "--tiny-512",
            action="store_true",
            help="Indicates the BLEURT-Tiny-512 model should be downloaded",
        )
        self.parser.add_argument(
            "--base-512",
            action="store_true",
            help="Indicates the BLEURT-Base-512 model should be downloaded",
        )
        self.parser.add_argument(
            "--large-128",
            action="store_true",
            help="Indicates the BLEURT-Large-128 model should be downloaded",
        )
        self.parser.add_argument(
            "--large-512",
            action="store_true",
            help="Indicates the BLEURT-Large-512 model should be downloaded",
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
            "TINY_128": "false" if args.not_tiny_128 else "true",
            "BASE_128": "false" if args.not_base_128 else "true",
            "TINY_512": "true" if args.tiny_512 else "false",
            "BASE_512": "true" if args.base_512 else "false",
            "LARGE_128": "true" if args.large_128 else "false",
            "LARGE_512": "true" if args.large_512 else "false",
        }
        build_image(
            self.root, args.image_name, build_args=build_args, silent=args.silent
        )
