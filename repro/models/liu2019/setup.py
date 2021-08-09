import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image
from repro.models.liu2019 import DEFAULT_IMAGE, MODEL_NAME


@SetupSubcommand.register(MODEL_NAME)
class Liu2019SetupSubcommand(BuildDockerImageSubcommand):
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
            "--not-transformerabs-cnndm",
            action="store_true",
            help="Indicates the TransformerAbs-CNNDM model should not be downloaded",
        )
        self.parser.add_argument(
            "--not-bertsumext-cnndm",
            action="store_true",
            help="Indicates the BertSumExt-CNNDM model should not be downloaded",
        )
        self.parser.add_argument(
            "--not-bertsumextabs-cnndm",
            action="store_true",
            help="Indicates the BertSumExtAbs-CNNDM model should not be downloaded",
        )
        self.parser.add_argument(
            "--not-bertsumextabs-xsum",
            action="store_true",
            help="Indicates the BertSumExtAbs-XSum model should not be downloaded",
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
            "TRANSFORMERABS_CNNDM": "false"
            if args.not_transformerabs_cnndm
            else "true",
            "BERTSUMEXT_CNNDM": "false" if args.not_bertsumext_cnndm else "true",
            "BERTSUMEXTABS_CNNDM": "false" if args.not_bertsumextabs_cnndm else "true",
            "BERTSUMEXTABS_XSUM": "false" if args.not_bertsumextabs_xsum else "true",
        }
        build_image(
            self.root, args.image_name, build_args=build_args, silent=args.silent
        )
