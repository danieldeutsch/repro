import argparse

from overrides import overrides

from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand, build_image


@SetupSubcommand.register("liu2019")
class Liu2019SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("liu2019", f"{MODELS_ROOT}/liu2019")

    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = f'Build the docker image "{self.image}"'
        self.parser = parser.add_parser(
            self.image, description=description, help=description
        )
        self.parser.add_argument(
            "--transformerabs-cnndm",
            action="store_true",
            help="Indicates the TransformerAbs-CNNDM model should be downloaded",
        )
        self.parser.add_argument(
            "--bertsumext-cnndm",
            action="store_true",
            help="Indicates the BertSumExt-CNNDM model should be downloaded",
        )
        self.parser.add_argument(
            "--bertsumextabs-cnndm",
            action="store_true",
            help="Indicates the BertSumExtAbs-CNNDM model should be downloaded",
        )
        self.parser.add_argument(
            "--bertsumextabs-xsum",
            action="store_true",
            help="Indicates the BertSumExtAbs-XSum model should be downloaded",
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
            "TRANSFORMERABS_CNNDM": "true" if args.transformerabs_cnndm else "false",
            "BERTSUMEXT_CNNDM": "true" if args.bertsumext_cnndm else "false",
            "BERTSUMEXTABS_CNNDM": "true" if args.bertsumextabs_cnndm else "false",
            "BERTSUMEXTABS_XSUM": "true" if args.bertsumextabs_xsum else "false",
        }
        build_image(self.root, self.image, build_args=build_args, silent=args.silent)
