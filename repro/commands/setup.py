import argparse
from overrides import overrides

from repro.common import Registrable
from repro.commands.subcommand import RootSubcommand, SetupSubcommand


@RootSubcommand.register("setup")
class SetupModelSubcommand(RootSubcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction):
        description = "Setup a model"
        self.parser = parser.add_parser(
            "setup", description=description, help=description
        )
        subparsers = self.parser.add_subparsers()

        # Add all of the setup commands using the registry
        for name, (cls_, _) in sorted(Registrable._registry[SetupSubcommand].items()):
            cls_().add_subparser(name, subparsers)

        self.parser.set_defaults(func=self.run)

    @overrides
    def run(self, args):
        if "subfunc" in dir(args):
            args.subfunc(args)
        else:
            self.parser.print_help()
