import argparse

from repro.common import Registrable


class Subcommand(Registrable):
    def add_subparser(self, parser: argparse._SubParsersAction) -> None:
        raise NotImplementedError

    def run(self, args):
        raise NotImplementedError


# These empty subcommand classes are used to be able to automatically add the
# subcommands to the argument parser in the right locations. For example, any command
# which is registered under the `RootSubcommand` will become a command directly under "repro"
class RootSubcommand(Subcommand):
    pass


class SetupSubcommand(Subcommand):
    pass
