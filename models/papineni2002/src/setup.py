from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("papineni2002")
class Papineni2002SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("papineni2002", f"{MODELS_ROOT}/papineni2002")
