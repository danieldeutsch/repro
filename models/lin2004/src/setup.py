from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("lin2004")
class Lin2004SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("lin2004", f"{MODELS_ROOT}/lin2004")
