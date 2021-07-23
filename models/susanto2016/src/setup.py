from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("susanto2016")
class Susanto2016SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("susanto2016", f"{MODELS_ROOT}/susanto2016")
