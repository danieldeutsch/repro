from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("dou2021")
class Dou2021SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("dou2021", f"{MODELS_ROOT}/dou2021")
