from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("sacrerouge")
class SacreROUGESetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("sacrerouge", f"{MODELS_ROOT}/sacrerouge")
