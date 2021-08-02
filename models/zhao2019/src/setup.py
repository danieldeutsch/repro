from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("zhao2019")
class Zhao2019SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("zhao2019", f"{MODELS_ROOT}/zhao2019")
