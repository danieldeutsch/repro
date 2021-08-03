from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("deutsch2021")
class Deutsch2021SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("deutsch2021", f"{MODELS_ROOT}/deutsch2021")
