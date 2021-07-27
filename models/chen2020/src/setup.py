from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("chen2020")
class Chen2020SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("chen2020", f"{MODELS_ROOT}/chen2020")
