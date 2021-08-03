from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("durmus2020")
class Durmus2020SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("durmus2020", f"{MODELS_ROOT}/durmus2020")
