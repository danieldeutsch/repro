from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("squad-v2")
class SQuADv2SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("squad-v2", f"{MODELS_ROOT}/squad_v2")
