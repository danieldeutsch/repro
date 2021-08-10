from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand
from repro.models.thompson2020 import DEFAULT_IMAGE, MODEL_NAME


@SetupSubcommand.register(MODEL_NAME)
class Thompson2020SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__(f"{MODELS_ROOT}/{MODEL_NAME}", DEFAULT_IMAGE)
