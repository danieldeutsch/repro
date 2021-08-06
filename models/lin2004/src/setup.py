from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand

from .metadata import MODEL_NAME, DEFAULT_IMAGE


@SetupSubcommand.register(MODEL_NAME)
class Lin2004SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__(MODEL_NAME, DEFAULT_IMAGE, f"{MODELS_ROOT}/{MODEL_NAME}")
