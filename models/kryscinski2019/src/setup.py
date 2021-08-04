from repro import MODELS_ROOT
from repro.commands.subcommand import SetupSubcommand
from repro.common.docker import BuildDockerImageSubcommand


@SetupSubcommand.register("kryscinski2019")
class Kryscinski2019SetupSubcommand(BuildDockerImageSubcommand):
    def __init__(self) -> None:
        super().__init__("kryscinski2019", f"{MODELS_ROOT}/kryscinski2019")
