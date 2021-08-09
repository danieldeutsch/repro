import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = False

from repro.models.dugan2020.model import RoFTRecipeGenerator
from repro.models.dugan2020.setup import Dugan2020SetupSubcommand
