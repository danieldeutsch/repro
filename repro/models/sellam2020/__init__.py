import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.sellam2020.model import BLEURT
from repro.models.sellam2020.setup import Sellam2020SetupSubcommand
