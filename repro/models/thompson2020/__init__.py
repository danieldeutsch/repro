import os

VERSION = "1.3"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.thompson2020.model import Prism, PrismSrc
from repro.models.thompson2020.setup import Thompson2020SetupSubcommand
