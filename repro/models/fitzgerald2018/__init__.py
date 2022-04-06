import os

VERSION = "1.1"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.fitzgerald2018.model import QASRLParser
from repro.models.fitzgerald2018.setup import FitzGerald2018SetupSubcommand
