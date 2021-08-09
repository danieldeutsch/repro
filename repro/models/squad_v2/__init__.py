import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = False

from repro.models.squad_v2.model import SQuADv2Evaluation
from repro.models.squad_v2.setup import SQuADv2SetupSubcommand
