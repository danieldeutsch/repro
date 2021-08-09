import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.papineni2002.models import BLEU, SentBLEU
from repro.models.papineni2002.setup import Papineni2002SetupSubcommand
