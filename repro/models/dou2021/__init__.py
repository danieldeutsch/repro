import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.dou2021.dataset_reader import Dou2021DatasetReader
from repro.models.dou2021.models import OracleSentenceGSumModel, SentenceGSumModel
from repro.models.dou2021.setup import Dou2021SetupSubcommand
