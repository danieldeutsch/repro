import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.chen2020.dataset_reader import Chen2020EvaluationDatasetReader
from repro.models.chen2020.models import LERC, MOCHAEvaluationMetric
from repro.models.chen2020.setup import Chen2020SetupSubcommand
