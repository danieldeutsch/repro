import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.scialom2021.models import (
    QuestEval,
    QuestEvalForSimplification,
    QuestEvalForSummarization,
)
from repro.models.scialom2021.setup import Scialom2021SetupSubcommand
