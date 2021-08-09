import os

VERSION = "1.0"
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
DOCKERHUB_REPRO = f"danieldeutsch/{MODEL_NAME}"
DEFAULT_IMAGE = f"{DOCKERHUB_REPRO}:{VERSION}"
AUTOMATICALLY_PUBLISH = True

from repro.models.deutsch2021.dataset_reader import (
    Deutsch2021QuestionAnsweringEvaluationDatasetReader,
)
from repro.models.deutsch2021.models import (
    QAEval,
    QAEvalQuestionAnsweringModel,
    QAEvalQuestionGenerationModel,
)
from repro.models.deutsch2021.setup import Deutsch2021SetupSubcommand
