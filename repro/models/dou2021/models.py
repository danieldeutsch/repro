import logging
from typing import Dict, List, Union

from repro.data.types import DocumentType, SummaryType
from repro.models import Model, SingleDocumentSummarizationModel
from repro.models.liu2019 import BertSumExt
from repro.models.dou2021 import DEFAULT_IMAGE, MODEL_NAME
from repro.models.dou2021.commands import (
    get_oracle_sentences,
    generate_summaries,
    sentence_split,
)

logger = logging.getLogger(__name__)


@Model.register(f"{MODEL_NAME}-oracle-sentence-gsum")
class OracleSentenceGSumModel(Model):
    def __init__(
        self, image: str = DEFAULT_IMAGE, device: int = 0, batch_size: int = 16
    ) -> None:
        self.model = "bart_sentence"
        self.image = image
        self.device = device
        self.batch_size = batch_size

    def predict(
        self,
        document: DocumentType,
        reference: SummaryType = None,
        guidance: SummaryType = None,
        **kwargs,
    ) -> SummaryType:
        if reference is None and guidance is None:
            raise Exception(f"Either `reference` or `guidance` must be provided")
        return self.predict_batch(
            [{"document": document, "reference": reference, "guidance": guidance}]
        )[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[DocumentType, SummaryType]]], **kwargs
    ) -> List[SummaryType]:
        logger.info(
            f"Generating summaries for {len(inputs)} inputs and image {self.image}."
        )

        # Check if all inputs have "guidance" provided. If they do, we will use that
        # instead of computing the guidance
        def _has_guidance(inp: Dict[str, Union[DocumentType, SummaryType]]) -> bool:
            return "guidance" in inp and inp["guidance"] is not None

        compute_guidance = True
        if any(_has_guidance(inp) for inp in inputs):
            if not all(_has_guidance(inp) for inp in inputs):
                raise Exception(
                    f"Only some of the inputs have `guidance` inputs. Only all or none is supported"
                )
            logger.info("Using input guidance")
            compute_guidance = False
        else:
            logger.info("Computing guidance from references")

        # If the documents and references are pre-sentence split, we will maintain
        # that split. Otherwise, we run sentence splitting
        documents = [inp["document"] for inp in inputs]
        if any(isinstance(document, str) for document in documents):
            if any(isinstance(document, list) for document in documents):
                logger.warning(
                    "`documents` contains both sentence-split and un-sentence-split documents. "
                    "The sentence-split boundaries will be ignored and sentence splitting will "
                    "be run again."
                )
            tokenized_documents = sentence_split(self.image, documents)
        else:
            tokenized_documents = documents

        if compute_guidance:
            references = [inp["reference"] for inp in inputs]
            if any(isinstance(reference, str) for reference in references):
                # Sentence splitting needs to be run
                if any(isinstance(reference, list) for reference in references):
                    logger.warning(
                        "`references` contains both sentence-split and un-sentence-split references. "
                        "The sentence-split boundaries will be ignored and sentence splitting will "
                        "be run again."
                    )
                references = sentence_split(self.image, references)
            guidance = get_oracle_sentences(self.image, tokenized_documents, references)
        else:
            guidance = [inp["guidance"] for inp in inputs]

        # We pass the original, untokenized documents to the generation.
        summaries = generate_summaries(
            self.image, self.model, self.device, self.batch_size, documents, guidance
        )
        return summaries


@Model.register(f"{MODEL_NAME}-sentence-gsum")
class SentenceGSumModel(SingleDocumentSummarizationModel):
    def __init__(
        self, image: str = DEFAULT_IMAGE, device: int = 0, batch_size: int = 16
    ) -> None:
        self.model = "bart_sentence"
        self.image = image
        self.device = device
        self.batch_size = batch_size
        self.extractive_model = BertSumExt(device=device)

    def predict_batch(
        self, inputs: List[Dict[str, Union[DocumentType, SummaryType]]], *args, **kwargs
    ) -> List[SummaryType]:
        logger.info(
            f"Generating summaries for {len(inputs)} inputs and image {self.image}."
        )
        documents = [inp["document"] for inp in inputs]

        # The sentence supervision is done using the `self.extractive_model`, which
        # is in its own Docker container, so we retrieve those summaries first
        logger.info(f"Extracting guidance signal")
        guidance = self.extractive_model.predict_batch(inputs)

        summaries = generate_summaries(
            self.image, self.model, self.device, self.batch_size, documents, guidance
        )
        return summaries
