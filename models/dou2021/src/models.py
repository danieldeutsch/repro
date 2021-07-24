import logging
from typing import Dict, List, Union

from repro.data.types import DocumentType, SummaryType
from repro.models import Model, SingleDocumentSummarizationModel
from repro.models.liu2019 import BertSumExt

from .commands import get_oracle_sentences, generate_summaries, sentence_split

logger = logging.getLogger(__name__)


@Model.register("dou2021-oracle-sentence-gsum")
class OracleSentenceGSumModel(Model):
    def __init__(
        self, image: str = "dou2021", device: int = 0, batch_size: int = 16
    ) -> None:
        self.model = "bart_sentence"
        self.image = image
        self.device = device
        self.batch_size = batch_size

    def predict(
        self, document: DocumentType, reference: SummaryType, *args, **kwargs
    ) -> SummaryType:
        return self.predict_batch([{"document": document, "reference": reference}])[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[DocumentType, SummaryType]]], *args, **kwargs
    ) -> List[SummaryType]:
        logger.info(
            f"Generating summaries for {len(inputs)} inputs and image {self.image}."
        )

        documents = [inp["document"] for inp in inputs]
        references = [inp["reference"] for inp in inputs]

        # If the documents and references are pre-sentence split, we will maintain
        # that split. Otherwise, we run sentence splitting
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

        # We pass the original, untokenized documents to the generation.
        summaries = generate_summaries(
            self.image, self.model, self.device, self.batch_size, documents, guidance
        )
        return summaries


@Model.register("dou2021-sentence-gsum")
class SentenceGSumModel(SingleDocumentSummarizationModel):
    def __init__(
        self, image: str = "dou2021", device: int = 0, batch_size: int = 16
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
