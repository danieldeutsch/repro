from typing import Any, Dict, List

from repro.common import Registrable
from repro.data.types import DocumentType, SummaryType


class Model(Registrable):
    def predict(self, *args, **kwargs):
        """
        Runs inference for the single input instance.
        """
        raise NotImplementedError

    def predict_batch(self, inputs: List[Dict[str, Any]], *args, **kwargs):
        """
        Runs inference for all of the instances in `inputs`. Each item in `inputs`
        should have a key which corresponds to a parameter of the `predict` method.
        For instance, if the signature of predict was:

            `def predict(input1, input2)`

        Then each item in `inputs` should be a dictionary with keys `"input1"` and `"input2"`.
        """
        raise NotImplementedError


class QuestionAnsweringModel(Model):
    def predict(self, context: str, question: str, *args, **kwargs) -> str:
        return self.predict_batch([{"context": context, "question": question}])[0]

    def predict_batch(self, inputs: List[Dict[str, str]], *args, **kwargs) -> List[str]:
        raise NotImplementedError


class SingleDocumentSummarizationModel(Model):
    def predict(self, document: DocumentType, *args, **kwargs) -> SummaryType:
        return self.predict_batch([{"document": document}])[0]

    def predict_batch(
        self, inputs: List[Dict[str, DocumentType]], *args, **kwargs
    ) -> List[SummaryType]:
        raise NotImplementedError
