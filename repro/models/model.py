import math
from joblib import Parallel, delayed
from typing import Any, Dict, List, Type, Union

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


class ParallelModel(Model):
    def __init__(self, model_cls: Type, model_kwargs_list: List[Dict[str, Any]]) -> None:
        self.model_cls = model_cls
        self.model_kwargs_list = model_kwargs_list

    @staticmethod
    def _divide_into_batches(inputs: List[Any], num_batches: int) -> List[List[Any]]:
        batch_size = int(math.ceil(len(inputs) / num_batches))
        batches = []
        for i in range(0, len(inputs), batch_size):
            batches.append(inputs[i:i + batch_size])
        return batches

    def _process(self, model_kwargs: Dict[str, Any], inputs: List[Dict[str, Any]], **kwargs) -> Any:
        model = self.model_cls(**model_kwargs)
        return model.predict_batch(inputs, **kwargs)

    def predict_batch(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Any]:
        # Divide all of the inputs into batches, maintaining the order
        num_jobs = len(self.model_kwargs_list)
        batches = self._divide_into_batches(inputs, num_jobs)

        # Create the jobs which will be run in parallel
        jobs = []
        for model_kwargs, batch in zip(self.model_kwargs_list, batches):
            jobs.append(delayed(self._process)(model_kwargs, batch))

        # Run the jobs
        results = Parallel(n_jobs=num_jobs)(jobs)
        return results


class QuestionAnsweringModel(Model):
    def predict(self, context: str, question: str, *args, **kwargs) -> str:
        return self.predict_batch(
            [{"context": context, "question": question}], **kwargs
        )[0]

    def predict_batch(self, inputs: List[Dict[str, str]], *args, **kwargs) -> List[str]:
        raise NotImplementedError


class QuestionGenerationModel(Model):
    def predict(self, context: str, start: int, end: int, **kwargs) -> str:
        return self.predict_batch([{"context": context, "start": start, "end": end}])[0]

    def predict_batch(self, inputs: List[Dict[str, str]], **kwargs) -> List[str]:
        raise NotImplementedError


class RecipeGenerationModel(Model):
    def predict(self, name: str, ingredients: List[str], *args, **kwargs) -> str:
        return self.predict_batch([{"name": name, "ingredients": ingredients}])[0]

    def predict_batch(
        self, inputs: List[Dict[str, Union[str, List[str]]]], *args, **kwargs
    ) -> List[str]:
        raise NotImplementedError


class SingleDocumentSummarizationModel(Model):
    def predict(self, document: DocumentType, *args, **kwargs) -> SummaryType:
        return self.predict_batch([{"document": document}])[0]

    def predict_batch(
        self, inputs: List[Dict[str, DocumentType]], *args, **kwargs
    ) -> List[SummaryType]:
        raise NotImplementedError


class TruecasingModel(Model):
    def predict(self, text: str, *args, **kwargs) -> str:
        return self.predict_batch([{"text": text}])[0]

    def predict_batch(
        self, inputs: List[Dict[str, DocumentType]], *args, **kwargs
    ) -> List[str]:
        raise NotImplementedError
