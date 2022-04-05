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
    """
    A :code:`ParallelModel` is a simple abstraction around the :code:`joblib` library
    for running models in parallel. It allows for parallel processing on multiple
    CPUs as well as GPUs.

    To create a :code:`ParallelModel`, you must specify the type of the model
    that is being run in parallel as well as a list of :code:`kwargs` that will
    be passed to the model's constructor to instantiate the parallel models. The
    number of parallel processes is equal to the length of the list of :code:`kwargs`.

    The :code:`ParallelModel`'s :code:`predict_batch()` methods will divide the
    :code:`inputs` into batches, pass the a :code:`kwargs` object and a batch
    to a worker. The worker then instantiates the model, processes the data, and
    returns the result. Because the implementation relies on :code:`joblib`, the :code:`kwargs`
    and the output from the model must be serializable by :code:`joblib`.

    The output from the :code:`predict_batch()` method will be the list of
    outputs returned by each of the individual processes.

    If the input ordering matters or the model does some final aggregation over
    all of the items in the :code:`inputs` passed to :code:`predict_batch()`, the
    :code:`ParallelModel` will not compute the right result.

    **Note: Please make sure you understand how ParallelModel is implemented
    before you use it to ensure the behavior is expected for your use case.**

    Parameters
    ----------
    model_cls : Type
        The model class which will be run in parallel
    model_kwargs_list : List
        A list of :code:`kwargs` that will be used to create models of
        type :code:`model_cls`. The length of the list is the
        number of parallel processes to use. If all of the :code:`kwargs`
        are equal to :code:`{}`, you may use the :code:`num_models`
        parameter instead.
    num_models : int
        The number of models to run in parallel. Ignored if :code:`model_kwargs_list`
        is provided. If `model_kwargs_list` is None, then passing :code:`num_models`
        is equivalent to passing a list of :code:`num_models` empty
        :code:`kwargs` (i.e., :code:`{}`).

    Examples
    --------
    First, we define a simple model to run in parallel. This model simply
    multiplies all of its inputs by 10.

    .. code-block:: python

        from repro.models import Model

        class TimesTen(Model):
            def predict_batch(self, inputs: List[Dict[str, int]]):
                return [inp["value"] * 10 for inp in inputs]

    Then we specify some inputs:

    .. code-block:: python

        inputs = [{"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}]

    Now, create a :code:`ParallelModel` with :code:`num_models=2`. This will
    result in running two :code:`TimesTen` models in parallel.

    .. code-block:: python

        from repro.models import ParallelModel
        parallel_model = ParallelModel(TimesTen, num_models=2)

    Then the :code:`inputs` can be processed in parallel:

    .. code-block:: python

        output_list = parallel_model.predict_batch(inputs)

    The :code:`output_list` will be equal to:

    .. code-block:: json

        [
            [{"value": 0, "value": 10}],
            [{"value": 20, "value": 30}],
        ]

    where each of the elements corresponds to the output from each of the
    two models. This example is equivalent to the following serial execution:

    .. code-block:: python

        model = TimesTen()
        outputs1 = model.predict_batch([inputs[0], inputs[1]])
        outputs2 = model.predict_batch([inputs[2], inputs[3]])
        outputs_list = [outputs1, outputs2]

    If you need to pass specific parameters to each of the model's constructors,
    you may do so using :code:`model_kwargs_list`. For example, if the model
    required a GPU ID, you could pass that information as such:

    .. code-block:: python

        class GPUModel(Model):
            def __init__(self, device: int):
                self.device = device

            def predict_batch(self, inputs: List[Dict[str, int]]):
                # Do some computation
                return result

        parallel_model = ParallelModel(GPUModel, [{"device": 0}, {"device": 2}])

    This will run two instances of :code:`GPUModel` in parallel. One process will
    use :code:`device=0` and the other :code:`device=2`.
    """

    def __init__(
        self,
        model_cls: Type[Model],
        model_kwargs_list: List[Dict[str, Any]] = None,
        num_models: int = None,
    ) -> None:
        self.model_cls = model_cls

        if model_kwargs_list is None and num_models is None:
            raise ValueError(
                f"Either `model_kwargs_list` or `num_models` must not be `None`"
            )
        if model_kwargs_list:
            self.model_kwargs_list = model_kwargs_list
        else:
            self.model_kwargs_list = [{} for _ in range(num_models)]

    @staticmethod
    def _divide_into_batches(inputs: List[Any], num_batches: int) -> List[List[Any]]:
        batch_size = int(math.ceil(len(inputs) / num_batches))
        batches = []
        for i in range(0, len(inputs), batch_size):
            batches.append(inputs[i : i + batch_size])
        return batches

    def _process(
        self, model_kwargs: Dict[str, Any], inputs: List[Dict[str, Any]], **kwargs
    ) -> Any:
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
