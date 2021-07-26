import json
from typing import Any, Dict, T, Type, Union

from repro.common import Registrable
from repro.data.dataset_readers import DatasetReader
from repro.data.output_writers import OutputWriter
from repro.models import Model


def _load_type(
    base_type: Type[T], name: str, kwargs: Union[str, Dict[str, Any]] = None
) -> T:
    kwargs = kwargs or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"`kwargs` is expected to be a dictionary or json-serialized dictionary: {kwargs}"
        )

    type_, _ = Registrable._registry[base_type][name]
    obj = type_(**kwargs)
    return obj


def load_model(name: str, kwargs: Union[str, Dict[str, Any]] = None) -> Model:
    """
    Loads a `Model` given the registered `name` of the model and any arguments
    which will be passed to the constructor as kwargs. `args` should be a dictionary
    or a json-serialized dictionary in which the keys correspond to constructor parameters.

    Parameters
    ----------
    name : str
        The name of the model to load
    kwargs : Union[str, Dict[str, Any]]
        The kwargs to be passed to the model's constructor

    Returns
    -------
    Model
        The model
    """
    return _load_type(Model, name, kwargs)


def load_dataset_reader(
    name: str, kwargs: Union[str, Dict[str, Any]] = None
) -> DatasetReader:
    """
    Loads a `DatasetReader` given the registered `name` of the reader and any arguments
    which will be passed to the constructor as kwargs. `args` should be a dictionary
    or a json-serialized dictionary in which the keys correspond to constructor parameters.

    Parameters
    ----------
    name : str
        The name of the dataset reader to load
    kwargs : Union[str, Dict[str, Any]]
        The kwargs to be passed to the dataset reader's constructor

    Returns
    -------
    DatasetReader
        The dataset reader
    """
    return _load_type(DatasetReader, name, kwargs)


def load_output_writer(
    name: str, kwargs: Union[str, Dict[str, Any]] = None
) -> DatasetReader:
    """
    Loads an `OutputWriter` given the registered `name` of the reader and any arguments
    which will be passed to the constructor as kwargs. `args` should be a dictionary
    or a json-serialized dictionary in which the keys correspond to constructor parameters.

    Parameters
    ----------
    name : str
        The name of the output writer to load
    kwargs : Union[str, Dict[str, Any]]
        The kwargs to be passed to the output writer's constructor

    Returns
    -------
    OutputWriter
        The output writer
    """
    return _load_type(OutputWriter, name, kwargs)
