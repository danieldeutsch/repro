import json
from typing import Any, Dict, List, T, Type, Union

from repro.common import Registrable
from repro.data.dataset_readers import DatasetReader
from repro.data.output_writers import OutputWriter
from repro.models import Model


def flatten(text: Union[str, List[str]], separator: str = None) -> str:
    """
    Flattens the text item to a string. If the input is a string, that
    same string is returned. Otherwise, the text is joined together with
    the separator.

    Parameters
    ----------
    text : Union[str, List[str]]
        The text to flatten
    separator : str, default=None
        The separator to join the list with. If `None`, the separator will be " "

    Returns
    -------
    str
        The flattened text
    """
    separator = separator or " "
    if isinstance(text, list):
        return separator.join(text)
    return text


def _load_type(
    base_type: Type[T], name: str, args: Union[str, Dict[str, Any]] = None
) -> T:
    args = args or {}
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError(
            f"`args` is expected to be a dictionary or json-serialized dictionary: {args}"
        )

    type_, _ = Registrable._registry[base_type][name]
    obj = type_(**args)
    return obj


def load_model(name: str, args: Union[str, Dict[str, Any]] = None) -> Model:
    """
    Loads a `Model` given the registered `name` of the model and any arguments
    which will be passed to the constructor as kwargs. `args` should be a dictionary
    or a json-serialized dictionary in which the keys correspond to constructor parameters.

    Parameters
    ----------
    name : str
        The name of the model to load
    args : Union[str, Dict[str, Any]]
        The kwargs to be passed to the model's constructor

    Returns
    -------
    Model
        The model
    """
    return _load_type(Model, name, args)


def load_dataset_reader(
    name: str, args: Union[str, Dict[str, Any]] = None
) -> DatasetReader:
    """
    Loads a `DatasetReader` given the registered `name` of the reader and any arguments
    which will be passed to the constructor as kwargs. `args` should be a dictionary
    or a json-serialized dictionary in which the keys correspond to constructor parameters.

    Parameters
    ----------
    name : str
        The name of the dataset reader to load
    args : Union[str, Dict[str, Any]]
        The kwargs to be passed to the dataset reader's constructor

    Returns
    -------
    DatasetReader
        The dataset reader
    """
    return _load_type(DatasetReader, name, args)


def load_output_writer(
    name: str, args: Union[str, Dict[str, Any]] = None
) -> DatasetReader:
    """
    Loads an `OutputWriter` given the registered `name` of the reader and any arguments
    which will be passed to the constructor as kwargs. `args` should be a dictionary
    or a json-serialized dictionary in which the keys correspond to constructor parameters.

    Parameters
    ----------
    name : str
        The name of the output writer to load
    args : Union[str, Dict[str, Any]]
        The kwargs to be passed to the output writer's constructor

    Returns
    -------
    OutputWriter
        The output writer
    """
    return _load_type(OutputWriter, name, args)
