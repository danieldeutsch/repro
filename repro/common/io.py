import json
import os
from typing import List, Union

from repro.common import util


def write_to_text_file(
    items: List[Union[str, List[str]]], file_path: str, separator: str = None
) -> None:
    """
    Writes the items in `items` to a text file with one item per line. If an individual
    item is a list, it is first flattened with the `separator`.

    Parameters
    ----------
    items : List[Union[str, List[str]]]
        The items to write to a file.
    file_path : str
        The path to the file where the output should be written
    separator : str, default=None
        The separator to use to join the items
    """
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, "w") as out:
        for item in items:
            item = util.flatten(item, separator=separator)
            out.write(item + "\n")


def write_to_jsonl_file(
    items: List[Union[str, List[str]]],
    key: str,
    file_path: str,
    flatten: bool = False,
    separator: str = None,
) -> None:
    """
    Writes the items in `items` to a jsonl file at `file_path`. Each item will correspond to one line. The
    item will be the value for the `key` in each json. If `flatten` is `True`, each item will
    first be flattened using `separator`.

    Parameters
    ----------
    items : List[Union[str, List[str]]]
        The items to serialize
    key : str
        The key for the items
    file_path : str
        The path to the file where the output should be written
    flatten : bool, default=False
        Indicates whether the items should first be flattened
    separator : str, default=None
        The separator for flattening the items
    """
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, "w") as out:
        for item in items:
            if flatten:
                item = util.flatten(item, separator=separator)
            out.write(json.dumps({key: item}) + "\n")


def read_jsonl_file(file_path: str, single_line: bool = False) -> List:
    """
    Loads a jsonl file and returns a list corresponding to each line in the file.

    Parameters
    ----------
    file_path : str
        The file to load the data from
    single_line : bool, default=False
        Indicates all of the json objects are on a single line (i.e., not separated
        by a newline character, but separated by "")

    Returns
    -------
    List
        The items, where each item corresponds to one line
    """
    items = []
    if single_line:
        decoder = json.JSONDecoder()
        contents = open(file_path, "r").read()
        offset = 0
        while offset < len(contents):
            item, length = decoder.raw_decode(contents[offset:])
            items.append(item)
            offset += length
    else:
        with open(file_path, "r") as f:
            for line in f:
                items.append(json.loads(line))
    return items
