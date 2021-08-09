from typing import Any, Dict, List, Set, T, Tuple, Union

from repro.data.types import TextType

# `Indexable` is something which maps from an int to a value.
Indexable = Union[List[T], Dict[int, T]]

NestedDict = Union[Dict[str, float], "NestedDict"]


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


def _hash_texts(texts: List[TextType]) -> int:
    hashes = []
    for text in texts:
        if isinstance(text, str):
            hashes.append(hash(text))
        else:
            hashes.append(hash(tuple(text)))
    return hash(tuple(hashes))


def group_by_references(
    candidates: List[TextType], references_list: List[List[TextType]]
) -> Tuple[List[List[TextType]], List[List[TextType]], List[Tuple[int, int]]]:
    """
    Groups the `candidates` by identical references in `references_list`. This is
    enables evaluation metrics which need to do a lot of preprocessing of the
    references to reducing the amount of duplicate effort required.

    Parameters
    ----------
    candidates : List[TextType]
        The candidate texts
    references_list : List[List[TextType]]
        The reference texts.

    Returns
    -------
    List[List[TextType]]
        The grouped candidate texts in which the candidates at index `i` all
        share the same references in the grouped references
    List[List[TextType]]
        The grouped reference texts
    List[Tuple[int, int]]
        A mapping from the input `candidates` to the `(i, j)` position
        in the grouped candidates that it corresponds to.
    """
    mapping = []
    references_to_index = {}
    grouped_candidates_list = []
    grouped_references_list = []

    for candidate, references in zip(candidates, references_list):
        references_key = _hash_texts(references)
        if references_key not in references_to_index:
            # This is a new set of references. Update the data structures
            group_index = len(grouped_references_list)
            references_to_index[references_key] = group_index
            grouped_candidates_list.append([])
            grouped_references_list.append(references)

        group_index = references_to_index[references_key]
        candidate_index = len(grouped_candidates_list[group_index])
        grouped_candidates_list[group_index].append(candidate)
        mapping.append((group_index, candidate_index))

    return grouped_candidates_list, grouped_references_list, mapping


def ungroup_values(
    values: Indexable[Indexable[T]], mapping: List[Tuple[int, int]]
) -> List[T]:
    """
    Ungroups the values in `values_list` based on the group `mapping`. `mapping[i]` specifies
    the `(j, k)` position in `values_list` that the value for `i` corresponds to. The
    result of the function is the ungrouped values for each item in `mapping`.

    Parameters
    ----------
    values : Indexable[Indexable[T]]
        The grouped values
    mapping : List[Tuple[int, int]]
        The mapping which specifies where in the grouped values each
        item in the original order is in.

    Returns
    -------
    List[T]
        The ungrouped values.
    """
    ungrouped_values = []
    for i, j in mapping:
        ungrouped_values.append(values[i][j])
    return ungrouped_values


def flatten_nested_dict(nested_dict: NestedDict) -> Dict[Tuple[str, ...], float]:
    def _recursively_flatten(
        target: Dict[Tuple[str, ...], float], d: NestedDict, prefix: List[str]
    ) -> Dict[Tuple[str, ...], float]:
        for key, value in d.items():
            if isinstance(value, dict):
                _recursively_flatten(target, value, prefix + [key])
            else:
                target[tuple(prefix + [key])] = value
        return target

    return _recursively_flatten({}, nested_dict, [])


def unflatten_dict(d: Dict[Tuple[str, ...], float]) -> NestedDict:
    def _recursively_set(
        target: NestedDict, keys: Tuple[str], value: float, index: int
    ) -> NestedDict:
        if index == len(keys) - 1:
            target[keys[index]] = value
        else:
            key = keys[index]
            if key in target:
                _recursively_set(target[key], keys, value, index + 1)
            else:
                target[key] = _recursively_set({}, keys, value, index + 1)
        return target

    unflattened = {}
    for keys, value in d.items():
        _recursively_set(unflattened, keys, value, 0)
    return unflattened


def average_dicts(dicts: List[NestedDict]) -> NestedDict:
    if len(dicts) == 0:
        return {}

    # First, flatten all the dicts to make processing eaiser
    flattened_dicts = [flatten_nested_dict(d) for d in dicts]

    # Ensure that they all have the same keys. Different keys is not
    # currently supported
    keys = flattened_dicts[0].keys()
    for d in flattened_dicts[1:]:
        if d.keys() != keys:
            raise Exception(
                f"Dicts do not have identical keys. Expected {keys}, found {d.keys()}"
            )

    # Sum the values
    aggregate = {}
    for d in flattened_dicts:
        for key in keys:
            if key not in aggregate:
                aggregate[key] = d[key]
            else:
                aggregate[key] += d[key]

    # Normalize the values to calculate the average
    average = {key: value / len(dicts) for key, value in aggregate.items()}

    # Unflatten the average to match the input dicts
    return unflatten_dict(average)


def is_empty_text(text: TextType) -> bool:
    if isinstance(text, str):
        return len(text.strip()) == 0
    else:
        if len(text) == 0:
            return True
        for sentence in text:
            if len(sentence.strip()) > 0:
                return False
        return True


def remove_empty_inputs(inputs: List[TextType], *contexts: List[T]) -> Tuple[Any, ...]:
    for context in contexts:
        if len(inputs) != len(context):
            raise Exception(
                f"Each context must have the same length as the input. "
                f"Found {len(context)}, expected {len(inputs)}"
            )

    empty_indices = set()
    non_empty = []
    for i, (inp, *ctxs) in enumerate(zip(inputs, *contexts)):
        is_empty = is_empty_text(inp)
        if is_empty:
            empty_indices.add(i)
        else:
            non_empty.append((inp, *ctxs))

    non_empty_inputs, *non_empty_contexts = zip(*non_empty)
    non_empty_inputs = list(non_empty_inputs)
    non_empty_contexts = [list(context) for context in non_empty_contexts]
    return (empty_indices, non_empty_inputs, *non_empty_contexts)


def insert_empty_values(
    inputs: List[T], empty_indices: Set[int], empty_value: T
) -> List[T]:
    with_empty = []
    num = len(inputs) + len(empty_indices)

    if len(empty_indices) > 0:
        max_empty_index = max(empty_indices)
        if max_empty_index >= num:
            raise Exception(
                f"Found invalid empty index. Found {max_empty_index} for length {num}"
            )

    index = 0
    for i in range(num):
        if i in empty_indices:
            with_empty.append(empty_value)
        else:
            with_empty.append(inputs[index])
            index += 1
    return with_empty


def get_default_dict(d: NestedDict, default: float) -> NestedDict:
    flat_dict = flatten_nested_dict(d)
    flat_default_dict = {key: default for key in flat_dict.keys()}
    default_dict = unflatten_dict(flat_default_dict)
    return default_dict


def check_for_single_texts(texts_list: List[List[TextType]]) -> List[TextType]:
    single_texts = []
    for texts in texts_list:
        if len(texts) != 1:
            raise Exception(f"Found {len(texts)} texts. Expected 1.")
        single_texts.append(texts[0])
    return single_texts
