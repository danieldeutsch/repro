from typing import Dict, List, T, Tuple, Union

from repro.data.types import TextType

Indexable = Union[List[T], Dict[int, T]]


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
