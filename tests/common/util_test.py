import unittest

from repro.common import util


class TestUtil(unittest.TestCase):
    def test_flatten(self):
        assert util.flatten("text") == "text"
        assert util.flatten("text", separator="-") == "text"
        assert util.flatten(["1", "2"]) == "1 2"
        assert util.flatten(["1", "2"], separator="-") == "1-2"

    def test_group_by_references(self):
        # Each `references` can either be
        #   (1) a List[str] of length 1 for a single reference
        #   (2) a List[List[str]] of length 1 for a single reference
        #   (3) a List[str] of length >1 for multiple references
        #   (4) a List[List[str]] of length >2 for multiple references
        # This method tests to ensure duplicates of each type are correctly
        # identified.
        candidates = ["1", "2", "3", "4"]

        # All unique
        references_list = [
            ["A"],
            [["B1", "B2"]],
            ["C", "D"],
            ["E", ["F1", "F2"]],
        ]
        (
            grouped_candidates_list,
            grouped_references_list,
            mapping,
        ) = util.group_by_references(candidates, references_list)
        assert grouped_candidates_list == [["1"], ["2"], ["3"], ["4"]]
        assert grouped_references_list == references_list
        assert mapping == [(0, 0), (1, 0), (2, 0), (3, 0)]

        # Duplicates which are single-reference strings
        references_list = [
            ["A"],
            [["B1", "B2"]],
            ["A"],
            ["E", ["F1", "F2"]],
        ]
        (
            grouped_candidates_list,
            grouped_references_list,
            mapping,
        ) = util.group_by_references(candidates, references_list)
        assert grouped_candidates_list == [["1", "3"], ["2"], ["4"]]
        assert grouped_references_list == [["A"], [["B1", "B2"]], ["E", ["F1", "F2"]]]
        assert mapping == [(0, 0), (1, 0), (0, 1), (2, 0)]

        # Duplicates which are single-reference List[str]. Index 2
        # should not be the same as 1 and 3 because it is 2 references, not 1.
        references_list = [
            ["A"],
            [["B1", "B2"]],
            ["B1", "B2"],
            [["B1", "B2"]],
        ]
        (
            grouped_candidates_list,
            grouped_references_list,
            mapping,
        ) = util.group_by_references(candidates, references_list)
        assert grouped_candidates_list == [["1"], ["2", "4"], ["3"]]
        assert grouped_references_list == [["A"], [["B1", "B2"]], ["B1", "B2"]]
        assert mapping == [(0, 0), (1, 0), (2, 0), (1, 1)]

        # Duplicates which are multi-ref, each reference is a str
        references_list = [
            ["A", "B"],
            [["C1", "C2"]],
            ["A", "B"],
            ["D", ["E1", "E2"]],
        ]
        (
            grouped_candidates_list,
            grouped_references_list,
            mapping,
        ) = util.group_by_references(candidates, references_list)
        assert grouped_candidates_list == [["1", "3"], ["2"], ["4"]]
        assert grouped_references_list == [
            ["A", "B"],
            [["C1", "C2"]],
            ["D", ["E1", "E2"]],
        ]
        assert mapping == [(0, 0), (1, 0), (0, 1), (2, 0)]

        # Duplicates which are multi-ref, each reference is a List[str]
        references_list = [
            ["A"],
            [["B1", "B2"], ["C1"]],
            [["B1", "B2"], ["C1"]],
            ["D", ["E1", "E2"]],
        ]
        (
            grouped_candidates_list,
            grouped_references_list,
            mapping,
        ) = util.group_by_references(candidates, references_list)
        assert grouped_candidates_list == [["1"], ["2", "3"], ["4"]]
        assert grouped_references_list == [
            ["A"],
            [["B1", "B2"], ["C1"]],
            ["D", ["E1", "E2"]],
        ]
        assert mapping == [(0, 0), (1, 0), (1, 1), (2, 0)]

        # Duplicates which are multi-ref, each reference is a different type
        references_list = [
            ["A"],
            ["B", ["C1", "C2"]],
            [["D1", "D2"], ["D1"]],
            ["B", ["C1", "C2"]],
        ]
        (
            grouped_candidates_list,
            grouped_references_list,
            mapping,
        ) = util.group_by_references(candidates, references_list)
        assert grouped_candidates_list == [["1"], ["2", "4"], ["3"]]
        assert grouped_references_list == [
            ["A"],
            ["B", ["C1", "C2"]],
            [["D1", "D2"], ["D1"]],
        ]
        assert mapping == [(0, 0), (1, 0), (2, 0), (1, 1)]

    def test_ungroup_values(self):
        values_list = [[1, 2], [3], [4, 5]]
        mapping = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
        assert util.ungroup_values(values_list, mapping) == [1, 2, 3, 4, 5]

        mapping = [(1, 0), (0, 0), (0, 1), (2, 1), (2, 0)]
        assert util.ungroup_values(values_list, mapping) == [3, 1, 2, 5, 4]
