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

    def test_flatten_nested_dict(self):
        flattened = util.flatten_nested_dict({})
        assert flattened == {}

        flattened = util.flatten_nested_dict({"A": 1})
        assert flattened == {("A",): 1}

        flattened = util.flatten_nested_dict(
            {"A": 1, "B": {"C": 2, "D": 3, "E": {"F": 4}}}
        )
        assert flattened == {
            ("A",): 1,
            ("B", "C"): 2,
            ("B", "D"): 3,
            ("B", "E", "F"): 4,
        }

    def test_unflatten_dict(self):
        unflattened = util.unflatten_dict({})
        assert unflattened == {}

        unflattened = util.unflatten_dict({("A",): 1})
        assert unflattened == {"A": 1}

        unflattened = util.unflatten_dict(
            {("A",): 1, ("B", "C"): 2, ("B", "D"): 3, ("B", "E", "F"): 4}
        )
        assert unflattened == {"A": 1, "B": {"C": 2, "D": 3, "E": {"F": 4}}}

    def test_average_dicts(self):
        assert util.average_dicts([]) == {}

        dicts = [{"A": 1, "B": {"C": 2, "D": 3, "E": {"F": 4}}}]
        assert util.average_dicts(dicts) == dicts[0]

        dicts = [
            {"A": 1, "B": {"C": 2, "D": 3, "E": {"F": 4}}},
            {"A": 5, "B": {"C": 6, "D": 7, "E": {"F": 8}}},
        ]
        assert util.average_dicts(dicts) == {
            "A": 3,
            "B": {"C": 4, "D": 5, "E": {"F": 6}},
        }

        with self.assertRaises(Exception):
            # They don't have the same keys
            dicts = [
                {"A": 1, "B": {"C": 2, "D": 3, "E": 4}},
                {"A": 5, "B": {"C": 6, "D": 7, "E": {"F": 8}}},
            ]
            util.average_dicts(dicts)

        with self.assertRaises(Exception):
            # They don't have the same keys
            dicts = [
                {"B": {"C": 2, "D": 3, "E": {"F": 4}}},
                {"A": 5, "B": {"C": 6, "D": 7, "E": {"F": 8}}},
            ]
            util.average_dicts(dicts)

        with self.assertRaises(Exception):
            # They don't have the same keys
            dicts = [
                {"A": 1, "B": {"C": 2, "E": {"F": 4}}},
                {"A": 5, "B": {"C": 6, "D": 7, "E": {"F": 8}}},
            ]
            util.average_dicts(dicts)

    def test_is_empty_text(self):
        assert util.is_empty_text("") is True
        assert util.is_empty_text("  ") is True
        assert util.is_empty_text(" \n ") is True

        assert util.is_empty_text([]) is True
        assert util.is_empty_text([""]) is True
        assert util.is_empty_text(["", ""]) is True
        assert util.is_empty_text(["", "  "]) is True

        assert util.is_empty_text("A") is False
        assert util.is_empty_text(["A"]) is False
        assert util.is_empty_text(["", "A"]) is False

    def test_remove_empty_inputs(self):
        inputs = ["A", "", "B", "", "C"]
        context1 = ["D1", "D2", "D3", "D4", "D5"]
        context2 = ["E1", "E2", "E3", "E4", "E5"]
        expected_empty_indices = {1, 3}
        expected_non_empty_inputs = ["A", "B", "C"]
        expected_non_empty_context1 = ["D1", "D3", "D5"]
        expected_non_empty_context2 = ["E1", "E3", "E5"]

        empty_indices, non_empty_inputs = util.remove_empty_inputs(inputs)
        assert empty_indices == expected_empty_indices
        assert non_empty_inputs == expected_non_empty_inputs

        empty_indices, non_empty_inputs, non_empty_context1 = util.remove_empty_inputs(
            inputs, context1
        )
        assert empty_indices == expected_empty_indices
        assert non_empty_inputs == expected_non_empty_inputs
        assert non_empty_context1 == expected_non_empty_context1

        (
            empty_indices,
            non_empty_inputs,
            non_empty_context1,
            non_empty_context2,
        ) = util.remove_empty_inputs(inputs, context1, context2)
        assert empty_indices == expected_empty_indices
        assert non_empty_inputs == expected_non_empty_inputs
        assert non_empty_context1 == expected_non_empty_context1
        assert non_empty_context2 == expected_non_empty_context2

        # Test for different length contexts
        with self.assertRaises(Exception):
            util.remove_empty_inputs(["A"], [])
        with self.assertRaises(Exception):
            util.remove_empty_inputs(["A"], ["B", "C"])
        with self.assertRaises(Exception):
            util.remove_empty_inputs(["A"], ["B"], [])
