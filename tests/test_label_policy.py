from __future__ import annotations

import unittest

import numpy as np

from src.labeling.components import connected_components
from src.labeling.policy import (
    FUNCTIONAL_STORAGE_VALUE,
    GRASP_STORAGE_VALUE,
    IGNORE_VALUE,
    convert_affgrasp_mask,
    convert_umd_mask,
)


class LabelPolicyTests(unittest.TestCase):
    def test_affgrasp_mapping(self) -> None:
        source = np.array([[0, 128, 255]], dtype=np.uint8)
        converted, summary = convert_affgrasp_mask(source)
        np.testing.assert_array_equal(converted, [[0, GRASP_STORAGE_VALUE, FUNCTIONAL_STORAGE_VALUE]])
        self.assertEqual(summary.unknown_source_values, ())

    def test_umd_mapping_and_contain_ignore(self) -> None:
        source = np.arange(8, dtype=np.uint8).reshape(2, 4)
        converted, summary = convert_umd_mask(source)
        expected = np.array(
            [
                [0, GRASP_STORAGE_VALUE, FUNCTIONAL_STORAGE_VALUE, FUNCTIONAL_STORAGE_VALUE],
                [IGNORE_VALUE, FUNCTIONAL_STORAGE_VALUE, FUNCTIONAL_STORAGE_VALUE, GRASP_STORAGE_VALUE],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(converted, expected)
        self.assertIn("umd_contain_policy", summary.ignore_reasons)

    def test_unknown_values_become_ignore(self) -> None:
        converted, summary = convert_umd_mask(np.array([[8]], dtype=np.uint8))
        self.assertEqual(int(converted[0, 0]), IGNORE_VALUE)
        self.assertEqual(summary.unknown_source_values, (8,))

    def test_disconnected_candidates_are_preserved(self) -> None:
        source = np.zeros((5, 6), dtype=np.uint8)
        source[0:2, 0:2] = GRASP_STORAGE_VALUE
        source[3:5, 4:6] = GRASP_STORAGE_VALUE
        instances, components = connected_components(source)
        self.assertEqual(len(components), 2)
        self.assertEqual({component.model_class_id for component in components}, {0})
        self.assertEqual(set(np.unique(instances)), {0, 1, 2})
        self.assertTrue(all(component.touches_boundary for component in components))


if __name__ == "__main__":
    unittest.main()

