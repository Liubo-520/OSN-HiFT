"""Guard the inner base learner's complete training boundary, without fitting ML."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
spec = importlib.util.spec_from_file_location("independent", ROOT / "analysis/17_independent_validation.py")
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


class NestedIsolation(unittest.TestCase):
    def test_every_inner_fit_excludes_validation_papers(self):
        ref = independent.REF
        train = pd.DataFrame({ref: np.repeat([f"paper{i}" for i in range(10)], 2),
                              "row_id": np.arange(20)})
        valid = pd.DataFrame({ref: ["held_out", "held_out"], "row_id": [20, 21]})
        fits = []

        def predict(dtr, dte, seed, extra):
            self.assertFalse(set(dtr[ref]) & set(dte[ref]))
            self.assertNotIn("held_out", set(dtr[ref]))
            self.assertTrue(set(dtr.row_id) <= set(train.row_id))
            fits.append((set(dtr.row_id), set(dte.row_id)))
            return dte.row_id.to_numpy(float)

        with patch.object(independent.fl, "gbm_fit_predict", side_effect=predict):
            oof, pred = independent.nested_base_predictions(train, valid, 0, "pure_solvent_permeance")
        self.assertEqual(len(fits), 6)
        self.assertEqual(fits[0][0], set(train.row_id))
        self.assertEqual(fits[0][1], set(valid.row_id))
        np.testing.assert_array_equal(oof, train.row_id)
        np.testing.assert_array_equal(pred, valid.row_id)

    def test_overlap_is_rejected_before_any_fit(self):
        d = pd.DataFrame({independent.REF: ["same_paper"]})
        with patch.object(independent.fl, "gbm_fit_predict") as fit:
            with self.assertRaises(ValueError):
                independent.nested_base_predictions(d, d, 0, "pure_solvent_permeance")
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
