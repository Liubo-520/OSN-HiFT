import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from reference_identity import canonical_reference, normalize_references


class ReferenceIntegrity(unittest.TestCase):
    def test_doi_aliases_merge(self):
        forms = ["https://doi.org/10.1016/J.MEMSCI.2012.08.017",
                 "http://dx.doi.org/10.1016/j.memsci.2012.08.017",
                 " doi:10.1016/j.memsci.2012.08.017. "]
        self.assertEqual({canonical_reference(s) for s in forms},
                         {"10.1016/j.memsci.2012.08.017"})

    def test_doi_parentheses_are_not_truncated(self):
        self.assertEqual(canonical_reference("https://doi.org/10.1016/S0376-7388(00)00001-2"),
                         "10.1016/s0376-7388(00)00001-2")

    def test_missing_does_not_become_one_invented_paper(self):
        for value in [None, float("nan"), "", " "]:
            with self.assertRaises(ValueError):
                canonical_reference(value)

    def test_retains_measurements_and_raw_identity(self):
        df = pd.DataFrame({"report.reference": ["https://doi.org/10.1234/ABC", "10.1234/abc"],
                           "name": ["M1", "M1"], "value": [1.0, 2.0]})
        out = normalize_references(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["report.reference"].nunique(), 1)
        self.assertEqual(out["report.reference_raw"].tolist(), df["report.reference"].tolist())
        self.assertEqual(out["value"].tolist(), [1.0, 2.0])

    def test_canonical_groups_and_external_overlap(self):
        raw = ["https://doi.org/10.1234/ABC", "10.1234/abc", "10.1234/b", "10.1234/c"]
        groups = [canonical_reference(s) for s in raw]
        for tr, te in GroupKFold(3).split(raw, groups=groups):
            self.assertFalse(set(groups[i] for i in tr) & set(groups[i] for i in te))
        self.assertEqual(set(map(canonical_reference, [raw[0]])) - {groups[1]}, set())


if __name__ == "__main__":
    unittest.main()
