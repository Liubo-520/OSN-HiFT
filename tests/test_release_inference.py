"""Release input guards and measured-target exclusion, using native model files."""
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from osn_hift import NativeHiFT


@unittest.skipUnless((ROOT/'models/pure_solvent_permeance/metadata.json').exists(),'Requires exported native models')
class ReleaseInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=NativeHiFT(ROOT/'models/pure_solvent_permeance')

    def panel(self):
        return pd.DataFrame({'name':['new membrane']*3,'report.reference':['new-study']*3,
                             'testConditions.solvent1':['Methanol','Toluene','Acetone']})

    def test_rejects_explicit_solvent_mixture(self):
        d=self.panel();d['testConditions.solventMixture']=True
        with self.assertRaises(ValueError):self.model.prepare_rows(d)

    def test_rejects_second_solvent_even_without_flag(self):
        d=self.panel();d['testConditions.solvent2']='water'
        with self.assertRaises(ValueError):self.model.prepare_rows(d)

    def test_entity_cannot_hide_multiple_names_or_papers(self):
        for field in ['name','report.reference']:
            d=self.panel();d['entity']='copied constant';d.loc[1,field]='other'
            with self.assertRaises(ValueError):self.model.prepare_rows(d)

    def test_target_observation_excluded_from_own_prediction(self):
        d=self.panel()
        a=self.model.predict_membrane(d,{'Methanol':0.0,'Toluene':0.2}).set_index('solvent')
        b=self.model.predict_membrane(d,{'Methanol':4.0,'Toluene':0.2}).set_index('solvent')
        self.assertAlmostEqual(a.loc['Methanol','hift_log10_P'],b.loc['Methanol','hift_log10_P'],places=13)
        self.assertEqual(a.loc['Methanol','anchors'],1)
