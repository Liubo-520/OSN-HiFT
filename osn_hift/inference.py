"""Load text/JSON models without executable pickle artifacts."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import fewshot_lib as fl
from osn_core import MULTI_LABEL, SINGLE_CAT, NUMERIC, REF, S1, SOLV, SOLVENT_ALIASES, featurize
from reference_identity import canonical_reference


class NativeHiFT:
    def __init__(self, directory):
        directory=Path(directory)
        self.meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        self.vocab=self.meta['vocabulary']
        self.models={name:lgb.Booster(model_str=(directory/f'{name}.txt').read_text(encoding='utf-8'))
                     for name in ['zero_shot','within_membrane','pairwise']}

    def prepare_rows(self, rows):
        d=rows.copy()
        if 'name' not in d or S1 not in d:raise ValueError('Input needs name and testConditions.solvent1')
        mixture='testConditions.solventMixture'
        if mixture in d and not d[mixture].fillna(False).astype(str).str.strip().str.lower().isin(['false','0','0.0','']).all():
            raise ValueError('Solvent mixtures are outside the training domain')
        second='testConditions.solvent2'
        if second in d and (d[second].notna() & d[second].astype(str).str.strip().ne('')).any():
            raise ValueError('Second-solvent mixtures are outside the training domain')
        if REF not in d:d[REF]='prediction-input'
        d[REF]=d[REF].map(canonical_reference)
        if d['name'].isna().any() or d['name'].nunique()!=1 or d[REF].nunique()!=1:
            raise ValueError('Predict one membrane name and canonical study at a time')
        expected=d[REF].astype(str)+' || '+d['name'].astype(str)
        if 'entity' in d and not d['entity'].eq(expected).all():
            raise ValueError('Supplied entity disagrees with canonical study and membrane name')
        d['entity']=expected
        d[S1]=d[S1].replace(SOLVENT_ALIASES)
        unknown=set(d[S1])-set(SOLV.index)
        if unknown:raise ValueError(f'Unsupported or ambiguous solvent: {sorted(unknown)}')
        for col in MULTI_LABEL+SINGLE_CAT:
            if col not in d:d[col]=None
        for col in NUMERIC:
            if col not in d:d[col]=np.nan
            d[col]=pd.to_numeric(d[col],errors='coerce')
        if 'log_solute_MW' not in d:
            mw=pd.to_numeric(d.get('soluteChoice.solute1.molecularWeight',pd.Series(np.nan,index=d.index)),errors='coerce')
            d['log_solute_MW']=np.log10(mw.where(mw>0))
        return d.reset_index(drop=True)

    def predict_membrane(self, rows, anchors=None):
        """Rows contain all candidate/anchor solvents; anchors map solvent to measured log10 permeance.

        A measured target is excluded from its own anchor set. With zero anchors,
        the HiFT prior level depends on this declared candidate-solvent panel.
        Raw SD is not a calibrated interval or a coverage guarantee.
        """
        d=self.prepare_rows(rows); anchors={} if anchors is None else dict(anchors)
        if len(anchors)>3:raise ValueError('This release supports at most three measured anchor solvents')
        if set(anchors)-set(d[S1]):raise ValueError('Every anchor solvent must appear in the input panel')
        if not all(np.isfinite(float(v)) for v in anchors.values()):raise ValueError('Anchor measurements must be finite log10 values')
        X,_=featurize(d,vocab=self.vocab,extra_numeric=fl.extra_for(self.meta['task']))
        X=X.reindex(columns=self.meta['zero_shot_features'])
        f=self.models['zero_shot'].predict(X,num_threads=1)
        cells=d.assign(f=f).groupby(['entity',S1],as_index=False).agg(f=('f','median'))
        cells['source']='target'
        memX=fl.membrane_features(d,self.vocab).reindex(columns=self.meta['membrane_features'])
        g=self.models['within_membrane'].predict(fl.design(cells,memX),num_threads=1)
        cells['g']=g; prior=float(np.mean(cells.f-g)); lookup=cells.set_index(S1)
        s2=self.meta['sigma2']; t2=self.meta['tau2_multi']; result=[]
        for row in cells.itertuples(index=False,name=None):
            entity,solvent,f_target,source,g_target=row
            known={s:float(v) for s,v in anchors.items() if s!=solvent}
            k=len(known)
            if k:
                names=list(known); ya=np.asarray(list(known.values()))
                M=np.repeat(memX.loc[[entity]].to_numpy(float),k,axis=0)
                pair=self.models['pairwise'].predict(fl.pair_X(np.repeat(solvent,k),np.asarray(names),ya,M),num_threads=1)
                pred=float(np.mean(ya+0.5*(g_target-lookup.loc[names,'g'].to_numpy())+0.5*pair))
                sd=float(np.sqrt(s2/k+s2))
            else:pred=prior+g_target; sd=float(np.sqrt(t2+s2))
            result.append(dict(solvent=solvent,anchors=k,zero_shot_log10_P=float(f_target),
                               hift_log10_P=pred,hift_P=float(10**pred),raw_sd_log10_P=sd))
        return pd.DataFrame(result)
