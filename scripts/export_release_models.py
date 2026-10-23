"""Fit full-snapshot deployment models and verify native-file prediction parity."""
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
import fewshot_lib as fl
import transfer_eval as te
from osn_core import RAW,REF,S1,featurize,transfer_design
from osn_hift import NativeHiFT


def main():
    reports=[]
    for tag in ['pure_solvent_permeance','solution_permeance']:
        print('Fit deployment bundle:',tag,flush=True)
        d=fl.load_domain(tag); aux=fl.load_domain(fl.AUX[tag]); seed=0
        extra=fl.extra_for(tag); X,vocab=featurize(d,extra_numeric=extra)
        model=lgb.LGBMRegressor(random_state=seed,**fl.LGB).fit(X,d.y)
        f_oof=np.zeros(len(d))
        for tr,te_idx in GroupKFold(5,shuffle=True,random_state=100).split(d,groups=d[REF]):
            f_oof[te_idx]=fl.gbm_fit_predict(d.iloc[tr],d.iloc[te_idx],seed,extra)
        center,scale=transfer_design(sorted(d[S1].unique()))[1:]
        comp=te.fit_components(d,f_oof,d.iloc[:0],aux,seed,tag,center,scale)
        out=ROOT/'models'/tag; out.mkdir(parents=True,exist_ok=True)
        for name,booster in [('zero_shot',model.booster_),('within_membrane',comp['wmc'].model.booster_),
                             ('pairwise',comp['pgbm'].booster_)]:
            (out/f'{name}.txt').write_text(booster.model_to_string(),encoding='utf-8')
        meta=dict(task=tag,version='1.0.0',training_snapshot='2025-04-08',seed=0,rows=len(d),
            canonical_papers=d[REF].nunique(),auxiliary_rows=len(aux),vocabulary=vocab,
            zero_shot_features=list(X.columns),membrane_features=list(comp['memX'].columns),
            sigma2=comp['s2'],tau2_multi=comp['t2'],ensemble_weight=0.5,prior_shrinkage=False,
            target='log10 permeance in L m^-2 h^-1 bar^-1',lightgbm_version=lgb.__version__,
            training_raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
            descriptor_sha256=hashlib.sha256((ROOT/'data/reference/solvent_descriptors_verified.csv').read_bytes()).hexdigest(),
            limitation='Full-snapshot deployment fit; benchmark results are separate held-out predictions. Raw SD is not calibrated coverage.')
        (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        native=NativeHiFT(out)
        ext=pd.read_csv(ROOT/'analysis/data'/f'{tag}_external_new.csv',low_memory=False)
        saved=pd.read_csv(ROOT/'analysis/results'/f'fewshot_external_{tag}.csv',low_memory=False)
        saved=saved[(saved.method=='HiFT')&(saved.seed==0)]
        # Two complete external membrane panels exercise k=0,1,2,3 where present.
        chosen=ext.groupby('entity')[S1].nunique().sort_values(ascending=False).head(2).index
        errors=[]; sd_errors=[]
        for entity in chosen:
            panel=ext[ext.entity==entity].copy(); y=panel.groupby(S1).logP.median()
            sample=saved[saved.entity==entity].groupby('k',group_keys=False).head(12)
            for row in sample.itertuples():
                names=[] if pd.isna(row.anchors) else str(row.anchors).split('|')
                observed={s:float(y[s]) for s in names}
                pred=native.predict_membrane(panel,observed).set_index('solvent').loc[row.target]
                errors.append(abs(pred.hift_log10_P-row.pred)); sd_errors.append(abs(pred.raw_sd_log10_P-row.sd))
        passed=bool(max(errors)<1e-9 and max(sd_errors)<1e-9)
        report=dict(task=tag,passed=passed,parity_rows=len(errors),maximum_prediction_error=max(errors),maximum_sd_error=max(sd_errors),
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
        reports.append(report); print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
        if not passed:raise RuntimeError('Native model parity failed')
    (ROOT/'reports/model_export_validation.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),models=reports),indent=2),encoding='utf-8')


if __name__=='__main__':main()
