"""Predict one membrane panel with a released native HiFT bundle."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from osn_hift import NativeHiFT


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['pure_solvent_permeance','solution_permeance'],required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--anchors',help='JSON file mapping anchor solvent to measured log10 permeance')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    anchors=json.loads(Path(a.anchors).read_text(encoding='utf-8')) if a.anchors else {}
    pred=NativeHiFT(ROOT/'models'/a.task).predict_membrane(pd.read_csv(a.input,low_memory=False),anchors)
    pred.to_csv(a.output,index=False)
    print(pred.to_string(index=False))


if __name__=='__main__':main()
