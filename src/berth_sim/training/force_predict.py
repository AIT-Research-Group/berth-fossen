"""Forecast six nonnegative load statistics from a historical load CSV."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .force_model import TARGETS
from .train import load_model, predict
from .evaluation import write_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=100)
    args=parser.parse_args()
    if args.samples<2:raise ValueError('Use at least 2 samples')
    torch.set_num_threads(4)
    model,checkpoint=load_model(args.checkpoint)
    if checkpoint['config'].get('task')!='loads':raise ValueError('A load checkpoint is required')
    with args.input.open(encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))[-checkpoint['config']['history_steps']:]
    values=np.array([[float(row[key]) for key in TARGETS] for row in rows],dtype=np.float32)
    if len(values)!=checkpoint['config']['history_steps'] or not np.isfinite(values).all():
        raise ValueError('Complete finite load history required')
    if np.any(values<0) or np.any(values[:,3:]<values[:,:3]):raise ValueError('Require absmax >= RMS >= 0')
    if not np.allclose(np.diff([float(row['time_s']) for row in rows]),60):raise ValueError('Require consecutive one-minute observations')
    ensemble=predict(model,values[None],args.samples)[0]
    low,high=np.quantile(ensemble,[.05,.95],axis=0)
    mean=ensemble.mean(0)
    result=[]
    for k in range(len(mean)):
        row={'horizon_min':k+1}
        for j,key in enumerate(TARGETS):
            row.update({f'{key}_mean':float(mean[k,j]),f'{key}_q05':float(low[k,j]),f'{key}_q95':float(high[k,j])})
        result.append(row)
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'forecast.csv',result)
    np.savez_compressed(args.output/'forecast.npz',history=values,ensemble=ensemble)
    (args.output/'metadata.json').write_text(json.dumps({'task':'loads','synthetic_only':True,
        'targets':TARGETS,'origin_s':float(rows[-1]['time_s']),'checkpoint':str(args.checkpoint)},indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output)}))


if __name__=='__main__':main()
