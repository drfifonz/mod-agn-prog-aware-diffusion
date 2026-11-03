"""Recompute global metrics from saved predictions or aggregate compatible repeats."""
import argparse
import csv
import json
from pathlib import Path
from .metrics import classification_metrics,aggregate_runs


def aggregate_metric_files(paths: list[Path]) -> dict:
    runs=[json.loads(p.read_text()) for p in paths]
    if not runs:raise ValueError('No run results provided')
    keys=['class_names','real_manifest_sha256','test_selection_sha256','training_selection_sha256','training_protocol_sha256']
    for key in keys:
        if any(key not in run for run in runs):raise ValueError(f'Missing run metadata: {key}')
        if any(run[key]!=runs[0][key] for run in runs):raise ValueError(f'Incompatible runs: {key}')
    seeds=[r.get('seed') for r in runs]
    if None in seeds or len(seeds)!=len(set(seeds)):raise ValueError('Repeated runs require distinct explicit seeds')
    return {'macro_f1':aggregate_runs([r['macro_f1'] for r in runs]),
            'accuracy':aggregate_runs([r['accuracy'] for r in runs]),
            'seeds':seeds,'test_selection_sha256':runs[0]['test_selection_sha256']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--predictions',type=Path)
    source.add_argument('--runs',type=Path,nargs='+')
    parser.add_argument('--classes',nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.predictions:
        if not args.classes:parser.error('--predictions requires --classes in training class-index order')
        with args.predictions.open(newline='') as stream:rows=list(csv.DictReader(stream))
        result=classification_metrics([int(r['y_true']) for r in rows],[int(r['y_pred']) for r in rows],args.classes)
    else:result=aggregate_metric_files(args.runs)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')

if __name__=='__main__':main()
