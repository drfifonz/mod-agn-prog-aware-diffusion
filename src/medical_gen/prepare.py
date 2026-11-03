"""Freeze explicit patient-disjoint real splits once, before comparing strategies."""
import argparse
import csv
from dataclasses import replace
import json
import logging
from pathlib import Path
from .datasets.records import ImageRecord, load_real_manifest, split_by_patient, validate_partitions, write_manifest
from .datasets.sampling import subsample_distribution

NIH_CLASSES=['No_Finding','Effusion','Pneumonia','Pneumothorax']


def prepare_nih_records(metadata: Path, root: Path) -> list[ImageRecord]:
    records=[]
    with metadata.open(newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'Image Index','Finding Labels','Patient ID'}.issubset(reader.fieldnames or []):
            raise ValueError('NIH metadata requires Image Index, Finding Labels and Patient ID')
        for row in reader:
            labels=set(row['Finding Labels'].split('|'))
            selected=labels & {'No Finding','Effusion','Pneumonia','Pneumothorax'}
            if len(selected)!=1 or ('No Finding' in selected and labels!={'No Finding'}):continue
            label=next(iter(selected)).replace(' ','_')
            candidates=[root/row['Image Index'],root/label/row['Image Index']]
            path=next((p.resolve() for p in candidates if p.is_file()),None)
            if path is None:continue
            records.append(ImageRecord(path,label,row['Patient ID'].strip(),''))
    if not records:raise ValueError('No eligible NIH images found')
    validate_partitions(records)
    return records


def prepare_protocol(manifest: Path,root: Path,classes: list[str],output: Path,seed: int,
                     train_fraction: float=.8,val_fraction: float=.1,
                     test_distribution: dict[str,float] | None=None) -> list[ImageRecord]:
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'Refusing to overwrite a frozen split manifest: {output}')
    rows=load_real_manifest(manifest,root,classes)
    if not any(r.split for r in rows):rows=split_by_patient(rows,seed,train_fraction,val_fraction)
    if {r.split for r in rows}!={'train','val','test'}:
        raise ValueError('All three patient-disjoint real subsets must be nonempty')
    if test_distribution is not None:
        test=subsample_distribution([r for r in rows if r.split=='test'],classes,test_distribution,seed)
        rows=[r for r in rows if r.split!='test']+test
    validate_partitions(rows)
    write_manifest(rows,output)
    from hashlib import sha256
    output.with_suffix('.json').write_text(json.dumps({'seed':seed,'source_manifest_sha256':sha256(manifest.read_bytes()).hexdigest(),
        'classes':classes,'test_distribution':test_distribution,'patient_disjoint':True,
        'note':'Freeze once and reuse unchanged across all training strategies.'},indent=2)+'\n')
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path)
    source.add_argument('--nih-metadata',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--classes',nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-fraction',type=float,default=.8)
    parser.add_argument('--val-fraction',type=float,default=.1)
    parser.add_argument('--test-distribution',type=Path,help='JSON class-to-weight mapping; applied only to the real test subset')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    root=args.root.expanduser().resolve();out=args.output.expanduser().resolve()
    classes=args.classes or NIH_CLASSES
    if len(classes)!=4 or len(set(classes))!=4:parser.error('Provide exactly four distinct classes')
    manifest=args.manifest
    if args.nih_metadata:
        if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(out)
        if set(classes)!=set(NIH_CLASSES):parser.error('NIH preparation requires the four selected ChestX classes')
        rows=prepare_nih_records(args.nih_metadata.expanduser(),root)
        raw=out.with_suffix('.input.csv')
        if raw.exists():raise FileExistsError(raw)
        write_manifest(rows,raw);manifest=raw
    distribution=json.loads(args.test_distribution.read_text()) if args.test_distribution else None
    rows=prepare_protocol(manifest.expanduser(),root,classes,out,args.seed,args.train_fraction,args.val_fraction,distribution)
    logging.info('Wrote %s (%d real image records)',out,len(rows))

if __name__=='__main__':
    main()
