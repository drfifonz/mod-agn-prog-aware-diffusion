"""Prepare one training selection and shared real held-out sets without torch."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from .records import ImageRecord, load_real_manifest, generated_pool, validate_partitions, write_manifest
from .sampling import select_training

@dataclass
class Selection:
    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]
    manifest_sha256: str

    def export(self,directory: Path) -> dict:
        directory.mkdir(parents=True,exist_ok=True)
        summary={'real_manifest_sha256':self.manifest_sha256}
        for name in ['train','val','test']:
            rows=getattr(self,name)
            write_manifest(rows,directory/f'{name}.csv')
            summary[name]={'total':len(rows),'by_class':dict(Counter(r.label for r in rows)),
                           'by_source':dict(Counter(r.source for r in rows)),
                           'selection_sha256':sha256(json.dumps([(str(r.path),r.label,r.patient_id,r.source) for r in rows]).encode()).hexdigest()}
        (directory/'selection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary


def build_selection(config) -> Selection:
    records=load_real_manifest(config.manifest,config.root,config.classes)
    if {r.split for r in records}!={'train','val','test'}:
        raise ValueError('Prepared real manifest must have nonempty train, val and test sets')
    real_train=[r for r in records if r.split=='train']
    synthetic=generated_pool(config.synthetic_root,config.classes) if config.mode!='real' else {}
    train=select_training(real_train,synthetic,config.classes,config.mode,config.stratified,
                          config.selection_seed,config.synthetic_per_class,config.target_distribution)
    if not train:
        raise ValueError('Empty training selection')
    val=[r for r in records if r.split=='val'];test=[r for r in records if r.split=='test']
    validate_partitions(train+val+test)
    return Selection(train,val,test,sha256(config.manifest.read_bytes()).hexdigest())
