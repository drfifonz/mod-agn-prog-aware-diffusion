from collections import Counter
from pathlib import Path
import csv
import pytest
from medical_gen.datasets.records import ImageRecord, load_real_manifest, validate_partitions, split_by_patient, generated_pool
from medical_gen.datasets.sampling import select_training, subsample_distribution
from medical_gen.metrics import classification_metrics, aggregate_runs

CLASSES = ['a', 'b', 'c', 'd']

def real_records(counts=(5, 3, 2, 1)):
    return [ImageRecord(Path(f'{c}-{i}.png'), c, f'p-{c}-{i}', 'train') for c,n in zip(CLASSES, counts) for i in range(n)]

def pool(n=12):
    return {c: [ImageRecord(Path(f'syn-{c}-{i}.png'), c, '', 'train', 'synthetic') for i in range(n)] for c in CLASSES}

def test_balancing_adds_only_deficits_and_never_majority():
    selected=select_training(real_records(),pool(),CLASSES,'mixed',True,42,9)
    assert Counter(r.label for r in selected)==dict.fromkeys(CLASSES,5)
    assert Counter(r.label for r in selected if r.source=='synthetic')=={'b':2,'c':3,'d':4}
    assert len({r.path for r in selected})==len(selected)

def test_equal_additions_keep_original_real_records():
    real=real_records();selected=select_training(real,pool(),CLASSES,'mixed',False,42,4)
    assert selected[:len(real)]==real
    assert Counter(r.label for r in selected if r.source=='synthetic')==dict.fromkeys(CLASSES,4)
    assert select_training(real,pool(),CLASSES,'mixed',False,42,4)==selected
    assert select_training(real,pool(),CLASSES,'mixed',False,43,4)!=selected

def test_shortage_fails_without_duplicates():
    with pytest.raises(ValueError,match='d.*need 4.*available 2'):
        select_training(real_records(),pool(2),CLASSES,'mixed',True,42,4)
    with pytest.raises(ValueError,match='Insufficient'):
        select_training(real_records(),pool(2),CLASSES,'mixed',False,42,4)

def test_already_balanced_requires_no_synthetics():
    real=real_records((2,2,2,2))
    assert select_training(real,{},CLASSES,'mixed',True,42,4)==real

def test_missing_real_class_can_be_filled():
    result=select_training(real_records((3,2,1,0)),pool(3),CLASSES,'mixed',True,42,3)
    assert Counter(r.label for r in result)==dict.fromkeys(CLASSES,3)

def test_independent_real_and_synthetic_sampling():
    real=real_records((70,70,70,70));dist=dict(zip(CLASSES,[.7,.15,.1,.05]))
    selected=subsample_distribution(real,CLASSES,dist,42)
    assert Counter(r.label for r in selected)=={'a':70,'b':15,'c':10,'d':5}
    assert select_training(real,{},CLASSES,'real',False,42,4)==real
    result=select_training(real,pool(100),CLASSES,'synthetic',True,42,100,dist)
    assert all(r.source=='synthetic' for r in result)

def test_grid_files_excluded(tmp_path):
    for cls in CLASSES:
        d=tmp_path/cls;d.mkdir()
        for name in ['sample_000001.png','sample_000002.png','grid_000.png','final_summary_grid.png','notes.txt']:(d/name).write_text('test')
    p=generated_pool(tmp_path,CLASSES)
    assert all(len(v)==2 for v in p.values())
    assert all(r.path.name.startswith('sample_') for rows in p.values() for r in rows)

def test_patient_and_path_leakage_rejected():
    rows=[ImageRecord(Path('a.png'),'a','p','train'),ImageRecord(Path('b.png'),'a','p','test')]
    with pytest.raises(ValueError,match='patient'):validate_partitions(rows)
    rows=[ImageRecord(Path('a.png'),'a','p1','train'),ImageRecord(Path('a.png'),'a','p2','test')]
    with pytest.raises(ValueError,match='image'):validate_partitions(rows)

def test_manifest_rejects_missing_patient_and_unknown_labels(tmp_path):
    (tmp_path/'a.png').write_text('image')
    path=tmp_path/'real.csv'
    path.write_text('path,label,patient_id,split\na.png,a,,train\n')
    with pytest.raises(ValueError,match='patient'):load_real_manifest(path,tmp_path,CLASSES)
    path.write_text('path,label,patient_id,split\na.png,unknown,p1,train\n')
    with pytest.raises(ValueError,match='class'):load_real_manifest(path,tmp_path,CLASSES)

def test_split_is_patient_grouped_and_deterministic():
    rows=[ImageRecord(Path(f'{i}-{j}.png'),'a',f'p{i}','') for i in range(20) for j in range(2)]
    first=split_by_patient(rows,42,.8,.1)
    assert first==split_by_patient(rows,42,.8,.1)
    assert {r.split for r in first}=={'train','val','test'}
    validate_partitions(first)

def test_global_macro_f1_and_batch_independence():
    true=[0,0,0,1,1,2];pred=[0,1,0,1,2,2]
    result=classification_metrics(true,pred,CLASSES)
    expected=(4/5+2/4+2/3+0)/4
    assert result['macro_f1']==pytest.approx(expected)
    assert result['per_class']['d']['f1']==0
    pairs=list(zip(true,pred));chunks=[pairs[:1],pairs[1:4],pairs[4:]]
    flattened=[p for batch in chunks for p in batch]
    assert classification_metrics([p[0] for p in flattened],[p[1] for p in flattened],CLASSES)==result
    assert aggregate_runs([.2,.4])['sample_sd']==pytest.approx(2**.5*.1)

def test_invalid_metric_input():
    for true,pred in [([],[]),([0],[]),([0],[4])]:
        with pytest.raises(ValueError):classification_metrics(true,pred,CLASSES)
