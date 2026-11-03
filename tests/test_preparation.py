from pathlib import Path
import csv,json
from PIL import Image
import pytest
from medical_gen.protocol_config import ClassificationConfig
from medical_gen.prepare import prepare_protocol, prepare_nih_records
from medical_gen.datasets.protocol import build_selection

CLASSES=['reference','benign','fluid','precancerous']

def fixture_manifest(tmp_path):
    rows=[]
    for ci,c in enumerate(CLASSES):
        for split,n in [('train',ci+1),('val',1),('test',1)]:
            for i in range(n):
                name=f'{c}-{split}-{i}.png';Image.new('L',(16,16),ci*35+i).save(tmp_path/name)
                rows.append([name,c,f'{c}-{split}-{i}',split])
    p=tmp_path/'real.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','patient_id','split']);w.writerows(rows)
    synthetic=tmp_path/'generated'
    for c in CLASSES:
        (synthetic/c).mkdir(parents=True)
        for i in range(5):Image.new('L',(16,16),i).save(synthetic/c/f'sample_{i:06}.png')
    return p,synthetic

def config(tmp_path,mode='mixed',stratified=True):
    p,syn=fixture_manifest(tmp_path)
    return ClassificationConfig.model_validate({'data':{'manifest':str(p),'root':str(tmp_path),'classes':CLASSES,'modality':'oct','synthetic_root':str(syn),'mode':mode,'stratified':stratified,'synthetic_per_class':2}})

def test_shared_real_validation_and_test_across_strategies(tmp_path):
    cfg=config(tmp_path)
    selections=[]
    for mode in ['real','synthetic','mixed']:
        updated=cfg.model_copy(deep=True);updated.data.mode=mode;updated.data.stratified=False
        selections.append(build_selection(updated.data))
    assert selections[0].val==selections[1].val==selections[2].val
    assert selections[0].test==selections[1].test==selections[2].test
    selected=build_selection(cfg.data)
    assert len(selected.train)==16
    assert all(r.source=='real' for r in selected.val+selected.test)

def test_prepared_manifest_preserves_existing_splits_and_rejects_overwrite(tmp_path):
    p,_=fixture_manifest(tmp_path);out=tmp_path/'prepared.csv'
    prepare_protocol(p,tmp_path,CLASSES,out,42)
    original=list(csv.DictReader(p.open()));prepared=list(csv.DictReader(out.open()))
    assert [r['patient_id'] for r in original]==[r['patient_id'] for r in prepared]
    assert [r['split'] for r in original]==[r['split'] for r in prepared]
    with pytest.raises(FileExistsError):prepare_protocol(p,tmp_path,CLASSES,out,42)

def test_insufficient_pool_is_preflight_failure(tmp_path):
    cfg=config(tmp_path);cfg.data.synthetic_per_class=100;cfg.data.stratified=False
    with pytest.raises(ValueError,match='Insufficient'):build_selection(cfg.data)

def test_explicit_nih_single_label_policy(tmp_path):
    meta=tmp_path/'nih.csv';meta.write_text('Image Index,Finding Labels,Patient ID\na.png,No Finding,1\nb.png,Effusion|Atelectasis,2\nc.png,Effusion|Pneumonia,3\n')
    for name in ['a.png','b.png','c.png']:Image.new('L',(8,8)).save(tmp_path/name)
    rows=prepare_nih_records(meta,tmp_path)
    assert [r.label for r in rows]==['No_Finding','Effusion']
    assert [r.patient_id for r in rows]==['1','2']

def test_invalid_configuration_fails():
    with pytest.raises(ValueError):ClassificationConfig.model_validate({'data':{'manifest':'a','root':'.','classes':['a','a'],'modality':'oct'}})

def test_real_preflight_exports_identical_test_for_other_modes(tmp_path):
    import subprocess,sys
    cfg=config(tmp_path)
    paths=[]
    for mode in ['real','synthetic','mixed']:
        current=cfg.model_copy(deep=True);current.data.mode=mode;current.data.stratified=False
        current.output_root=tmp_path/'outputs';current.experiment_name=mode
        path=tmp_path/f'{mode}.yaml'
        import yaml
        path.write_text(yaml.safe_dump(json.loads(current.model_dump_json())))
        result=subprocess.run([sys.executable,'-m','medical_gen.classify','--config',str(path),'--prepare-only'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        paths.append(current.output_root/mode/'selection/test.csv')
    assert paths[0].read_bytes()==paths[1].read_bytes()==paths[2].read_bytes()
