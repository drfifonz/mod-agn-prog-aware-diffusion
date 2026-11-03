import json,subprocess,sys
from pathlib import Path
import pytest
from medical_gen.evaluate import aggregate_metric_files

def test_aggregation_rejects_changed_test_or_duplicate_seed(tmp_path):
    first={'macro_f1':.2,'accuracy':.4,'class_names':['a','b','c','d'],'real_manifest_sha256':'m','test_selection_sha256':'t','training_selection_sha256':'train','training_protocol_sha256':'protocol','seed':1}
    paths=[]
    for i in range(2):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps({**first,'seed':i}));paths.append(p)
    assert aggregate_metric_files(paths)['macro_f1']['n']==2
    paths[1].write_text(json.dumps({**first,'seed':0}))
    with pytest.raises(ValueError,match='distinct'):aggregate_metric_files(paths)
    paths[1].write_text(json.dumps({**first,'test_selection_sha256':'other'}))
    with pytest.raises(ValueError,match='Incompatible'):aggregate_metric_files(paths)

@pytest.mark.parametrize('module',['prepare','train','generate','classify','evaluate'])
def test_cli_help_without_training_dependencies(module):
    result=subprocess.run([sys.executable,'-m',f'medical_gen.{module}','--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout


def test_aggregation_rejects_different_training_protocol(tmp_path):
    base={'macro_f1':.2,'accuracy':.4,'class_names':['a','b','c','d'],'real_manifest_sha256':'m','test_selection_sha256':'t',
          'training_selection_sha256':'train','training_protocol_sha256':'protocol'}
    paths=[]
    for i in range(2):
        p=tmp_path/f'run{i}.json';p.write_text(json.dumps({**base,'seed':i}));paths.append(p)
    paths[1].write_text(json.dumps({**base,'seed':1,'training_protocol_sha256':'different'}))
    with pytest.raises(ValueError,match='training_protocol'):aggregate_metric_files(paths)
