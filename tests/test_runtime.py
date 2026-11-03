"""CPU smoke checks. Install the training extra to run these tests."""
import importlib.util
import json
from pathlib import Path
import pytest
if importlib.util.find_spec('torch') is None:
    pytest.skip('Requires the optional training extra',allow_module_level=True)
import torch
from PIL import Image


def test_runtime_imports_and_all_example_configs():
    from medical_gen.config import ExperimentConfig
    from medical_gen.generation_runtime import GenerationConfig
    from medical_gen.protocol_config import ClassificationConfig
    from medical_gen.models import MODEL_REGISTRY
    root=Path(__file__).resolve().parents[1]
    for p in (root/'configs/train').rglob('*.yaml'):ExperimentConfig.load(p,env_prefixes=())
    for p in (root/'configs/generate').glob('*.yaml'):GenerationConfig.load(p)
    for p in (root/'configs/classify').glob('*.yaml'):ClassificationConfig.load(p)
    assert len(MODEL_REGISTRY)>=3


def test_smote_single_class_really_interpolates(monkeypatch):
    from medical_gen.models.smote import SMOTEGenerator
    from medical_gen.models.mixins import FIDValidationMixin
    monkeypatch.setattr(FIDValidationMixin,'__init__',lambda *args,**kwargs:None)
    model=SMOTEGenerator(sample_size=(8,8),fid_enabled=False)
    images=torch.stack([torch.zeros(1,8,8),torch.ones(1,8,8),torch.full((1,8,8),.5)])
    model.build_interpolation_pool(images)
    torch.manual_seed(42)
    generated=model.sample(8)
    assert generated.shape==(8,1,8,8)
    assert any(not any(torch.equal(generated[i],source) for source in images) for i in range(8))
    state=model.state_dict();other=SMOTEGenerator(sample_size=(8,8),fid_enabled=False)
    other.load_state_dict(state)
    assert other.sample(2).shape==(2,1,8,8)


def test_resnet_training_and_global_evaluation(tmp_path):
    import csv
    from medical_gen.protocol_config import ClassificationConfig
    from medical_gen.datasets.protocol import build_selection
    from medical_gen.classification_runtime import run_repeats
    torch.set_num_threads(1)
    classes=['a','b','c','d'];manifest=tmp_path/'real.csv'
    rows=[]
    for ci,c in enumerate(classes):
        for split in ['train','val','test']:
            path=tmp_path/f'{c}-{split}.png';Image.new('L',(32,32),ci*60).save(path)
            rows.append([str(path),c,f'{c}-{split}',split])
    with manifest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','patient_id','split']);w.writerows(rows)
    cfg=ClassificationConfig.model_validate({'repeats':1,'data':{'manifest':manifest,'root':tmp_path,
        'classes':classes,'modality':'chestx','mode':'real','batch_size':4,'image_size':[32,32]},
        'model':{'pretrained':False},'trainer':{'max_epochs':1,'accelerator':'cpu','devices':1}})
    selection=build_selection(cfg.data);output=tmp_path/'run';summary=selection.export(output/'selection')
    run_repeats(cfg,selection,summary,output)
    metric=json.loads((output/'repeat-00/metrics.json').read_text())
    assert metric['n_test']==4
    assert 0<=metric['macro_f1']<=1
    assert Path(metric['best_checkpoint']).is_file()
    assert len(list(csv.DictReader((output/'repeat-00/test_predictions.csv').open())))==4
    assert json.loads((output/'summary.json').read_text())['macro_f1']['n']==1

def test_generator_rectangular_forward_sampling_and_checkpoint(tmp_path,monkeypatch):
    from medical_gen.models.mixins import FIDValidationMixin
    from medical_gen.models.gausian_diffusion import GaussianDiffusionGenerator
    from medical_gen.generation_runtime import ModelGenerationConfig,load_model_from_checkpoint
    monkeypatch.setattr(FIDValidationMixin,'__init__',lambda *args,**kwargs:None)
    torch.set_num_threads(1)
    kwargs={'num_timesteps':4,'sampling_timesteps':2,'feature_maps':8,'dim_mults':[1,2],
            'use_medicraft_settings':False,'objective':'pred_noise'}
    model=GaussianDiffusionGenerator(sample_size=(16,32),**kwargs)
    assert torch.isfinite(model(torch.rand(2,1,16,32)))
    generated=model.sample(2)
    assert generated.shape==(2,1,16,32)
    assert generated.min()>=0 and generated.max()<=1
    checkpoint=tmp_path/'model.ckpt';torch.save({'state_dict':model.state_dict()},checkpoint)
    cfg=ModelGenerationConfig(name='example',model_type='gaussian_diffusion_generator',checkpoint_path=str(checkpoint),
        sample_size=(16,32),model_kwargs=kwargs)
    restored=load_model_from_checkpoint(cfg,torch.device('cpu'))
    assert restored.sample(1).shape==(1,1,16,32)

def test_fid_real_input_scale_matches_generated_input():
    from types import SimpleNamespace
    from medical_gen.models.mixins import FIDValidationMixin
    images=torch.full((2,1,8,8),.2)
    probe=SimpleNamespace(fid_num_samples=2,_fid_real_count=0,_real_features=[],
        _real_images_in_unit_interval=True,_fid_low_vram_mode=False,
        _prepare_for_fid=FIDValidationMixin._prepare_for_fid,
        _move_to_fid_device=lambda x:x,_extract_features=lambda x:x,
        _log=lambda message:None)
    FIDValidationMixin._update_fid_real(probe,images)
    expected=FIDValidationMixin._prepare_for_fid(images,already_normalized=True)
    assert torch.equal(probe._real_features[0],expected)


def test_generation_failure_is_not_success(tmp_path):
    from medical_gen.generation_runtime import GenerationConfig,run_generation
    cfg=GenerationConfig.model_validate({'output_root':str(tmp_path/'missing'),'device':'cpu',
        'save_grids':False,'models':[{'name':'a','model_type':'gaussian_diffusion_generator',
        'checkpoint_path':str(tmp_path/'absent.ckpt'),'num_images':2,'batch_size':2}]})
    with pytest.raises(RuntimeError,match='Incomplete'):run_generation(cfg)
    assert (tmp_path/'missing/generation_summary.yaml').is_file()


def test_partial_generation_is_failure_and_grid_false_is_honored(tmp_path,monkeypatch):
    import medical_gen.generation_runtime as runtime
    monkeypatch.setattr(runtime,'load_model_from_checkpoint',lambda *args:torch.nn.Identity())
    calls=[]
    def batch(*args):
        calls.append(True)
        if len(calls)==2:raise RuntimeError('deliberate batch failure')
        return torch.zeros(2,1,8,8)
    monkeypatch.setattr(runtime,'generate_batch',batch)
    cfg=runtime.GenerationConfig.model_validate({'output_root':str(tmp_path/'partial'),'device':'cpu',
        'save_grids':False,'models':[{'name':'a','model_type':'gaussian_diffusion_generator',
        'checkpoint_path':'unused.ckpt','num_images':4,'batch_size':2}]})
    with pytest.raises(RuntimeError,match='Incomplete'):runtime.run_generation(cfg)
    assert not list((tmp_path/'partial').rglob('*grid*.png'))

def test_tiny_generator_training_checkpoint_and_generation(tmp_path,monkeypatch):
    import csv,yaml
    from torch import nn
    import medical_gen.models.mixins as mixins
    from medical_gen.training_runtime import run_training
    from medical_gen.generation_runtime import GenerationConfig,run_generation
    class TinyInception(nn.Module):
        def forward(self,images):return images.mean(dim=(-1,-2))
    monkeypatch.setattr(mixins,'inception_v3',lambda **kwargs:TinyInception())
    torch.set_num_threads(1)
    manifest=tmp_path/'real.csv';rows=[]
    for c in ['a','b','c','d']:
        for split in ['train','val','test']:
            for i in range(2):
                p=tmp_path/f'{c}-{split}-{i}.png';Image.new('L',(16,32),40+70*i).save(p)
                rows.append([str(p),c,f'{c}-{split}-{i}',split])
    with manifest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','patient_id','split']);w.writerows(rows)
    model_kwargs={'num_timesteps':4,'sampling_timesteps':2,'feature_maps':8,'dim_mults':[1,2],
                  'use_medicraft_settings':False,'objective':'pred_noise'}
    raw={'paths':{'output_root':str(tmp_path/'training'),'experiment_name':'tiny'},
         'data':{'dataset':{'name':'manifest','params':{'manifest':str(manifest),'root':str(tmp_path),
          'classes':['a','b','c','d'],'class_name':'a','modality':'chestx','image_size':[16,32]}},'batch_size':2,'num_workers':0},
         'model':{'name':'gaussian_diffusion_generator','sample_size':[16,32],**model_kwargs},
         'trainer':{'max_steps':2,'max_epochs':-1,'precision':'32-true','accelerator':'cpu','devices':1,'log_every_n_steps':1},
         'validation':{'enabled':True,'every_n_steps':2,'every_n_epochs':None,'fid_num_samples':2},
         'sampling':{'every_n_steps':0,'num_samples':2},'wandb':{'enabled':False}}
    config=tmp_path/'train.yaml';config.write_text(yaml.safe_dump(raw));run_training(config)
    checkpoint=tmp_path/'training/tiny/checkpoints/best.ckpt';assert checkpoint.is_file()
    generation=GenerationConfig.model_validate({'output_root':str(tmp_path/'generated'),'device':'cpu',
         'save_grids':False,'models':[{'name':'a','model_type':'gaussian_diffusion_generator','checkpoint_path':str(checkpoint),
         'num_images':2,'batch_size':2,'sample_size':[16,32],'model_kwargs':model_kwargs}]})
    summary=run_generation(generation)
    assert summary['successful_models']==1
    assert len(list((tmp_path/'generated/a').glob('sample_*.png')))==2
