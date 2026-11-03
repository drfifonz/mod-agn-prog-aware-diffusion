"""Portable generator training; no scheduler, remote execution or cluster paths."""
import json
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger,WandbLogger
import torch
from .config import ExperimentConfig,ModelName
from .datasets import build_datamodule
from .models import build_generator
from .datasets.records import write_manifest


def run_training(config_path: Path,resume: Path | None=None) -> None:
    cfg=ExperimentConfig.load(config_path,env_prefixes=('DDPM_',))
    output=Path(cfg.paths.output_root).expanduser()/cfg.paths.experiment_name
    if output.exists() and resume is None:raise FileExistsError(f'Refusing to overwrite generator run: {output}')
    if resume is not None:
        resume=resume.expanduser().resolve()
        if not resume.is_file():raise FileNotFoundError(resume)
    if resume and cfg.paths.init_checkpoint:raise ValueError('Use either init_checkpoint or resume, not both')
    if cfg.validation.enabled and not cfg.validation.fid_enabled:
        raise ValueError('Generator checkpoint selection requires FID when validation is enabled')
    pl.seed_everything(cfg.seed,workers=True)
    data=build_datamodule(cfg.data,enable_validation=cfg.validation.enabled)
    data.prepare_data();data.setup('fit')
    output.mkdir(parents=True,exist_ok=True)
    for split in ['train','val']:
        records=[r for r in data.records if r.split==split and r.label==data.config.class_name]
        write_manifest(records,output/'selection'/f'{split}.csv')
    (output/'config.json').write_text(cfg.model_dump_json(indent=2)+'\n')
    model=build_generator(cfg.model,cfg.sampling,cfg.validation,output/'samples')
    model._real_images_in_unit_interval=not data.config.standardize
    if cfg.paths.init_checkpoint:
        path=Path(cfg.paths.init_checkpoint).expanduser()
        if not path.is_file():raise FileNotFoundError(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        # New model instance and trainer: optimizer state is intentionally not inherited.
    logger=CSVLogger(str(output),name='logs')
    if cfg.wandb.enabled:
        logger=WandbLogger(project=cfg.wandb.project,entity=cfg.wandb.entity,name=cfg.wandb.run_name or cfg.paths.experiment_name,save_dir=str(output))
    checkpoint=ModelCheckpoint(dirpath=output/'checkpoints',filename='best',monitor='fid' if cfg.validation.enabled else None,
        mode='min',save_top_k=1,save_last=True,enable_version_counter=False)
    precision=cfg.trainer.precision
    if precision=='auto':precision='16-mixed' if torch.cuda.is_available() else '32-true'
    settings=dict(max_epochs=cfg.trainer.max_epochs,accelerator=cfg.trainer.accelerator,devices=cfg.trainer.devices,
        precision=precision,logger=logger,callbacks=[checkpoint],num_sanity_val_steps=0,
        default_root_dir=str(output),log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        accumulate_grad_batches=cfg.trainer.gradient_accumulate_every)
    is_diffusion=cfg.model.name in {ModelName.DDPM,ModelName.DDPM_V2,ModelName.GAUSSIAN_DIFFUSION}
    if is_diffusion:
        if cfg.trainer.max_steps is None or cfg.trainer.max_steps<=0:raise ValueError('Diffusion requires positive trainer.max_steps')
        settings['max_steps']=cfg.trainer.max_steps
        data.set_train_steps_per_epoch(cfg.trainer.max_steps)
        if cfg.validation.enabled:
            settings['val_check_interval']=cfg.validation.every_n_steps
            settings['check_val_every_n_epoch']=None
    elif cfg.validation.enabled:settings['check_val_every_n_epoch']=cfg.validation.every_n_epochs or 1
    trainer=pl.Trainer(**settings)
    trainer.fit(model,datamodule=data,ckpt_path=str(resume) if resume else None)
    trainer.save_checkpoint(output/'checkpoints/final.ckpt')
