"""Heavy training dependencies live at the CLI edge, outside sampling utilities."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from .datasets.images import RecordDataset
from .models.classifier import ResNetClassifier
from .metrics import aggregate_runs


def run_repeats(cfg,selection,summary,output: Path) -> None:
    def loader(records,training=False):
        dataset=RecordDataset(records,cfg.data.image_size,cfg.data.modality,augment=training,standardize=True)
        dataset.class_to_index={c:i for i,c in enumerate(cfg.data.classes)}
        return DataLoader(dataset,batch_size=cfg.data.batch_size,num_workers=cfg.data.num_workers,shuffle=training)
    protocol_sha256=sha256(json.dumps({"data":cfg.data.model_dump(mode="json"),
        "model":cfg.model.model_dump(mode="json"),"trainer":cfg.trainer.model_dump(mode="json")},sort_keys=True).encode()).hexdigest()
    runs=[]
    for repeat in range(cfg.repeats):
        seed=cfg.seed+repeat;pl.seed_everything(seed,workers=True)
        directory=output/f'repeat-{repeat:02d}';directory.mkdir()
        checkpoint=ModelCheckpoint(dirpath=directory/'checkpoints',filename='best',monitor='val_loss',mode='min',save_top_k=1)
        model=ResNetClassifier(cfg.data.classes,**cfg.model.model_dump())
        trainer=pl.Trainer(max_epochs=cfg.trainer.max_epochs,accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,precision=cfg.trainer.precision,
            callbacks=[checkpoint,EarlyStopping(monitor='val_loss',patience=cfg.trainer.patience)],
            logger=CSVLogger(str(directory),name='logs'),num_sanity_val_steps=0,log_every_n_steps=1,
            default_root_dir=directory,enable_progress_bar=False)
        trainer.fit(model,train_dataloaders=loader(selection.train,True),val_dataloaders=loader(selection.val))
        if not checkpoint.best_model_path:raise RuntimeError('No best classifier checkpoint was saved')
        trainer.test(model,dataloaders=loader(selection.test),ckpt_path=checkpoint.best_model_path)
        metrics={**model.test_metrics,'seed':seed,'repeat':repeat,'best_checkpoint':str(checkpoint.best_model_path),
                 'real_manifest_sha256':selection.manifest_sha256,'test_selection_sha256':summary['test']['selection_sha256'],
                 'training_selection_sha256':summary['train']['selection_sha256'],'training_protocol_sha256':protocol_sha256}
        (directory/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        with (directory/'test_predictions.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['path','y_true','y_pred'])
            for r,actual,pred in zip(selection.test,model.test_true,model.test_pred,strict=True):writer.writerow([str(r.path),actual,pred])
        runs.append(metrics)
    (output/'summary.json').write_text(json.dumps({'macro_f1':aggregate_runs([r['macro_f1'] for r in runs]),
        'accuracy':aggregate_runs([r['accuracy'] for r in runs]),'test_selection_sha256':summary['test']['selection_sha256'],
        'seeds':[r['seed'] for r in runs],'metric':'Global test macro-F1; equal class weights; zero denominator gives zero.'},indent=2)+'\n')
