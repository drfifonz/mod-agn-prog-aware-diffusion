"""Generator data module for private OCT or public ChestX via the same split schema."""
from pathlib import Path
from typing import Literal
from .base import BaseDatasetConfig,BaseDataModule
from .records import load_real_manifest
from .images import RecordDataset

class ManifestDatasetConfig(BaseDatasetConfig):
    manifest: Path
    root: Path = Path('.')
    classes: list[str]
    class_name: str
    modality: Literal['oct','chestx']
    image_size: tuple[int,int]=(256,256)
    standardize: bool=False

class ManifestDataModule(BaseDataModule[ManifestDatasetConfig]):
    def prepare_data(self):
        if self.config.class_name not in self.config.classes:
            raise ValueError('class_name must be included in classes')
        self.records=load_real_manifest(self.config.manifest,self.config.root,self.config.classes)
        if {r.split for r in self.records}!={'train','val','test'}:
            raise ValueError('Generator requires the same prepared train/val/test manifest as classification')

    def setup(self,stage=None):
        if not hasattr(self,'records'):self.prepare_data()
        for split,attribute in [('train','_train_dataset'),('val','_val_dataset')]:
            records=[r for r in self.records if r.split==split and r.label==self.config.class_name]
            if not records:raise ValueError(f'No {split} images for {self.config.class_name}')
            setattr(self,attribute,RecordDataset(records,self.config.image_size,self.config.modality,
                augment=split=='train',standardize=self.config.standardize))

    def train_dataloader(self):
        return self._make_dataloader(self._train_dataset,shuffle=True,infinite=self._train_steps_per_epoch is not None)

    def val_dataloader(self):
        return self._make_dataloader(self._val_dataset,shuffle=False)
