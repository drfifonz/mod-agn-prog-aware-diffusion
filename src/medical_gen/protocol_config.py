"""Validated public classification configuration, separate from generator settings."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

class StrictConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

class ClassificationDataConfig(StrictConfig):
    manifest: Path
    root: Path = Path('.')
    classes: list[str]
    modality: Literal['oct','chestx']
    synthetic_root: Path | None = None
    mode: Literal['real','synthetic','mixed'] = 'mixed'
    stratified: bool = False
    synthetic_per_class: int = Field(default=4000,ge=0)
    target_distribution: dict[str,float] | None = None
    selection_seed: int = 42
    image_size: tuple[int,int] = (256,256)
    batch_size: int = Field(default=32,ge=1)
    num_workers: int = Field(default=0,ge=0)

    @model_validator(mode='after')
    def validate_policy(self):
        if len(self.classes)!=4 or len(set(self.classes))!=4:
            raise ValueError('Exactly four distinct class names are required')
        if any(not c.strip() or '/' in c or '\\' in c or c in {'.','..'} for c in self.classes):
            raise ValueError('Class names must be nonempty directory names')
        if len(self.image_size)!=2 or min(self.image_size)<8:
            raise ValueError('image_size must contain two positive dimensions >= 8')
        if self.mode!='real' and self.synthetic_root is None:
            raise ValueError('Synthetic or mixed training requires synthetic_root')
        if self.stratified and self.mode!='mixed' and self.target_distribution is None:
            raise ValueError('Non-mixed stratified sampling requires explicit target_distribution')
        for name in ['manifest','root','synthetic_root']:
            value=getattr(self,name)
            if value is not None:setattr(self,name,value.expanduser().resolve())
        return self

class ClassifierModelConfig(StrictConfig):
    pretrained: bool = True
    learning_rate: float = Field(default=1e-4,gt=0)

class ClassifierTrainerConfig(StrictConfig):
    max_epochs: int = Field(default=15,ge=1)
    precision: str = '32-true'
    accelerator: str = 'auto'
    devices: int = Field(default=1,ge=1,le=1)
    patience: int = Field(default=5,ge=1)

class ClassificationConfig(StrictConfig):
    data: ClassificationDataConfig
    model: ClassifierModelConfig = Field(default_factory=ClassifierModelConfig)
    trainer: ClassifierTrainerConfig = Field(default_factory=ClassifierTrainerConfig)
    output_root: Path = Path('outputs/classification')
    experiment_name: str = 'mixed'
    seed: int = 42
    repeats: int = Field(default=10,ge=1)

    @classmethod
    def load(cls,path: Path):
        return cls.model_validate(yaml.safe_load(path.expanduser().read_text()))
