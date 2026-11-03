"""Class-specific SMOTE-style interpolation in flattened image space."""
from pathlib import Path
from typing import Any
import pytorch_lightning as pl
import torch
from .mixins import FIDValidationMixin,SampleVisualizationMixin
from ..utils import normalize_hw

class SMOTEGenerator(FIDValidationMixin,SampleVisualizationMixin,pl.LightningModule):
    """Uniform interpolation between a training image and one of its k neighbors.

    Unlike a multiclass oversampler, this generator operates on one already
    selected class. It never treats other classes as potential neighbors.
    """
    def __init__(self,sample_size=(256,512),save_dir='samples',smote_k_neighbors=5,
                 fid_num_samples=256,fid_enabled=True,fid_low_vram_mode=False,
                 fid_clear_cuda_cache=True,use_cpu_for_fid=False,
                 sampling_num_images=16,sampling_grid_nrow=4,**_: Any):
        pl.LightningModule.__init__(self)
        self.automatic_optimization=False
        self.fid_enabled=fid_enabled
        if fid_enabled:
            FIDValidationMixin.__init__(self,fid_num_samples=fid_num_samples,
                low_vram_mode=fid_low_vram_mode,clear_cuda_cache=fid_clear_cuda_cache,
                use_cpu_for_fid=use_cpu_for_fid)
        self.sample_size=normalize_hw(sample_size,field_name='model.sample_size')
        self.save_dir=Path(save_dir);self.smote_k_neighbors=smote_k_neighbors
        self.sampling_num_images=sampling_num_images;self.sampling_grid_nrow=sampling_grid_nrow
        self._pool=None;self._neighbors=None

    def build_interpolation_pool(self,images: torch.Tensor) -> None:
        images=images.detach().float().cpu()
        if len(images)<2:raise ValueError('Within-class interpolation requires at least two real training images')
        if self.smote_k_neighbors<1:raise ValueError('smote_k_neighbors must be positive')
        if not torch.isfinite(images).all():raise ValueError('Non-finite interpolation input')
        if images.min()<0 or images.max()>1:raise ValueError('Interpolation inputs must be scaled to [0,1]')
        self._pool=images
        flat=images.flatten(1);k=min(self.smote_k_neighbors,len(images)-1)
        neighbors=[]
        for start in range(0,len(images),128):
            distances=torch.cdist(flat[start:start+128],flat)
            for local in range(len(distances)):distances[local,start+local]=float('inf')
            neighbors.append(distances.topk(k,largest=False).indices)
        self._neighbors=torch.cat(neighbors)

    def setup(self,stage=None):
        if stage not in (None,'fit') or self._pool is not None:return
        dataset=self.trainer.datamodule._train_dataset
        self.build_interpolation_pool(torch.stack([dataset[i][0] for i in range(len(dataset))]))

    def configure_optimizers(self):return []
    def training_step(self,batch,batch_idx):return None

    def sample(self,num_samples=16):
        if self._pool is None or self._neighbors is None:raise RuntimeError('Interpolation pool is not initialized')
        indices=torch.randint(len(self._pool),(num_samples,))
        neighbor_slots=torch.randint(self._neighbors.shape[1],(num_samples,))
        neighbor_ids=self._neighbors[indices,neighbor_slots]
        lam=torch.rand(num_samples,1,1,1)
        return (self._pool[indices]+lam*(self._pool[neighbor_ids]-self._pool[indices])).clamp(0,1)

    def on_validation_epoch_start(self):
        if self.fid_enabled:FIDValidationMixin.on_validation_epoch_start(self)
    def validation_step(self,batch,batch_idx):
        if self.fid_enabled:FIDValidationMixin.validation_step(self,batch,batch_idx)
    def on_validation_epoch_end(self):
        if self.fid_enabled:FIDValidationMixin.on_validation_epoch_end(self)
    def generate_samples_for_fid(self,num_samples):return self.sample(num_samples)
    def on_train_end(self):
        self._save_samples_grid(self.sample(self.sampling_num_images),step=None,prefix='smote_samples')

    def state_dict(self,*args,**kwargs):
        state=super().state_dict(*args,**kwargs)
        state['_interpolation_pool']=self._pool;state['_interpolation_neighbors']=self._neighbors
        return state

    def load_state_dict(self,state_dict,strict=True):
        state=dict(state_dict)
        self._pool=state.pop('_interpolation_pool',None)
        self._neighbors=state.pop('_interpolation_neighbors',None)
        return super().load_state_dict(state,strict=strict)
