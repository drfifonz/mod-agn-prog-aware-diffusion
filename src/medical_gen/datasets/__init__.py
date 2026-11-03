"""Manifest-based data entry point, with no bundled patient data."""
def build_datamodule(data_cfg,enable_validation):
    from .manifest import ManifestDataModule,ManifestDatasetConfig
    if data_cfg.dataset.name!='manifest':
        raise ValueError('The public protocol requires dataset.name: manifest')
    return ManifestDataModule(ManifestDatasetConfig.model_validate(data_cfg.dataset.params),
        batch_size=data_cfg.batch_size,num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,enable_validation=enable_validation)
