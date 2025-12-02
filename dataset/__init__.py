from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

def get_loaders(config_file):
    """
    Get DataLoaders based on the configuration file path.
    """
    with initialize(version_base=None, config_path=config_file):
        cfg = compose(config_name='dataset_CHO')
        OmegaConf.resolve(cfg)
        dataloaders = instantiate(cfg.data_loaders, _recursive_=True)

    return dataloaders
