import sys
import os
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Get the grandparent directory path
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the grandparent directory to sys.path
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)


def build_model(config_file):
    with initialize(version_base=None, config_path=config_file):
        cfg = compose(config_name='model')
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True).cuda()
    return model

