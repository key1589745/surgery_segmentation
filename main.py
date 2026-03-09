
from datetime import datetime
import argparse,warnings
from torch.backends import cudnn
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
OmegaConf.register_new_resolver("eval", eval)
import torch
warnings.filterwarnings("ignore")


def main(configs, overrides=None):
    
    # load dataset    
    with initialize(version_base=None, config_path=configs):
        if overrides is None:
            overrides = []
        cfg = compose(config_name='experiments', overrides=overrides)
        OmegaConf.resolve(cfg)
        runner = instantiate(cfg.runner, _recursive_=True)


    runner.train()
    runner.evaluate()
    runner.save_model()



    print('Train finished: ', datetime.now().strftime("%m_%d_%Y_%H:%M:%S"))


if __name__ == '__main__':
    
    # set parameters
    parser = argparse.ArgumentParser()

        # dataset param
    parser.add_argument('--args', type=str, default='cfgs')
    parser.add_argument('--cuda', type=int, default=0)

    CONFIGs, unknown_args = parser.parse_known_args()
    torch.cuda.set_device(CONFIGs.cuda)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = CONFIGs.cuda
    cudnn.benchmark = True

    # Pass unknown args as overrides to Hydra
    main(CONFIGs.args, overrides=unknown_args)
