
from datetime import datetime
import argparse,warnings
from torch.backends import cudnn
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
OmegaConf.register_new_resolver("eval", eval)
import torch
warnings.filterwarnings("ignore")


def main(configs):
    
    # load dataset    
    with initialize(version_base=None, config_path=configs):
        cfg = compose(config_name='experiments')
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

    CONFIGs = parser.parse_args()
    torch.cuda.set_device(CONFIGs.cuda)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = CONFIGs.cuda
    cudnn.benchmark = True

    main(CONFIGs.args)
