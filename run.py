import yaml
import argparse
from easydict import EasyDict as edict
from scripts.atari.train_dqn import Trainer as Trainer_DQN
from scripts.atari.train_double_dqn import Trainer as Trainer_DoubleDQN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type = str, default = None, help = 'Environment Name')
    parser.add_argument('--method', type = str, default = None, help = 'Method Name')
    parser.add_argument('--cfg', type = str, default = None, help = 'Path to Configuration File')
    args = parser.parse_args()

    if args.cfg is not None:
        with open(args.cfg, 'r') as cfg_file:
            cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)
    else:
        cfgs = {}
    cfgs = edict(cfgs)
    cfgs.environment = args.env_name
    print(cfgs.environment)

    if args.method == 'DQN':
        Trainer = Trainer_DQN
    elif args.method == 'DoubleDQN':
        Trainer = Trainer_DoubleDQN
    
    trainer = Trainer(cfgs)
    trainer.training()
