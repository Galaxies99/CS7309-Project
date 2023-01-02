import yaml
import argparse
from easydict import EasyDict as edict
from utils.configs import generate_paths_in_cfgs
from scripts.discrete.test_dqn import Tester as Tester_DQN
from scripts.discrete.train_dqn import Trainer as Trainer_DQN
from scripts.discrete.test_double_dqn import Tester as Tester_DoubleDQN
from scripts.discrete.train_double_dqn import Trainer as Trainer_DoubleDQN
from scripts.discrete.test_dueling_dqn import Tester as Tester_DuelingDQN
from scripts.discrete.train_dueling_dqn import Trainer as Trainer_DuelingDQN
from scripts.continuous.train_ddpg import Trainer as Trainer_DDPG
from scripts.continuous.train_td3 import Trainer as Trainer_TD3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type = str, default = None, help = 'Environment Name')
    parser.add_argument('--method', type = str, default = None, help = 'Method Name')
    parser.add_argument('--cfg', type = str, default = None, help = 'Path to Configuration File')
    parser.add_argument('--test', type = int, default = 0, help = 'Number of Episode in Testing')
    args = parser.parse_args()

    if args.cfg is not None:
        with open(args.cfg, 'r') as cfg_file:
            cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)
    else:
        cfgs = {}
    cfgs = edict(cfgs)
    cfgs.environment = args.env_name
    cfgs = generate_paths_in_cfgs(cfgs, args.env_name, args.method)
    if args.test == 0:
        if args.method == 'DQN':
            Trainer = Trainer_DQN
        elif args.method == 'DoubleDQN':
            Trainer = Trainer_DoubleDQN
        elif args.method == 'DuelingDQN':
            Trainer = Trainer_DuelingDQN
        elif args.method == 'DDPG':
            Trainer = Trainer_DDPG
        elif args.method == 'TD3':
            Trainer = Trainer_TD3
        
        trainer = Trainer(cfgs)
        trainer.training()
    else:
        print('Testing ...')
        if args.method == 'DQN':
            Tester = Tester_DQN
        elif args.method == 'DoubleDQN':
            Tester = Tester_DoubleDQN
        elif args.method == 'DuelingDQN':
            Tester = Tester_DuelingDQN
        tester = Tester(cfgs)
        res = tester.testing(args.test)
        print('Mean: {}, Min: {}, Max: {}'.format(res.mean(), res.min(), res.max()))
