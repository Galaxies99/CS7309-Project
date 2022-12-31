from easydict import EasyDict as edict

default_cfgs = edict()

default_cfgs.seed = 23                           # Random seed
default_cfgs.environment = "PongNoFrameskip-v4"  # Environment

default_cfgs.agent = edict()
default_cfgs.agent.gamma = 0.99                  # Discount factor
default_cfgs.agent.batch_size = 32               # Batch number
default_cfgs.agent.replay_buffer_size = 5000     # Size of replay buffer
default_cfgs.agent.learning_rate = 1e-4          # Learning rate of the optimizer
default_cfgs.agent.optimizer = edict()
default_cfgs.agent.optimizer.type = "RMSprop"    # Optimizer Type
default_cfgs.agent.optimizer.params = {}         # Optimizer Parameters

default_cfgs.training = edict()
default_cfgs.training.eps_begin = 1              # E-greedy start threshold
default_cfgs.training.eps_end = 0.01             # E-greedy end threshold
default_cfgs.training.eps_fraction = 0.1         # E-greedy fraction of step number

default_cfgs.training.num_steps = 1000000        # Total number of steps

default_cfgs.training.train_opt_freq = 1         # Number of iterations between each optimization step
default_cfgs.training.start_step = 10000         # Number of steps before learning starts
default_cfgs.training.target_upd_freq = 1000     # Number of iterations between every target network update

default_cfgs.logs = edict()
default_cfgs.logs.checkpoint = edict()
default_cfgs.logs.checkpoint.load = False        # Whether to load checkpoint
default_cfgs.logs.checkpoint.path = "logs/atari/PongNoFrameskip-v4/double_dqn/checkpoints"
                                                 # Path to the checkpoint
default_cfgs.logs.checkpoint.name = "checkpoint.pth"
                                                 # Checkpoint name
default_cfgs.logs.video = edict()
default_cfgs.logs.video.path = "logs/atari/PongNoFrameskip-v4/double_dqn/videos"
                                                 # Path to the videos
default_cfgs.logs.video.freq = 50                # Video frequency
default_cfgs.logs.print = edict()
default_cfgs.logs.print.path = "logs/atari/PongNoFrameskip-v4/double_dqn/records"
                                                 # Path to the records
default_cfgs.logs.print.freq = 10                # Print frequency