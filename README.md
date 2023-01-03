# CS7309 Final Project

**Author**: [Hongjie Fang](http://github.com/galaxies99/)

[[Report]](assets/report.pdf) 

## Introduction

This is the codebase of the final project of "CS7309 Reinforcement Learning" course in SJTU.

## Preliminary

Our codebase relies on several Python packages, including: `pytorch`, `einops`, `tqdm`, `gym` (version 0.21, with Atari and Mujoco (mjpro150)), `opencv-python`, `easydict`, `numpy`. Please install the packages before running our codes.

Our code is tested under the following environments:

- **Atari**: Ubuntu 20.04.3 LTS with NVIDIA A100 GPU (CUDA 11.4); Python 3.8.
- **Mujoco**: MacOS Big Sur with CPU; Python 3.8.

## Checkpoint

Download checkpoints at [Baidu Netdisk](https://pan.baidu.com/s/1CdFhlUiTs741v9yXq--U3A) (extraction code: onuk), then extract the zipped file into a folder `logs`, and put it in the root directory of our codebase.

## Testing

For testing, use the following command:

```bash
python run.py --env_name [Environment Name] 
              --method [Method Name] 
              --test [Test Episodes]
```

where `[Environment Name]` is the environment name, `[Method Name]` is the method name and `[Test Episodes]` is the episodes during testing (100 in our experiments).

Currently we support the following combinations of `[Environment Name]` and `[Method Name]`:

- **Atari**:
  - `[Environment Name]`: `PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4` and `BoxingNoFrameskip-v4`.
  - `[Method Name]`: `DQN`, `DoubleDQN` and `DuelingDQN`.
- **Mujoco**:
  - `[Environment Name]`: `Hopper-v2`, `HalfCheetah-v2` and `Ant-v2`.
  - `[Method Name]`: `DDPG` and `TD3`.

## Training (Optional)

For training (optional, since we provide trained checkpoints), use the following command:

```bash
python run.py --env_name [Environment Name] 
              --method [Method Name] 
              (--cfg [Configuration File])
```

where `[Environment Name]` is the environment name, `[Method Name]` is the method name and `[Configuration File]` is the optional configuration files (we provide the configuration files for training in the `configs` folder, if you want to train by yourselves, please refer to the configuration settings we provided).

## References

[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” arXiv preprint arXiv:1312.5602, 2013.

[2] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in IROS, pp. 5026–5033, IEEE, 2012.

[3] R. Bellman, “A markovian decision process,” Journal of Mathematics and Mechanics, pp. 679–684, 1957.

[4] C. J. C. H. Watkins, “Learning from delayed rewards,” 1989.

[5] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[6] M. Roderick, J. MacGlashan, and S. Tellex, “Implementing the deep q-network,” arXiv preprint arXiv:1711.07478, 2017.

[7] H. Hasselt, “Double q-learning,” in NeurIPS, 2010.

[8] H. Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in AAAI, 2016.

[9] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, and N. Freitas, “Dueling network architectures for deep reinforcement learning,” in ICML, 2016.

[10] S. Fujimoto, H. Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in ICML, 2018.

[11] I. Loshchilov and F. Hutter, “Fixing weight decay regularization in adam,” 2018.
