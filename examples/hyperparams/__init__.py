from rsl_rl.algorithms import PPO, DPPO
from .dppo import dppo_hyperparams
from .ppo import ppo_hyperparams

hyperparams = {DPPO.__name__: dppo_hyperparams, PPO.__name__: ppo_hyperparams}

__all__ = ["hyperparams"]
