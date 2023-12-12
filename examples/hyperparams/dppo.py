import copy
import numpy as np

from rsl_rl.algorithms import DPPO
from rsl_rl.modules import QuantileNetwork

default = dict()
default["env_kwargs"] = dict(environment_count=1)
default["runner_kwargs"] = dict(num_steps_per_env=2048)
default["agent_kwargs"] = dict(
    actor_activations=["tanh", "tanh", "linear"],
    actor_hidden_dims=[64, 64],
    actor_input_normalization=False,
    actor_noise_std=np.exp(0.0),
    batch_count=(default["env_kwargs"]["environment_count"] * default["runner_kwargs"]["num_steps_per_env"] // 64),
    clip_ratio=0.2,
    critic_activations=["tanh", "tanh"],
    critic_hidden_dims=[64, 64],
    critic_input_normalization=False,
    entropy_coeff=0.0,
    gae_lambda=0.95,
    gamma=0.99,
    gradient_clip=0.5,
    learning_rate=0.0003,
    qrdqn_quantile_count=50,
    schedule="adaptive",
    target_kl=0.01,
    value_coeff=0.5,
    value_measure=QuantileNetwork.measure_neutral,
    value_measure_kwargs={},
)

# Parameters optimized for PPO
ant_v4 = copy.deepcopy(default)
ant_v4["env_kwargs"]["environment_count"] = 128
ant_v4["runner_kwargs"]["num_steps_per_env"] = 64
ant_v4["agent_kwargs"]["actor_activations"] = ["tanh", "tanh", "linear"]
ant_v4["agent_kwargs"]["actor_hidden_dims"] = [64, 64]
ant_v4["agent_kwargs"]["actor_noise_std"] = 0.2611
ant_v4["agent_kwargs"]["batch_count"] = 12
ant_v4["agent_kwargs"]["clip_ratio"] = 0.4
ant_v4["agent_kwargs"]["critic_activations"] = ["tanh", "tanh"]
ant_v4["agent_kwargs"]["critic_hidden_dims"] = [64, 64]
ant_v4["agent_kwargs"]["entropy_coeff"] = 0.0102
ant_v4["agent_kwargs"]["gae_lambda"] = 0.92
ant_v4["agent_kwargs"]["gamma"] = 0.9731
ant_v4["agent_kwargs"]["gradient_clip"] = 5.0
ant_v4["agent_kwargs"]["learning_rate"] = 0.8755
ant_v4["agent_kwargs"]["target_kl"] = 0.1711
ant_v4["agent_kwargs"]["value_coeff"] = 0.6840

"""
Tuned for environment interactions:
[I 2023-01-03 03:11:29,212] Trial 19 finished with value: 0.5272218152693111 and parameters: {
    'env_count': 16,
    'actor_noise_std': 0.7304437880901905,
    'batch_count': 10,
    'clip_ratio': 0.3,
    'entropy_coeff': 0.004236574285220795,
    'gae_lambda': 0.95,
    'gamma': 0.9890074826092162,
    'gradient_clip': 0.9,
    'learning_rate': 0.18594043324129061,
    'steps_per_env': 256,
    'target_kl': 0.05838576142010138,
    'value_coeff': 0.14402022632575992,
    'net_arch': 'small',
    'net_activation': 'relu'
}. Best is trial 19 with value: 0.5272218152693111.
Tuned for training time:
[I 2023-01-08 21:09:06,069] Trial 407 finished with value: 7.497591958940029 and parameters: {
    'actor_noise_std': 0.1907398121300662,
    'batch_count': 3,
    'clip_ratio': 0.1,
    'entropy_coeff': 0.0053458057035692735,
    'env_count': 16,
    'gae_lambda': 0.8,
    'gamma': 0.985000267068182,
    'gradient_clip': 2.0,
    'learning_rate': 0.605956844400053,
    'steps_per_env': 512,
    'target_kl': 0.17611450607281642,
    'value_coeff': 0.46015664905111847,
    'actor_net_arch': 'small',
    'critic_net_arch': 'medium',
    'actor_net_activation': 'relu',
    'critic_net_activation': 'relu',
    'qrdqn_quantile_count': 200,
    'value_measure': 'neutral'
}. Best is trial 407 with value: 7.497591958940029.
"""
bipedal_walker_v3 = copy.deepcopy(default)
bipedal_walker_v3["env_kwargs"]["environment_count"] = 256
bipedal_walker_v3["runner_kwargs"]["num_steps_per_env"] = 16
bipedal_walker_v3["agent_kwargs"]["actor_activations"] = ["relu", "relu", "relu", "linear"]
bipedal_walker_v3["agent_kwargs"]["actor_hidden_dims"] = [512, 256, 128]
bipedal_walker_v3["agent_kwargs"]["actor_noise_std"] = 0.8505
bipedal_walker_v3["agent_kwargs"]["batch_count"] = 10
bipedal_walker_v3["agent_kwargs"]["clip_ratio"] = 0.1
bipedal_walker_v3["agent_kwargs"]["critic_activations"] = ["relu", "relu"]
bipedal_walker_v3["agent_kwargs"]["critic_hidden_dims"] = [256, 256]
bipedal_walker_v3["agent_kwargs"]["critic_network"] = DPPO.network_qrdqn
bipedal_walker_v3["agent_kwargs"]["entropy_coeff"] = 0.0917
bipedal_walker_v3["agent_kwargs"]["gae_lambda"] = 0.95
bipedal_walker_v3["agent_kwargs"]["gamma"] = 0.9553
bipedal_walker_v3["agent_kwargs"]["gradient_clip"] = 2.0
bipedal_walker_v3["agent_kwargs"]["iqn_action_samples"] = 32
bipedal_walker_v3["agent_kwargs"]["iqn_embedding_size"] = 64
bipedal_walker_v3["agent_kwargs"]["iqn_feature_layers"] = 1
bipedal_walker_v3["agent_kwargs"]["iqn_value_samples"] = 8
bipedal_walker_v3["agent_kwargs"]["learning_rate"] = 0.4762
bipedal_walker_v3["agent_kwargs"]["qrdqn_quantile_count"] = 200
bipedal_walker_v3["agent_kwargs"]["recurrent"] = False
bipedal_walker_v3["agent_kwargs"]["target_kl"] = 0.1999
bipedal_walker_v3["agent_kwargs"]["value_coeff"] = 0.4435

"""
[I 2023-01-12 08:01:35,514] Trial 476 finished with value: 5202.960759290059 and parameters: {
    'actor_noise_std': 0.15412869066185989,
    'batch_count': 11,
    'clip_ratio': 0.3,
    'entropy_coeff': 0.036031209302206955,
    'env_count': 128,
    'gae_lambda': 0.92,
    'gamma': 0.973937576989299,
    'gradient_clip': 5.0,
    'learning_rate': 0.1621249118505433,
    'steps_per_env': 128,
    'target_kl': 0.05054738172852222,
    'value_coeff': 0.1647632125820593,
    'actor_net_arch': 'small',
    'critic_net_arch': 'medium',
    'actor_net_activation': 'tanh',
    'critic_net_activation': 'relu',
    'qrdqn_quantile_count': 50,
    'value_measure': 'var-risk-averse'
}. Best is trial 476 with value: 5202.960759290059.
"""
half_cheetah_v4 = copy.deepcopy(default)
half_cheetah_v4["env_kwargs"]["environment_count"] = 128
half_cheetah_v4["runner_kwargs"]["num_steps_per_env"] = 128
half_cheetah_v4["agent_kwargs"]["actor_activations"] = ["tanh", "tanh", "linear"]
half_cheetah_v4["agent_kwargs"]["actor_hidden_dims"] = [64, 64]
half_cheetah_v4["agent_kwargs"]["actor_noise_std"] = 0.1541
half_cheetah_v4["agent_kwargs"]["batch_count"] = 11
half_cheetah_v4["agent_kwargs"]["clip_ratio"] = 0.3
half_cheetah_v4["agent_kwargs"]["critic_activations"] = ["relu", "relu"]
half_cheetah_v4["agent_kwargs"]["critic_hidden_dims"] = [256, 256]
half_cheetah_v4["agent_kwargs"]["entropy_coeff"] = 0.03603
half_cheetah_v4["agent_kwargs"]["gae_lambda"] = 0.92
half_cheetah_v4["agent_kwargs"]["gamma"] = 0.9739
half_cheetah_v4["agent_kwargs"]["gradient_clip"] = 5.0
half_cheetah_v4["agent_kwargs"]["learning_rate"] = 0.1621
half_cheetah_v4["agent_kwargs"]["qrdqn_quantile_count"] = 50
half_cheetah_v4["agent_kwargs"]["target_kl"] = 0.0505
half_cheetah_v4["agent_kwargs"]["value_coeff"] = 0.1648
half_cheetah_v4["agent_kwargs"]["value_measure"] = QuantileNetwork.measure_percentile
half_cheetah_v4["agent_kwargs"]["value_measure_kwargs"] = dict(confidence_level=0.25)


# Parameters optimized for PPO
hopper_v4 = copy.deepcopy(default)
hopper_v4["runner_kwargs"]["num_steps_per_env"] = 128
hopper_v4["agent_kwargs"]["actor_activations"] = ["relu", "relu", "linear"]
hopper_v4["agent_kwargs"]["actor_hidden_dims"] = [256, 256]
hopper_v4["agent_kwargs"]["actor_noise_std"] = 0.5590
hopper_v4["agent_kwargs"]["batch_count"] = 15
hopper_v4["agent_kwargs"]["clip_ratio"] = 0.2
hopper_v4["agent_kwargs"]["critic_activations"] = ["relu", "relu", "linear"]
hopper_v4["agent_kwargs"]["critic_hidden_dims"] = [32, 32]
hopper_v4["agent_kwargs"]["entropy_coeff"] = 0.03874
hopper_v4["agent_kwargs"]["gae_lambda"] = 0.98
hopper_v4["agent_kwargs"]["gamma"] = 0.9890
hopper_v4["agent_kwargs"]["gradient_clip"] = 1.0
hopper_v4["agent_kwargs"]["learning_rate"] = 0.3732
hopper_v4["agent_kwargs"]["value_coeff"] = 0.8163

swimmer_v4 = copy.deepcopy(default)
swimmer_v4["agent_kwargs"]["gamma"] = 0.9999

walker2d_v4 = copy.deepcopy(default)
walker2d_v4["runner_kwargs"]["num_steps_per_env"] = 512
walker2d_v4["agent_kwargs"]["batch_count"] = (
    walker2d_v4["env_kwargs"]["environment_count"] * walker2d_v4["runner_kwargs"]["num_steps_per_env"] // 32
)
walker2d_v4["agent_kwargs"]["clip_ratio"] = 0.1
walker2d_v4["agent_kwargs"]["entropy_coeff"] = 0.000585045
walker2d_v4["agent_kwargs"]["gae_lambda"] = 0.95
walker2d_v4["agent_kwargs"]["gamma"] = 0.99
walker2d_v4["agent_kwargs"]["gradient_clip"] = 1.0
walker2d_v4["agent_kwargs"]["learning_rate"] = 5.05041e-05
walker2d_v4["agent_kwargs"]["value_coeff"] = 0.871923

dppo_hyperparams = {
    "default": default,
    "Ant-v4": ant_v4,
    "BipedalWalker-v3": bipedal_walker_v3,
    "HalfCheetah-v4": half_cheetah_v4,
    "Hopper-v4": hopper_v4,
    "Swimmer-v4": swimmer_v4,
    "Walker2d-v4": walker2d_v4,
}
