import copy
import numpy as np

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
    critic_activations=["tanh", "tanh", "linear"],
    critic_hidden_dims=[64, 64],
    critic_input_normalization=False,
    entropy_coeff=0.0,
    gae_lambda=0.95,
    gamma=0.99,
    gradient_clip=0.5,
    learning_rate=0.0003,
    schedule="adaptive",
    target_kl=0.01,
    value_coeff=0.5,
)

"""
[I 2023-01-09 00:33:02,217] Trial 85 finished with value: 2191.0249068421276 and parameters: {
    'actor_noise_std': 0.2611334861249876,
    'batch_count': 12,
    'clip_ratio': 0.4,
    'entropy_coeff': 0.010204149626344796,
    'env_count': 128,
    'gae_lambda': 0.92,
    'gamma': 0.9730549104215155,
    'gradient_clip': 5.0,
    'learning_rate': 0.8754540531090014,
    'steps_per_env': 64,
    'target_kl': 0.17110535070344035,
    'value_coeff': 0.6840401569818934,
    'actor_net_arch': 'small',
    'critic_net_arch': 'small',
    'actor_net_activation': 'tanh',
    'critic_net_activation': 'tanh'
}. Best is trial 85 with value: 2191.0249068421276.
"""
ant_v3 = copy.deepcopy(default)
ant_v3["env_kwargs"]["environment_count"] = 128
ant_v3["runner_kwargs"]["num_steps_per_env"] = 64
ant_v3["agent_kwargs"]["actor_activations"] = ["tanh", "tanh", "linear"]
ant_v3["agent_kwargs"]["actor_hidden_dims"] = [64, 64]
ant_v3["agent_kwargs"]["actor_noise_std"] = 0.2611
ant_v3["agent_kwargs"]["batch_count"] = 12
ant_v3["agent_kwargs"]["clip_ratio"] = 0.4
ant_v3["agent_kwargs"]["critic_activations"] = ["tanh", "tanh", "linear"]
ant_v3["agent_kwargs"]["critic_hidden_dims"] = [64, 64]
ant_v3["agent_kwargs"]["entropy_coeff"] = 0.0102
ant_v3["agent_kwargs"]["gae_lambda"] = 0.92
ant_v3["agent_kwargs"]["gamma"] = 0.9731
ant_v3["agent_kwargs"]["gradient_clip"] = 5.0
ant_v3["agent_kwargs"]["learning_rate"] = 0.8755
ant_v3["agent_kwargs"]["target_kl"] = 0.1711
ant_v3["agent_kwargs"]["value_coeff"] = 0.6840

"""
Standard:
[I 2023-01-17 07:43:46,884] Trial 125 finished with value: 150.23491836690064 and parameters: {
    'actor_net_activation': 'relu',
    'actor_net_arch': 'large',
    'actor_noise_std': 0.8504545432069994,
    'batch_count': 10,
    'clip_ratio': 0.1,
    'critic_net_activation': 'relu',
    'critic_net_arch': 'medium',
    'entropy_coeff': 0.0916881539697197,
    'env_count': 256,
    'gae_lambda': 0.95,
    'gamma': 0.955285858564339,
    'gradient_clip': 2.0,
    'learning_rate': 0.4762365866431558,
    'steps_per_env': 16,
    'recurrent': False,
    'target_kl': 0.19991906392721126,
    'value_coeff': 0.4434793554275927
}. Best is trial 125 with value: 150.23491836690064.
Hardcore:
[I 2023-01-09 06:25:44,000] Trial 262 finished with value: 2.290071208278338 and parameters: {
    'actor_noise_std': 0.2710521003644249,
    'batch_count': 6,
    'clip_ratio': 0.1,
    'entropy_coeff': 0.005105282891378981,
    'env_count': 16,
    'gae_lambda': 1.0,
    'gamma': 0.9718119008688937,
    'gradient_clip': 0.1,
    'learning_rate': 0.4569184610431825,
    'steps_per_env': 256,
    'target_kl': 0.11068348002480229,
    'value_coeff': 0.19453900570701116,
    'actor_net_arch': 'small',
    'critic_net_arch': 'medium',
    'actor_net_activation': 'relu',
    'critic_net_activation': 'relu'
}. Best is trial 262 with value: 2.290071208278338.
"""
bipedal_walker_v3 = copy.deepcopy(default)
bipedal_walker_v3["env_kwargs"]["environment_count"] = 256
bipedal_walker_v3["runner_kwargs"]["num_steps_per_env"] = 16
bipedal_walker_v3["agent_kwargs"]["actor_activations"] = ["relu", "relu", "relu", "linear"]
bipedal_walker_v3["agent_kwargs"]["actor_hidden_dims"] = [512, 256, 128]
bipedal_walker_v3["agent_kwargs"]["actor_noise_std"] = 0.8505
bipedal_walker_v3["agent_kwargs"]["batch_count"] = 10
bipedal_walker_v3["agent_kwargs"]["clip_ratio"] = 0.1
bipedal_walker_v3["agent_kwargs"]["critic_activations"] = ["relu", "relu", "linear"]
bipedal_walker_v3["agent_kwargs"]["critic_hidden_dims"] = [256, 256]
bipedal_walker_v3["agent_kwargs"]["entropy_coeff"] = 0.0917
bipedal_walker_v3["agent_kwargs"]["gae_lambda"] = 0.95
bipedal_walker_v3["agent_kwargs"]["gamma"] = 0.9553
bipedal_walker_v3["agent_kwargs"]["gradient_clip"] = 2.0
bipedal_walker_v3["agent_kwargs"]["learning_rate"] = 0.4762
bipedal_walker_v3["agent_kwargs"]["target_kl"] = 0.1999
bipedal_walker_v3["agent_kwargs"]["value_coeff"] = 0.4435

"""
[I 2023-01-04 05:57:20,749] Trial 1451 finished with value: 5260.338678148058 and parameters: {
    'env_count': 32,
    'actor_noise_std': 0.3397405098274084,
    'batch_count': 6,
    'clip_ratio': 0.3,
    'entropy_coeff': 0.009392937880259133,
    'gae_lambda': 0.8,
    'gamma': 0.9683403243382301,
    'gradient_clip': 5.0,
    'learning_rate': 0.5985206877398142,
    'steps_per_env': 16,
    'target_kl': 0.027651917189297347,
    'value_coeff': 0.26705235341068373,
    'net_arch': 'medium',
    'net_activation': 'tanh'
}. Best is trial 1451 with value: 5260.338678148058.
"""
half_cheetah_v3 = copy.deepcopy(default)
half_cheetah_v3["env_kwargs"]["environment_count"] = 32
half_cheetah_v3["runner_kwargs"]["num_steps_per_env"] = 16
half_cheetah_v3["agent_kwargs"]["actor_activations"] = ["tanh", "tanh", "linear"]
half_cheetah_v3["agent_kwargs"]["actor_hidden_dims"] = [256, 256]
half_cheetah_v3["agent_kwargs"]["actor_noise_std"] = 0.3397
half_cheetah_v3["agent_kwargs"]["batch_count"] = 6
half_cheetah_v3["agent_kwargs"]["clip_ratio"] = 0.3
half_cheetah_v3["agent_kwargs"]["critic_activations"] = ["tanh", "tanh", "linear"]
half_cheetah_v3["agent_kwargs"]["critic_hidden_dims"] = [256, 256]
half_cheetah_v3["agent_kwargs"]["entropy_coeff"] = 0.009393
half_cheetah_v3["agent_kwargs"]["gae_lambda"] = 0.8
half_cheetah_v3["agent_kwargs"]["gamma"] = 0.9683
half_cheetah_v3["agent_kwargs"]["gradient_clip"] = 5.0
half_cheetah_v3["agent_kwargs"]["learning_rate"] = 0.5985
half_cheetah_v3["agent_kwargs"]["target_kl"] = 0.02765
half_cheetah_v3["agent_kwargs"]["value_coeff"] = 0.2671

"""
[I 2023-01-08 18:38:51,481] Trial 25 finished with value: 2225.9547948810073 and parameters: {
    'actor_noise_std': 0.5589708917145111,
    'batch_count': 15,
    'clip_ratio': 0.2,
    'entropy_coeff': 0.03874027035272886,
    'env_count': 128,
    'gae_lambda': 0.98,
    'gamma': 0.9879577396280973,
    'gradient_clip': 1.0,
    'learning_rate': 0.3732431793266761,
    'steps_per_env': 128,
    'target_kl': 0.12851506672519566,
    'value_coeff': 0.8162548885723906,
    'actor_net_arch': 'medium',
    'critic_net_arch': 'small',
    'actor_net_activation': 'relu',
    'critic_net_activation': 'relu'
}. Best is trial 25 with value: 2225.9547948810073.
"""
hopper_v3 = copy.deepcopy(default)
half_cheetah_v3["env_kwargs"]["environment_count"] = 128
hopper_v3["runner_kwargs"]["num_steps_per_env"] = 128
hopper_v3["agent_kwargs"]["actor_activations"] = ["relu", "relu", "linear"]
hopper_v3["agent_kwargs"]["actor_hidden_dims"] = [256, 256]
hopper_v3["agent_kwargs"]["actor_noise_std"] = 0.5590
hopper_v3["agent_kwargs"]["batch_count"] = 15
hopper_v3["agent_kwargs"]["clip_ratio"] = 0.2
hopper_v3["agent_kwargs"]["critic_activations"] = ["relu", "relu", "linear"]
hopper_v3["agent_kwargs"]["critic_hidden_dims"] = [32, 32]
hopper_v3["agent_kwargs"]["entropy_coeff"] = 0.03874
hopper_v3["agent_kwargs"]["gae_lambda"] = 0.98
hopper_v3["agent_kwargs"]["gamma"] = 0.9890
hopper_v3["agent_kwargs"]["gradient_clip"] = 1.0
hopper_v3["agent_kwargs"]["learning_rate"] = 0.3732
hopper_v3["agent_kwargs"]["value_coeff"] = 0.8163

swimmer_v3 = copy.deepcopy(default)
swimmer_v3["agent_kwargs"]["gamma"] = 0.9999

walker2d_v3 = copy.deepcopy(default)
walker2d_v3["runner_kwargs"]["num_steps_per_env"] = 512
walker2d_v3["agent_kwargs"]["batch_count"] = (
    walker2d_v3["env_kwargs"]["environment_count"] * walker2d_v3["runner_kwargs"]["num_steps_per_env"] // 32
)
walker2d_v3["agent_kwargs"]["clip_ratio"] = 0.1
walker2d_v3["agent_kwargs"]["entropy_coeff"] = 0.000585045
walker2d_v3["agent_kwargs"]["gae_lambda"] = 0.95
walker2d_v3["agent_kwargs"]["gamma"] = 0.99
walker2d_v3["agent_kwargs"]["gradient_clip"] = 1.0
walker2d_v3["agent_kwargs"]["learning_rate"] = 5.05041e-05
walker2d_v3["agent_kwargs"]["value_coeff"] = 0.871923

ppo_hyperparams = {
    "default": default,
    "Ant-v3": ant_v3,
    "Ant-v4": ant_v3,
    "BipedalWalker-v3": bipedal_walker_v3,
    "HalfCheetah-v3": half_cheetah_v3,
    "HalfCheetah-v4": half_cheetah_v3,
    "Hopper-v3": hopper_v3,
    "Hopper-v4": hopper_v3,
    "Swimmer-v3": swimmer_v3,
    "Swimmer-v4": swimmer_v3,
    "Walker2d-v3": walker2d_v3,
    "Walker2d-v4": walker2d_v3,
}
