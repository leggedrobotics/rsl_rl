import torch

from rsl_rl.algorithms import DPPO, PPO
from rsl_rl.modules import QuantileNetwork

NETWORKS = {"small": [64, 64], "medium": [256, 256], "large": [512, 256, 128]}


def sample_dppo_hyperparams(trial):
    actor_net_activation = trial.suggest_categorical("actor_net_activation", ["relu", "tanh"])
    actor_net_arch = trial.suggest_categorical("actor_net_arch", list(NETWORKS.keys()))
    actor_noise_std = trial.suggest_float("actor_noise_std", 0.05, 1.0)
    batch_count = trial.suggest_int("batch_count", 1, 20)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.1, 0.2, 0.3, 0.4])
    critic_net_activation = trial.suggest_categorical("critic_net_activation", ["relu", "tanh"])
    critic_net_arch = trial.suggest_categorical("critic_net_arch", list(NETWORKS.keys()))
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.00000001, 0.1)
    env_count = trial.suggest_categorical("env_count", [1, 8, 16, 32, 64, 128, 256, 512])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gradient_clip = trial.suggest_categorical("gradient_clip", [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1)
    num_steps_per_env = trial.suggest_categorical("steps_per_env", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    quantile_count = trial.suggest_categorical("quantile_count", [20, 50, 100, 200])
    recurrent = trial.suggest_categorical("recurrent", [True, False])
    target_kl = trial.suggest_float("target_kl", 0.01, 0.3)
    value_coeff = trial.suggest_float("value_coeff", 0.0, 1.0)
    value_measure = trial.suggest_categorical(
        "value_measure",
        ["neutral", "var-risk-averse", "var-risk-seeking", "var-super-risk-averse", "var-super-risk-seeking"],
    )

    actor_net_arch = NETWORKS[actor_net_arch]
    critic_net_arch = NETWORKS[critic_net_arch]
    value_measure_kwargs = {
        "neutral": dict(),
        "var-risk-averse": dict(confidence_level=0.25),
        "var-risk-seeking": dict(confidence_level=0.75),
        "var-super-risk-averse": dict(confidence_level=0.1),
        "var-super-risk-seeking": dict(confidence_level=0.9),
    }[value_measure]
    value_measure = {
        "neutral": QuantileNetwork.measure_neutral,
        "var-risk-averse": QuantileNetwork.measure_percentile,
        "var-risk-seeking": QuantileNetwork.measure_percentile,
        "var-super-risk-averse": QuantileNetwork.measure_percentile,
        "var-super-risk-seeking": QuantileNetwork.measure_percentile,
    }[value_measure]
    device = "cuda:0" if env_count * num_steps_per_env > 2048 and torch.cuda.is_available() else "cpu"

    agent_kwargs = dict(
        actor_activations=([actor_net_activation] * len(actor_net_arch)) + ["linear"],
        actor_hidden_dims=actor_net_arch,
        actor_input_normalization=False,
        actor_noise_std=actor_noise_std,
        batch_count=batch_count,
        clip_ratio=clip_ratio,
        critic_activations=([critic_net_activation] * len(critic_net_arch)),
        critic_hidden_dims=critic_net_arch,
        critic_input_normalization=False,
        device=device,
        entropy_coeff=entropy_coeff,
        gae_lambda=gae_lambda,
        gamma=gamma,
        gradient_clip=gradient_clip,
        learning_rate=learning_rate,
        quantile_count=quantile_count,
        recurrent=recurrent,
        schedule="adaptive",
        target_kl=target_kl,
        value_coeff=value_coeff,
        value_measure=value_measure,
        value_measure_kwargs=value_measure_kwargs,
    )
    env_kwargs = dict(device=device, environment_count=env_count)
    runner_kwargs = dict(device=device, num_steps_per_env=num_steps_per_env)

    return agent_kwargs, env_kwargs, runner_kwargs


def sample_ppo_hyperparams(trial):
    actor_net_activation = trial.suggest_categorical("actor_net_activation", ["relu", "tanh"])
    actor_net_arch = trial.suggest_categorical("actor_net_arch", list(NETWORKS.keys()))
    actor_noise_std = trial.suggest_float("actor_noise_std", 0.05, 1.0)
    batch_count = trial.suggest_int("batch_count", 1, 20)
    clip_ratio = trial.suggest_categorical("clip_ratio", [0.1, 0.2, 0.3, 0.4])
    critic_net_activation = trial.suggest_categorical("critic_net_activation", ["relu", "tanh"])
    critic_net_arch = trial.suggest_categorical("critic_net_arch", list(NETWORKS.keys()))
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.00000001, 0.1)
    env_count = trial.suggest_categorical("env_count", [1, 8, 16, 32, 64, 128, 256, 512])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gradient_clip = trial.suggest_categorical("gradient_clip", [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1)
    num_steps_per_env = trial.suggest_categorical("steps_per_env", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    recurrent = trial.suggest_categorical("recurrent", [True, False])
    target_kl = trial.suggest_float("target_kl", 0.01, 0.3)
    value_coeff = trial.suggest_float("value_coeff", 0.0, 1.0)

    actor_net_arch = NETWORKS[actor_net_arch]
    critic_net_arch = NETWORKS[critic_net_arch]
    device = "cuda:0" if env_count * num_steps_per_env > 2048 and torch.cuda.is_available() else "cpu"

    agent_kwargs = dict(
        actor_activations=([actor_net_activation] * len(actor_net_arch)) + ["linear"],
        actor_hidden_dims=actor_net_arch,
        actor_input_normalization=False,
        actor_noise_std=actor_noise_std,
        batch_count=batch_count,
        clip_ratio=clip_ratio,
        critic_activations=([critic_net_activation] * len(critic_net_arch)) + ["linear"],
        critic_hidden_dims=critic_net_arch,
        critic_input_normalization=False,
        device=device,
        entropy_coeff=entropy_coeff,
        gae_lambda=gae_lambda,
        gamma=gamma,
        gradient_clip=gradient_clip,
        learning_rate=learning_rate,
        recurrent=recurrent,
        schedule="adaptive",
        target_kl=target_kl,
        value_coeff=value_coeff,
    )
    env_kwargs = dict(device=device, environment_count=env_count)
    runner_kwargs = dict(device=device, num_steps_per_env=num_steps_per_env)

    return agent_kwargs, env_kwargs, runner_kwargs


samplers = {
    DPPO.__name__: sample_dppo_hyperparams,
    PPO.__name__: sample_ppo_hyperparams,
}
