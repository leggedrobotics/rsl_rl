from dataclasses import MISSING, fields
from typing import TypeVar

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

RunnerCfg = TypeVar("RunnerCfg", bound=RslRlOnPolicyRunnerCfg)


@configclass
class RobotRlActorCriticCfg(RslRlPpoActorCriticCfg):
    estimator_index: int = -1
    oracle: bool = False

@configclass
class MHACfg:
    n_heads: int
    n_latent: int
    n_channels: int
    kernel_size: int
    dropout: float

@configclass
class ActorCriticMHACfg(RslRlPpoActorCriticCfg):
    mha: MHACfg = MISSING


@configclass
class SAECfg:
    """Configuration for the SAE network."""

    class_name: str = "SAE"
    """The policy class name. Default is SAE."""

    hidden_dim_scale: int = MISSING
    """The hidden dimensions of the SAE network."""

    sparsity_lambda: float = 1e-3
    """The loss coefficient encouraging sparsity in the feature dimension"""

    activation: str = "relu"
    """The activation function for the probe network."""


@configclass
class ProbeCfg:
    """Configuration for the Probe network."""

    class_name: str = "Probe"
    """The policy class name. Default is Probe."""

    probe_hidden_dims: list = []
    """The hidden dimensions of the probe network."""

    activation: str = "elu"
    """The activation function for the probe network."""


@configclass
class ProbeAlgorithmCfg:
    """Configuration for the Probe algorithm."""

    class_name: str = "Probe"
    """The algorithm class name. Default is Probe."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    learning_rate: float = MISSING
    """The learning rate for the probe."""

    estimate_obs: bool = False
    """Whether to estimate the observation (time t)."""

    estimate_next_obs: bool = False
    """Whether to estimate the next observation (time t+1)."""


@configclass
class RobotRlProbeRunner(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for probe algorithms."""

    algorithm: ProbeAlgorithmCfg = MISSING
    """The algorithm configuration."""

    probe: ProbeCfg | SAECfg = MISSING
    """The probe configuration."""

    layers: list = MISSING
    """The ActorCritic layers to probe."""

    policy_module: str = "actor"
    """ActorCritic attribute to probe.
    
    e.g. policy.policy_module[i]
    """

    probe_obs: bool = MISSING
    """Whether to probe the observation (time t)."""

    probe_next_obs: bool = MISSING
    """Whether to probe the next observation (time t+1)."""


@configclass
class EstimatorCfg:
    estimate_loss_coef: float = MISSING
    """Coef for estimator loss term."""

    estimate_loss_ramp: int = MISSING
    """Steps to linearly ramp recons coef."""

    estimate_obs: bool = MISSING
    """Whether to estimate the observation (time t)."""

    estimate_next_obs: bool = MISSING
    """Whether to estimate the next observation (time t+1)."""


@configclass
class RobotRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    estimator_cfg: EstimatorCfg | None = None
    """The estimator configuration. Default is None, in which case it is not used."""
