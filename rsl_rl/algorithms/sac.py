# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class SAC:
    """Soft Actor-Critic algorithm for continuous control."""

    def __init__(
        self,
        actor_obs_groups: list[str],
        critic_obs_groups: list[str],
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        actor_activation: str = "elu",
        critic_activation: str = "elu",
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float | None = None,
        replay_buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gradient_steps: int = 1,
        learning_starts: int = 1_000,
        optimizer: str = "adam",
        device: str = "cpu",
        actor: MLPModel | None = None,
        critic_1: MLPModel | None = None,
        critic_2: MLPModel | None = None,
        **kwargs: dict,
    ) -> None:
        del kwargs

        self.device = device
        self.actor_obs_groups = actor_obs_groups
        self.critic_obs_groups = critic_obs_groups
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.learning_rate = learning_rate

        if actor is None or critic_1 is None or critic_2 is None:
            actor, critic_1, critic_2 = self._build_legacy_models(
                actor_obs_dim=actor_obs_dim,
                critic_obs_dim=critic_obs_dim,
                action_dim=action_dim,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                actor_activation=actor_activation,
                critic_activation=critic_activation,
            )

        self.actor = actor.to(device)
        self.critic_1 = critic_1.to(device)
        self.critic_2 = critic_2.to(device)

        if self.actor.distribution is None:
            raise ValueError("SAC actor requires a stochastic output distribution.")

        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)

        optimizer_cls = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_cls(self.actor.parameters(), lr=learning_rate)  # type: ignore
        self.critic_optimizer = optimizer_cls(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=learning_rate,
        )  # type: ignore

        self.auto_entropy_tuning = auto_entropy_tuning
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor([torch.log(torch.tensor(alpha))], device=device, requires_grad=True)
            self.alpha_optimizer = optimizer_cls([self.log_alpha], lr=learning_rate)  # type: ignore
            self._alpha = self.log_alpha.exp().detach()
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self._alpha = torch.tensor([alpha], device=device)

        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=device,
        )

        self._last_action_std = torch.ones(action_dim, device=device)
        self._last_actor_obs: torch.Tensor | None = None
        self._last_critic_obs: torch.Tensor | None = None
        self._last_actions: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_entropy_tuning:
            return self.log_alpha.exp()  # type: ignore
        return self._alpha

    def cat_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[g] for g in groups], dim=-1)

    def _build_actor_mlp_output(self, actor_obs: torch.Tensor) -> torch.Tensor:
        latent = self.actor.obs_normalizer(actor_obs)
        return self.actor.mlp(latent)

    def _critic_forward(self, critic: MLPModel, critic_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat((critic_obs, actions), dim=-1)
        critic_input = critic.obs_normalizer(critic_input)
        return critic.mlp(critic_input)

    def _sample_action_and_log_prob(
        self, actor_obs: torch.Tensor, reparameterize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mlp_output = self._build_actor_mlp_output(actor_obs)
        self.actor.distribution.update(mlp_output)
        self._last_action_std = self.actor.distribution.std.detach().mean(dim=0)

        if hasattr(self.actor.distribution, "sample_with_log_prob"):
            actions, log_prob = self.actor.distribution.sample_with_log_prob(reparameterize=reparameterize)  # type: ignore[attr-defined]
            return actions, log_prob

        if reparameterize and hasattr(self.actor.distribution, "rsample"):
            actions = self.actor.distribution.rsample()  # type: ignore[attr-defined]
        else:
            actions = self.actor.distribution.sample()
        log_prob = self.actor.distribution.log_prob(actions).unsqueeze(-1)
        return actions, log_prob

    def act(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.cat_obs(obs, self.actor_obs_groups)
        critic_obs = self.cat_obs(obs, self.critic_obs_groups)

        with torch.no_grad():
            if self.replay_buffer.size < self.learning_starts:
                actions = torch.empty(actor_obs.shape[0], self.action_dim, device=self.device).uniform_(-1.0, 1.0)
                self._last_action_std = torch.ones(self.action_dim, device=self.device)
            else:
                actions, _ = self._sample_action_and_log_prob(actor_obs, reparameterize=False)

        self._last_actor_obs = actor_obs.detach()
        self._last_critic_obs = critic_obs.detach()
        self._last_actions = actions.detach()
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        del extras
        if self._last_actor_obs is None or self._last_critic_obs is None or self._last_actions is None:
            raise RuntimeError("act() must be called before process_env_step().")

        next_actor_obs = self.cat_obs(obs, self.actor_obs_groups).detach()
        next_critic_obs = self.cat_obs(obs, self.critic_obs_groups).detach()

        # Keep running normalizer statistics in sync with streamed environment data.
        if self.actor.obs_normalization:
            self.actor.obs_normalizer.update(next_actor_obs)  # type: ignore[attr-defined]
        if self.critic_1.obs_normalization:
            critic_input_1 = torch.cat((self._last_critic_obs, self._last_actions), dim=-1)
            self.critic_1.obs_normalizer.update(critic_input_1)  # type: ignore[attr-defined]
        if self.critic_2.obs_normalization:
            critic_input_2 = torch.cat((self._last_critic_obs, self._last_actions), dim=-1)
            self.critic_2.obs_normalizer.update(critic_input_2)  # type: ignore[attr-defined]

        rewards = rewards.view(-1, 1).detach()
        dones = dones.view(-1, 1).detach()

        self.replay_buffer.add(
            actor_obs=self._last_actor_obs,
            critic_obs=self._last_critic_obs,
            actions=self._last_actions,
            rewards=rewards,
            dones=dones,
            next_actor_obs=next_actor_obs,
            next_critic_obs=next_critic_obs,
        )

    def update(self) -> dict[str, float]:
        if self.replay_buffer.size < max(self.batch_size, self.learning_starts):
            return {
                "critic": 0.0,
                "actor": 0.0,
                "alpha": self.alpha.item(),
                "buffer_size": float(self.replay_buffer.size),
            }

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0

        for _ in range(self.gradient_steps):
            actor_obs, critic_obs, actions, rewards, dones, next_actor_obs, next_critic_obs = self.replay_buffer.sample(
                self.batch_size
            )

            with torch.no_grad():
                next_actions, next_log_prob = self._sample_action_and_log_prob(next_actor_obs, reparameterize=False)
                next_q1 = self._critic_forward(self.target_critic_1, next_critic_obs, next_actions)
                next_q2 = self._critic_forward(self.target_critic_2, next_critic_obs, next_actions)
                next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_prob
                target_q = rewards + (torch.ones_like(dones) - dones) * self.gamma * next_q

            current_q1 = self._critic_forward(self.critic_1, critic_obs, actions)
            current_q2 = self._critic_forward(self.critic_2, critic_obs, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            new_actions, log_prob = self._sample_action_and_log_prob(actor_obs, reparameterize=True)
            q1_pi = self._critic_forward(self.critic_1, critic_obs, new_actions)
            q2_pi = self._critic_forward(self.critic_2, critic_obs, new_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()  # type: ignore
                self.alpha_optimizer.zero_grad()  # type: ignore
                alpha_loss.backward()
                self.alpha_optimizer.step()  # type: ignore
                total_alpha_loss += alpha_loss.item()

            with torch.no_grad():
                for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)
                for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()

        loss_dict = {
            "critic": total_critic_loss / self.gradient_steps,
            "actor": total_actor_loss / self.gradient_steps,
            "alpha": self.alpha.item(),
            "buffer_size": float(self.replay_buffer.size),
        }
        if self.auto_entropy_tuning:
            loss_dict["alpha_loss"] = total_alpha_loss / self.gradient_steps
        return loss_dict

    def train_mode(self) -> None:
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def save(self) -> dict:
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "target_critic_1_state_dict": self.target_critic_1.state_dict(),
            "target_critic_2_state_dict": self.target_critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "replay_buffer_size": self.replay_buffer.size,
            "replay_buffer_ptr": self.replay_buffer.ptr,
        }
        if self.auto_entropy_tuning:
            saved_dict["log_alpha"] = self.log_alpha.detach().clone()  # type: ignore
            saved_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()  # type: ignore
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "target_critic": True,
                "optimizer": True,
                "alpha": True,
                "iteration": True,
            }

        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic_1.load_state_dict(loaded_dict["critic_1_state_dict"], strict=strict)
            self.critic_2.load_state_dict(loaded_dict["critic_2_state_dict"], strict=strict)
        if load_cfg.get("target_critic"):
            self.target_critic_1.load_state_dict(loaded_dict["target_critic_1_state_dict"], strict=strict)
            self.target_critic_2.load_state_dict(loaded_dict["target_critic_2_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        if load_cfg.get("alpha") and self.auto_entropy_tuning and "log_alpha" in loaded_dict:
            self.log_alpha.data.copy_(loaded_dict["log_alpha"])  # type: ignore
            if "alpha_optimizer_state_dict" in loaded_dict:
                self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])  # type: ignore

        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.actor

    def get_action_std(self) -> torch.Tensor:
        return self._last_action_std

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        alg_class: type[SAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        actor_cfg = cfg.get("actor", {})
        critic_cfg = cfg.get("critic", {})

        actor_class_name = actor_cfg.get("class_name", "MLPModel")
        critic_class_name = critic_cfg.get("class_name", "MLPModel")
        if actor_class_name not in ("MLPModel", "rsl_rl.models:MLPModel", "rsl_rl.models.mlp_model:MLPModel"):
            raise ValueError("SAC currently supports only MLP-style actor configuration.")
        if critic_class_name not in ("MLPModel", "rsl_rl.models:MLPModel", "rsl_rl.models.mlp_model:MLPModel"):
            raise ValueError("SAC currently supports only MLP-style critic configuration.")

        actor_cfg = dict(actor_cfg)
        critic_cfg = dict(critic_cfg)
        actor_cfg.pop("class_name", None)
        critic_cfg.pop("class_name", None)

        # Use a tanh-squashed Gaussian policy by default for SAC.
        actor_cfg.setdefault(
            "distribution_cfg",
            {
                "class_name": "SquashedGaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        )

        actor_obs_groups = cfg["obs_groups"]["actor"]
        critic_obs_groups = cfg["obs_groups"]["critic"]
        actor_obs_dim = sum(obs[group].shape[-1] for group in actor_obs_groups)
        critic_obs_dim = sum(obs[group].shape[-1] for group in critic_obs_groups)

        actor_class: type[MLPModel] = resolve_callable(actor_class_name)  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_class_name)  # type: ignore

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg)

        # Critic input is [critic_obs, actions]. Build a small template obs for model construction.
        critic_obs_template = TensorDict(
            {"critic_input": torch.zeros(obs.batch_size[0], critic_obs_dim + env.num_actions, device=device)},
            batch_size=obs.batch_size,
            device=device,
        )
        critic_obs_groups_cfg = {"critic": ["critic_input"]}
        critic_1 = critic_class(critic_obs_template, critic_obs_groups_cfg, "critic", 1, **critic_cfg)
        critic_2 = critic_class(critic_obs_template, critic_obs_groups_cfg, "critic", 1, **critic_cfg)

        alg: SAC = alg_class(
            actor_obs_groups=actor_obs_groups,
            critic_obs_groups=critic_obs_groups,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=env.num_actions,
            actor=actor,
            critic_1=critic_1,
            critic_2=critic_2,
            device=device,
            **cfg["algorithm"],
        )
        return alg

    @staticmethod
    def _build_legacy_models(
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...] | list[int],
        critic_hidden_dims: tuple[int, ...] | list[int],
        actor_activation: str,
        critic_activation: str,
    ) -> tuple[MLPModel, MLPModel, MLPModel]:
        """Build default MLP models for backward-compatible SAC construction."""
        actor_obs = TensorDict({"actor_obs": torch.zeros(1, actor_obs_dim)}, batch_size=[1])
        actor_groups = {"actor": ["actor_obs"]}
        actor = MLPModel(
            actor_obs,
            actor_groups,
            "actor",
            action_dim,
            hidden_dims=actor_hidden_dims,
            activation=actor_activation,
            distribution_cfg={
                "class_name": "SquashedGaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        )

        critic_obs = TensorDict({"critic_input": torch.zeros(1, critic_obs_dim + action_dim)}, batch_size=[1])
        critic_groups = {"critic": ["critic_input"]}
        critic_1 = MLPModel(
            critic_obs,
            critic_groups,
            "critic",
            1,
            hidden_dims=critic_hidden_dims,
            activation=critic_activation,
        )
        critic_2 = MLPModel(
            critic_obs,
            critic_groups,
            "critic",
            1,
            hidden_dims=critic_hidden_dims,
            activation=critic_activation,
        )
        return actor, critic_1, critic_2