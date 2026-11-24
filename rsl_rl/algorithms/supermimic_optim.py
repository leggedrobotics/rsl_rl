# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic optimizer parameter selection, actor learning-rate multiplier and deployment actor EMA."""

from __future__ import annotations

import copy
import torch
import torch.distributed as dist
from torch import nn

from rsl_rl.models.supermimic_actor import SuperMimicActor
from rsl_rl.modules.mlp import MLP


def hidden_matrices(module: nn.Module) -> list[nn.Parameter]:
    """Return the weights of the linear layers between the first and last layer of each MLP in a module."""
    matrices = []
    for mlp in module.modules():
        if isinstance(mlp, MLP):
            linears = [layer for layer in mlp if isinstance(layer, nn.Linear)]
            matrices += [layer.weight for layer in linears[1:-1]]
    return matrices


class ActorLearningRateMultiplier:
    """Raise the actor learning rate early in training and decay it to its base value."""

    def __init__(
        self,
        multiplier: float,
        decay_iterations: int,
        decay_return: float | None,
        decay_start: int | None,
        max_episode_length: int | torch.Tensor | None,
        num_envs: int,
        device: str,
    ) -> None:
        """Validate the triggers and allocate device accumulators for episode returns."""
        if multiplier > 1 and decay_return is None and decay_start is None:
            raise ValueError("An actor learning-rate multiplier needs a decay return or a decay start iteration.")
        if decay_return is not None and max_episode_length is None:
            raise ValueError(
                "actor_learning_rate_decay_return needs the environment's max_episode_length to normalize returns."
            )
        self.multiplier = multiplier
        self.decay_iterations = decay_iterations
        self.decay_return = decay_return
        self.decay_start = decay_start
        self.max_episode_length = (
            max_episode_length.to(device) if isinstance(max_episode_length, torch.Tensor) else max_episode_length
        )

        self.decay_from: int | None = None if multiplier > 1 else 0
        self.episode_return = torch.zeros(num_envs, device=device)
        self.completed_episodes = torch.zeros(2, device=device)

    def record_rewards(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Accumulate the normalized returns of completed episodes using only device operations."""
        if self.decay_return is None:
            return
        self.episode_return.add_(rewards)
        normalized = torch.where(dones, self.episode_return / self.max_episode_length, 0.0)
        self.completed_episodes.add_(torch.stack((normalized.sum(), dones.sum())))
        self.episode_return.masked_fill_(dones, 0.0)

    def update_decay_from_returns(self, return_sum: float, episodes: float, iteration: int) -> None:
        """Start the decay once the reduced episodes completed since the previous update reach the return threshold."""
        self.completed_episodes.zero_()
        if (
            self.decay_from is None
            and self.decay_return is not None
            and episodes
            and return_sum / episodes >= self.decay_return
        ):
            self.decay_from = iteration

    def reset_episode_returns(self) -> None:
        """Discard the accumulated returns of completed and in-progress episodes."""
        self.episode_return.zero_()
        self.completed_episodes.zero_()

    def __call__(self, iteration: int) -> float:
        """Return the multiplier, starting its decay at the latest scheduled iteration."""
        if self.decay_from is None and self.decay_start is not None and iteration >= self.decay_start:
            self.decay_from = self.decay_start
        if self.decay_from is None:
            return self.multiplier
        progress = min((iteration - self.decay_from) / self.decay_iterations, 1.0)
        return self.multiplier + (1 - self.multiplier) * progress

    def state_dict(self) -> dict:
        """Return the decay start under its checkpoint key while a multiplier is active."""
        return {"actor_learning_rate_decay_from": self.decay_from} if self.multiplier > 1 else {}

    def load_state_dict(self, loaded_dict: dict) -> None:
        """Restore the decay start of an active multiplier."""
        if self.multiplier > 1:
            self.decay_from = loaded_dict.get("actor_learning_rate_decay_from")


class ActorEMA:
    """Maintain a deployment actor with exponentially averaged parameters."""

    def __init__(self, actor: SuperMimicActor, halflife: int) -> None:
        """Start the average at the current actor parameters."""
        self.actor = actor
        self.halflife = halflife
        self.averaged_parameters = [parameter.detach().clone() for parameter in actor.parameters()]
        self.num_updates = torch.zeros((), dtype=torch.long, device=self.averaged_parameters[0].device)
        self._policy: SuperMimicActor | None = None

    @torch.no_grad()
    def update(self, finite: torch.Tensor) -> None:
        """Average the actor after a successful step using the half-life and warm-start decay."""
        self.num_updates += finite.long()
        updates = self.num_updates.float()
        decay = torch.clamp((1 + updates) / (10 + updates), max=0.5 ** (1 / self.halflife))
        weight = (1 - decay) * finite.float()
        torch._foreach_lerp_(
            self.averaged_parameters, list(self.actor.parameters()), [weight] * len(self.averaged_parameters)
        )

    @torch.no_grad()
    def get_policy(self) -> SuperMimicActor:
        """Return a deployment actor with the averaged parameters and the live normalization and noise buffers.

        The deployment actor shares the training task until it moves to another device and takes its own copy.
        """
        if self._policy is None:
            with torch.inference_mode(False):
                self._policy = copy.deepcopy(self.actor, {id(self.actor.task): self.actor.task}).requires_grad_(False)
            self._policy.owns_task = False
        policy = self._policy.eval()
        for parameter, average in zip(policy.parameters(), self.averaged_parameters):
            parameter.copy_(average)
        for target, source in zip(policy.buffers(), self.actor.buffers()):
            target.copy_(source)
        return policy

    def state_dict(self) -> dict:
        """Return the deployment actor state and the number of averaged steps under their checkpoint keys."""
        averages = {
            name: average for (name, _), average in zip(self.actor.named_parameters(), self.averaged_parameters)
        }
        return {
            "actor_ema_state_dict": {**self.actor.state_dict(), **averages},
            "actor_ema_updates": int(self.num_updates),
        }

    def load_state_dict(self, loaded_dict: dict, strict: bool) -> None:
        """Restore the average, or restart it from the loaded live actor when the checkpoint has none."""
        policy = self.get_policy()
        policy.load_state_dict(loaded_dict.get("actor_ema_state_dict", self.actor.state_dict()), strict=strict)
        with torch.no_grad():
            for parameter, average in zip(policy.parameters(), self.averaged_parameters):
                average.copy_(parameter)
        self.num_updates.fill_(loaded_dict.get("actor_ema_updates", 0) if "actor_ema_state_dict" in loaded_dict else 0)

    @torch.no_grad()
    def broadcast(self) -> None:
        """Synchronize the average and its step count from rank 0."""
        for tensor in (*self.averaged_parameters, self.num_updates):
            dist.broadcast(tensor, src=0)
