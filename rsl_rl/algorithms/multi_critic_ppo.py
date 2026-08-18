from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from collections.abc import Sequence

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import MultiCriticRolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer

def _detach_hidden_state(hidden_state):
    """Recursively detach a (possibly nested) hidden-state structure."""
    if hidden_state is None:
        return None
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    if isinstance(hidden_state, (tuple, list)):
        return type(hidden_state)(_detach_hidden_state(h) for h in hidden_state)
    return hidden_state

class MultiCriticPPO:
    """Proximal Policy Optimization algorithm with multiple critics.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critics: Sequence[MLPModel]
    """The critic models."""

    def __init__(
        self,
        actor: MLPModel,
        critics: Sequence[MLPModel],
        storage: MultiCriticRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.rnd_target_critics: list[tuple[int, float]] = []
        if rnd_cfg:
            rnd_cfg = dict(rnd_cfg)  # copy — don't mutate caller's dict
            target_critics = rnd_cfg.pop("target_critics", None)
            if target_critics is None:
                # Backward-compatible default: previous hardcoded behavior.
                target_critics = {"critic_0": 1.0}

            critic_names = [f"critic_{i}" for i in range(len(critics))]
            for name, weight in target_critics.items():
                if name not in critic_names:
                    raise ValueError(
                        f"rnd_cfg['target_critics'] references {name!r}, which is not "
                        f"a known critic. Available critics: {critic_names}."
                    )
                self.rnd_target_critics.append((critic_names.index(name), float(weight)))

            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)

        # Symmetry augmentation is not implemented for multi-critic PPO
        if symmetry_cfg is not None:
            raise NotImplementedError("Symmetry augmentation is not supported for MultiCriticPPO.")


        # PPO components
        self.actor = actor.to(self.device)
        self.critics = [critic.to(self.device) for critic in critics]

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critics``.
        self._raw_actor = self.actor
        self._raw_critics = self.critics

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), *(critic.parameters() for critic in self.critics)), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = MultiCriticRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies: (actor, critic_0, ..., critic_{n-1})
        # Detached because these are only used to initialize truncated-BPTT segments during update(),
        # not to carry gradients across the whole rollout.
        self.transition.hidden_states = tuple(
            _detach_hidden_state(h)
            for h in (self.actor.get_hidden_state(), *(critic.get_hidden_state() for critic in self.critics))
        )
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = tuple(critic(obs).detach() for critic in self.critics)
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: tuple[torch.Tensor, ...], dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers.

        Args:
            rewards: One reward tensor per critic, each shape [num_envs]. Critics no longer share a
                single reward stream — each is free to optimize a distinct signal (e.g. task reward,
                energy penalty, smoothness penalty, ...).
        """
        # Update the normalizers
        self.actor.update_normalization(obs)
        for critic in self.critics:
            critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones (clone each critic's reward stream independently, since we
        # bootstrap on time-outs below)
        self.transition.rewards = tuple(r.clone() for r in rewards)
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to the *first* critic's extrinsic reward stream.
        # NOTE: RND was designed for a single shared reward; if you want intrinsic reward added to
        # every critic instead, decide that policy explicitly rather than defaulting silently.
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            rewards_list = list(self.transition.rewards)
            for idx, weight in self.rnd_target_critics:
                rewards_list[idx] = rewards_list[idx] + weight * self.intrinsic_rewards
            self.transition.rewards = tuple(rewards_list)

        # Bootstrapping on time outs — now per-critic, using each critic's own value estimate rather
        # than a mean across critics, since each critic now has its own reward scale/target.
        if "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards = tuple(
                r + self.gamma * torch.squeeze(v * time_outs, 1)
                for r, v in zip(self.transition.rewards, self.transition.values)  # type: ignore
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        for critic in self.critics:
            critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions, per critic."""
        st = self.storage
        last_values = []
        for critic in self.critics:
            hidden_state = critic.get_hidden_state()
            last_values.append(critic(obs).detach())
            critic.reset(hidden_state=hidden_state)
        last_values = tuple(last_values)

        advantage = tuple(torch.zeros_like(v) for v in last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = (
                last_values if step == st.num_transitions_per_env - 1 else tuple(v[step + 1] for v in st.values)
            )
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error per critic, now using that critic's OWN reward stream: st.rewards[c_idx][step]
            delta = tuple(
                st.rewards[c_idx][step] + next_is_not_terminal * self.gamma * next_v - values_c[step]
                for c_idx, (values_c, next_v) in enumerate(zip(st.values, next_values))
            )
            advantage = tuple(
                d + next_is_not_terminal * self.gamma * self.lam * adv for d, adv in zip(delta, advantage)
            )
            for c_idx, (adv_c, values_c) in enumerate(zip(advantage, st.values)):
                st.returns[c_idx][step] = adv_c + values_c[step]

        st.advantages = [ret - val for ret, val in zip(st.returns, st.values)]
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = [(adv - adv.mean()) / (adv.std() + 1e-8) for adv in st.advantages]
    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = None

        # Get mini-batch generator
        if self.actor.is_recurrent or any(critic.is_recurrent for critic in self.critics):
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = tuple(  # type: ignore
                        (adv - adv.mean()) / (adv.std() + 1e-8) for adv in batch.advantages  # type: ignore
                    )


            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters.
            # `batch.hidden_states` is only populated for recurrent models (see
            # MultiCriticRolloutStorage.recurrent_mini_batch_generator); the feedforward generator
            # leaves it as the empty-tuple default, so guard the indexing here.
            actor_hidden_state = batch.hidden_states[0] if batch.hidden_states else None
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=actor_hidden_state,
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore

            # Recompute values per critic, using each critic's own hidden state slot (if any)
            values = tuple(
                critic(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=(batch.hidden_states[c_idx + 1] if batch.hidden_states else None),
                )
                for c_idx, critic in enumerate(self.critics)
            )

            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss — one actor, but advantages differ per critic; average across critics into
            # a single scalar surrogate loss. If critics represent different reward components you want
            # weighted differently, replace the plain mean with a weighted sum here.
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate_losses = []
            for adv in batch.advantages:  # type: ignore
                surrogate = -torch.squeeze(adv) * ratio
                surrogate_clipped = -torch.squeeze(adv) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_losses.append(torch.max(surrogate, surrogate_clipped).mean())
            surrogate_loss = torch.stack(surrogate_losses).mean()

            # Value function loss — computed per critic, then averaged into a single scalar
            value_losses = []
            for values_c, batch_values_c, batch_returns_c in zip(values, batch.values, batch.returns):  # type: ignore
                if self.use_clipped_value_loss:
                    value_clipped = batch_values_c + (values_c - batch_values_c).clamp(
                        -self.clip_param, self.clip_param
                    )
                    vl = (values_c - batch_returns_c).pow(2)
                    vl_clipped = (value_clipped - batch_returns_c).pow(2)
                    value_losses.append(torch.max(vl, vl_clipped).mean())
                else:
                    value_losses.append((batch_returns_c - values_c).pow(2).mean())
            value_loss = torch.stack(value_losses).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            for critic in self.critics:
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            if mean_rnd_loss is not None :
                loss_dict["rnd"] = mean_rnd_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        for critic in self.critics:
            critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        for critic in self.critics:
            critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": [critic.state_dict() for critic in self._raw_critics],
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            for critic, state_dict in zip(self._raw_critics, loaded_dict["critic_state_dict"]):
                critic.load_state_dict(state_dict, strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critics with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critics = [compile_model(critic, mode) for critic in self._raw_critics]  # type: ignore

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> MultiCriticPPO:
        """Construct the MultiCriticPPO algorithm."""

        # ------------------------------------------------------------------
        # Resolve classes WITHOUT mutating cfg.
        #
        # Do not use .pop() here. The runner/tests may reuse the config,
        # and the class_name fields are still needed by the construction
        # logic or other code.
        # ------------------------------------------------------------------
        alg_class: type[MultiCriticPPO] = resolve_callable( #type: ignore
            cfg["algorithm"]["class_name"]
        )
        actor_class: type[MLPModel] = resolve_callable( #type: ignore
            cfg["actor"]["class_name"]
        )
        critic_class: type[MLPModel] = resolve_callable( #type: ignore
            cfg["critic"]["class_name"]
        )

        # ------------------------------------------------------------------
        # Number of critics
        # ------------------------------------------------------------------
        num_critics = cfg["algorithm"].get("num_critics", 1)

        # ------------------------------------------------------------------
        # Observation groups
        #
        # Actor:
        #   "actor"
        #
        # Critics:
        #   "critic_0"
        #   "critic_1"
        #   ...
        #
        # RND:
        #   "rnd_state"
        # ------------------------------------------------------------------
        default_sets = [
            "actor",
            *(f"critic_{i}" for i in range(num_critics)),
        ]

        if (
            "rnd_cfg" in cfg["algorithm"]
            and cfg["algorithm"]["rnd_cfg"] is not None
        ):
            default_sets.append("rnd_state")

        cfg["obs_groups"] = resolve_obs_groups(
            obs,
            cfg["obs_groups"],
            default_sets,
        )

        # ------------------------------------------------------------------
        # Resolve extensions
        # ------------------------------------------------------------------
        cfg["algorithm"] = resolve_rnd_config(
            cfg["algorithm"],
            obs,
            cfg["obs_groups"],
            env,
        )

        cfg["algorithm"] = resolve_symmetry_config(
            cfg["algorithm"],
            env,
        )

        # ------------------------------------------------------------------
        # Prepare model configs.
        #
        # IMPORTANT:
        # `class_name` is for resolve_callable(), NOT for MLPModel/RNNModel.
        # Remove it before using **cfg[...] in the constructors.
        # ------------------------------------------------------------------
        actor_cfg = {
            key: value
            for key, value in cfg["actor"].items()
            if key != "class_name"
        }

        critic_cfg = {
            key: value
            for key, value in cfg["critic"].items()
            if key != "class_name"
        }

        # ------------------------------------------------------------------
        # Initialize actor
        # ------------------------------------------------------------------
        actor: MLPModel = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            **actor_cfg,
        ).to(device)

        print(f"Actor Model: {actor}")

        # ------------------------------------------------------------------
        # Optionally share CNN encoders.
        #
        # This mutates critic_cfg rather than cfg["critic"], which avoids
        # accidentally modifying the original configuration.
        # ------------------------------------------------------------------
        if cfg["algorithm"].get("share_cnn_encoders", False):
            critic_cfg["cnns"] = actor.cnns  # type: ignore

        # ------------------------------------------------------------------
        # Initialize one critic per observation group.
        #
        # Critic 0 -> "critic_0"
        # Critic 1 -> "critic_1"
        # Critic 2 -> "critic_2"
        # ...
        # ------------------------------------------------------------------
        critics: list[MLPModel] = []

        for i in range(num_critics):
            critic_obs_group = f"critic_{i}"

            critic: MLPModel = critic_class(
                obs,
                cfg["obs_groups"],
                critic_obs_group,
                1,
                **critic_cfg,
            ).to(device)

            print(
                f"Critic {i} Model "
                f"(obs_group={critic_obs_group}): {critic}"
            )

            critics.append(critic)

        # ------------------------------------------------------------------
        # Initialize storage
        # ------------------------------------------------------------------
        storage = MultiCriticRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            num_critics=num_critics,
            device=device,
        )

        # ------------------------------------------------------------------
        # Prepare algorithm config.
        #
        # Remove fields that are constructor-selection metadata rather than
        # MultiCriticPPO.__init__ arguments.
        # ------------------------------------------------------------------
        algorithm_cfg = {
            key: value
            for key, value in cfg["algorithm"].items()
            if key not in {
                "class_name",
                "num_critics",
                "share_cnn_encoders",
            }
        }

        # ------------------------------------------------------------------
        # Initialize algorithm
        # ------------------------------------------------------------------
        alg: MultiCriticPPO = alg_class(
            actor,
            critics,
            storage,
            device=device,
            **algorithm_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # ------------------------------------------------------------------
        # Compile models if requested
        # ------------------------------------------------------------------
        alg.compile(
            cfg.get("torch_compile_mode")
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), *(critic.state_dict() for critic in self._raw_critics)]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        for critic, state_dict in zip(self._raw_critics, model_params[1 : 1 + len(self._raw_critics)]):
            critic.load_state_dict(state_dict)
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1 + len(self._raw_critics)])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), *(critic.parameters() for critic in self.critics))
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel