# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Short-horizon actor-critic training with differentiable dynamics."""

from __future__ import annotations

import copy
import functools
import math
import torch
import torch.distributed as dist
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from tensordict import TensorDict
from torch import nn

from rsl_rl.env import VecEnv
from rsl_rl.env.differentiable_task import DifferentiableTask, concatenate_observation_set, prepare_actor_obs
from rsl_rl.extensions import LearnedDynamics
from rsl_rl.models import MLPModel, SuperMimicActor
from rsl_rl.utils import (
    CudaGraphCallable,
    Muon,
    compile_model,
    reduce_gradients_in_buckets,
    resolve_class,
    resolve_obs_groups,
)
from rsl_rl.utils.muon import select_hidden_weight_matrices
from rsl_rl.utils.utils import normalize_loss_across_ranks


def compute_lambda_returns(
    rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute TD(lambda) targets, retaining terminal rewards and the final bootstrap state."""
    target, targets = values[-1], []
    for index in reversed(range(rewards.shape[0])):
        continuation = (1 - lam) * values[index + 1] + lam * target
        target = rewards[index] + gamma * torch.where(dones[index] == 1, 0.0, (1 - dones[index]) * continuation)
        targets.append(target)
    return torch.stack(targets[::-1])


class SHAC:
    """Optimize short-horizon differentiable returns with a learned value bootstrap.

    Reference:
        Xu et al. "Accelerated Policy Learning with Parallel Differentiable Simulation." ICLR (2022).
    """

    def __init__(
        self,
        actor: MLPModel | SuperMimicActor,
        critic: MLPModel,
        task: DifferentiableTask,
        obs_groups: dict[str, list[str]],
        action_history: torch.Tensor,
        learned_dynamics: LearnedDynamics | None = None,
        # Update cadence parameters
        updates_per_step: float = 0.4,
        update_schedule: str = "after_warmup",
        # Policy parameters
        num_policy_updates: int = 1,
        horizon: int = 32,
        policy_batch_size: int = 4096,
        gamma: float = 0.99,
        max_step_reward: float | None = None,
        action_noise: float = 0.1,
        normalize_before_action: bool = True,
        use_reference_action: bool = False,
        actor_ema_halflife: int | None = None,
        # Critic parameters
        lam: float = 0.95,
        num_critic_epochs: int = 5,
        num_critic_mini_batches: int = 4,
        critic_survival_weighting: bool = True,
        critic_bootstrap_clipping: bool = True,
        critic_shuffle_each_epoch: bool = True,
        # Optimizer parameters
        learning_rate: float = 2e-4,
        actor_learning_rate: float | None = None,
        actor_learning_rate_multiplier: float = 1.0,
        actor_learning_rate_decay_iterations: int = 300,
        actor_learning_rate_decay_return: float | None = None,
        actor_learning_rate_decay_start: int | None = None,
        max_episode_length: int | torch.Tensor | None = None,
        muon_models: tuple[str, ...] = (),
        max_grad_norm: float = 1.0,
        # Learning rate schedule parameters
        num_schedule_updates: int = 250_000,
        final_learning_rate_ratio: float = 0.03,
        learning_rate_schedule_unit: str = "learner_updates",
        # Execution parameters
        use_mixed_precision: bool = False,
        use_cuda_graph: bool = False,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize short-horizon policy and value updates with the selected dynamics."""
        # Validate configuration
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        if self.is_multi_gpu:
            if not dist.is_initialized():
                raise ValueError("Distributed SHAC requires an initialized process group.")
            if dist.get_world_size() != self.gpu_world_size or dist.get_rank() != self.gpu_global_rank:
                raise ValueError("SHAC multi_gpu_cfg does not match the initialized process group.")
            if use_cuda_graph and dist.get_backend() != "nccl":
                raise ValueError("Distributed SHAC can only capture CUDA graphs with the NCCL backend.")
        if critic.is_recurrent:
            raise ValueError("SHAC requires a feed-forward critic.")

        if update_schedule not in ("after_warmup", "from_start"):
            raise ValueError("update_schedule must be 'after_warmup' or 'from_start'.")
        if learning_rate_schedule_unit not in ("learner_updates", "env_steps"):
            raise ValueError("learning_rate_schedule_unit must be 'learner_updates' or 'env_steps'.")
        if max_step_reward is not None and gamma == 1:
            raise ValueError("A reward-based value cap requires a discount below one.")
        if not set(muon_models) <= {"actor", "world_model"}:
            raise ValueError("muon_models may only contain 'actor' and 'world_model'.")

        if learned_dynamics is None and (
            type(task).start_rollout is DifferentiableTask.start_rollout or type(task).step is DifferentiableTask.step
        ):
            raise ValueError(
                "SHAC requires world_model_cfg or a differentiable task implementing start_rollout and step."
            )
        if use_reference_action and type(task).reference_action is DifferentiableTask.reference_action:
            raise ValueError("use_reference_action requires a task that implements reference_action.")

        if use_cuda_graph and torch.device(device).type != "cuda":
            raise ValueError("CUDA graphs require a CUDA device.")

        # Device-related parameters
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.use_cuda_graph = use_cuda_graph
        self._autocast = functools.partial(
            torch.autocast, torch.device(device).type, dtype=torch.bfloat16, enabled=use_mixed_precision
        )

        self.actor = actor.to(device)
        if isinstance(actor, SuperMimicActor):
            actor.export_reference_action = use_reference_action
        self.learned_dynamics = learned_dynamics
        self.critic = critic.to(device)
        self.task = task
        self.obs_groups = obs_groups
        self.action_history = action_history
        self.action_noise = action_noise
        self.normalize_before_action = normalize_before_action
        self.use_reference_action = use_reference_action

        # Observation groups read during collection, so captured collection calls receive fixed inputs
        def select_observation_keys(*names: str) -> list[str]:
            return list(dict.fromkeys(key for name in names for key in obs_groups.get(name, ())))

        self._actor_obs_keys = select_observation_keys(
            "actor", "state", "goal", "context", "prev_action", "prev_prev_action"
        )
        self._next_obs_keys = select_observation_keys("state", "prev_action", "prev_prev_action")
        self._final_obs_keys = select_observation_keys("state", "labels")

        self._act_graphs: dict[bool, CudaGraphCallable] = {}
        self._process_env_step_graphs: dict[bool, CudaGraphCallable] = {}
        self._update_graphs: dict[bool, CudaGraphCallable] = {}

        self._critic_forward = self.critic
        self._actor_forward = self.actor

        # Create the optimizers
        self._base_learning_rates = {
            "actor": learning_rate if actor_learning_rate is None else actor_learning_rate,
            "critic": learning_rate,
        }
        self._models: dict[str, nn.Module] = {"actor": self.actor, "critic": self.critic}
        if learned_dynamics is not None:
            self._models["world_model"] = learned_dynamics.world_model
            self._base_learning_rates["world_model"] = learning_rate
        elif "world_model" in muon_models:
            raise ValueError("A world model optimizer requires world_model_cfg.")
        fused = torch.device(device).type == "cuda"
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for name, module in self._models.items():
            matrices = select_hidden_weight_matrices(module) if name in muon_models else []
            if matrices:
                self.optimizers[f"{name}_muon"] = Muon(matrices, lr=self._base_learning_rates[name])
            muon_parameters = set(matrices)
            others = [parameter for parameter in module.parameters() if parameter not in muon_parameters]
            betas = (0.9, 0.95) if name == "actor" else (0.9, 0.999)
            self.optimizers[name] = torch.optim.Adam(
                others, lr=self._base_learning_rates[name], betas=betas, fused=fused
            )
        self._configure_optimizer_capture()

        # Update cadence parameters
        self.updates_per_step = updates_per_step
        self.update_schedule = update_schedule

        # Policy parameters
        self.num_policy_updates = num_policy_updates
        self.num_imagined_steps = horizon
        self.policy_batch_size = policy_batch_size
        self.gamma = gamma
        self.value_cap = None if max_step_reward is None else max_step_reward / (1 - gamma)
        self.max_grad_norm = max_grad_norm

        # Actor EMA parameters
        self.actor_ema = None if actor_ema_halflife is None else ActorEMA(self.actor, actor_ema_halflife)

        # Critic parameters
        self.lam = lam
        self.num_critic_epochs = num_critic_epochs
        self.num_critic_mini_batches = num_critic_mini_batches
        self.critic_survival_weighting = critic_survival_weighting
        self.critic_bootstrap_clipping = critic_bootstrap_clipping
        self.critic_shuffle_each_epoch = critic_shuffle_each_epoch

        # Learning rate schedule parameters
        self.num_schedule_updates = num_schedule_updates
        self.learning_rate_schedule_unit = learning_rate_schedule_unit
        self.final_learning_rate_ratio = final_learning_rate_ratio
        self.actor_learning_rate_multiplier = ActorLearningRateMultiplier(
            actor_learning_rate_multiplier,
            actor_learning_rate_decay_iterations,
            actor_learning_rate_decay_return,
            actor_learning_rate_decay_start,
            max_episode_length,
            action_history.shape[0],
            device,
        )

        # Initialize update counters and collection state
        self.num_learner_updates = 0
        self.num_env_steps = 0
        self._restore_update_credit()
        self._transition: TensorDict | None = None
        self._set_learning_rates()

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Record the physical start and sample an action from the collection policy."""
        obs = obs.select(*self._actor_obs_keys)
        act = self._get_graph_callable(self._act, self._act_graphs, self.actor.training)
        outputs = act(obs)
        env_actions = outputs.pop("env_actions")
        self._transition = TensorDict(outputs, batch_size=obs.batch_size)
        return env_actions

    def _act(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        if self.actor.training and self.normalize_before_action:
            self.actor.update_normalization(obs)
        actions = self.actor(obs).float()
        if self.action_noise:
            actions = actions + torch.randn_like(actions) * self.action_noise
        if self.actor.training and not self.normalize_before_action:
            self.actor.update_normalization(obs)

        transition = {
            name: concatenate_observation_set(obs, self.obs_groups, name) for name in ("state", "goal", "context")
        }
        applied = actions
        if self.use_reference_action:
            state = self.task.representation.unflatten(transition["state"])
            applied = actions + self.task.reference_action(state, transition["goal"])
        for name, offset in (("prev_action", -1), ("prev_prev_action", -2)):
            if name in self.obs_groups:
                self.action_history[:, offset].copy_(concatenate_observation_set(obs, self.obs_groups, name))
        return {**transition, "env_actions": actions, "actions": applied, "action_history": self.action_history.clone()}

    @torch.no_grad()
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> dict[str, torch.Tensor]:
        """Record physical successors captured before resets or interval edits on every step."""
        if self._transition is None:
            raise RuntimeError("act() must precede process_env_step().")
        if self.learned_dynamics is not None and (
            extras.get("final_obs") is None or extras.get("final_obs_every_step") is not True
        ):
            raise ValueError("Learned dynamics requires every-step extras['final_obs'] with final_obs_every_step=True.")
        step_rewards = rewards.reshape(-1).to(self.device)
        step_dones = dones.reshape(-1).to(self.device).bool()
        process = self._get_graph_callable(self._process_env_step, self._process_env_step_graphs, self.actor.training)
        process(
            self._transition,
            obs.select(*self._next_obs_keys).to(self.device),
            extras["final_obs"].select(*self._final_obs_keys).to(self.device)
            if self.learned_dynamics is not None
            else obs.select(),
            step_dones,
            step_rewards,
        )
        if self.learned_dynamics is not None:
            self.learned_dynamics.advance()
        self.num_env_steps += 1

        replay_ready = self.learned_dynamics is None or self.learned_dynamics.ready
        if replay_ready or self.update_schedule == "from_start":
            self._update_credit += self.updates_per_step
            if not replay_ready:
                self._update_credit -= max(int(self._update_credit + 1e-9), 0)
        self.task.update_curriculum(self._transition["goal"], step_rewards, step_dones, extras)

        self._transition = None
        return {"Training/mean_reward": rewards}

    def compute_returns(self, obs: TensorDict) -> None:
        """Replay and imagination construct their own return targets during update."""
        pass

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run accumulated learner updates and return mean losses and statistics."""
        batches = [self.learned_dynamics.collection_metrics()] if self.learned_dynamics is not None else []
        self.actor_learning_rate_multiplier.update_decay_from_returns(self.num_learner_updates, self.is_multi_gpu)

        num_updates, cumulative_counts = self._consume_update_credit()
        for _ in range(num_updates):
            if self.learning_rate_schedule_unit == "env_steps":
                self._set_learning_rates()

            update_policy = (
                self.learned_dynamics is None or self.num_learner_updates >= self.learned_dynamics.num_warmup_iterations
            )
            if self.use_cuda_graph and update_policy:
                self._update_graphs.pop(False, None)
            update = self._get_graph_callable(
                functools.partial(self._run_learner_update, update_policy), self._update_graphs, update_policy
            )
            batches.append(update() if cumulative_counts is None else update(cumulative_counts))
            self.num_learner_updates += 1
            if self.learning_rate_schedule_unit == "learner_updates":
                self._set_learning_rates()

        metrics = {"learner_updates": float(self.num_learner_updates)}
        averages = _mean_metrics(batches)
        if averages:
            means = torch.stack(list(averages.values()))
            if self.is_multi_gpu:
                dist.all_reduce(means)
                means /= self.gpu_world_size
            metrics.update(zip(averages, means.tolist()))

        losses = {key.removesuffix("_loss"): value for key, value in metrics.items() if key.endswith("_loss")}
        stats = {f"SHAC/{key}": value for key, value in metrics.items() if not key.endswith("_loss")}
        stats["Optimizer/learning_rate"] = self.learning_rate
        stats["Optimizer/actor_learning_rate"] = self.actor_learning_rate
        return losses, stats

    def train_mode(self) -> None:
        """Enable training and collection of normalization statistics."""
        for model in self._models.values():
            model.train()

    def eval_mode(self) -> None:
        """Select deterministic deployment inference without updating actor moments."""
        for model in self._models.values():
            model.eval()

    def save(self) -> dict:
        """Save models, optimizers and training counters."""
        saved = {
            **{f"{name}_state_dict": model.state_dict() for name, model in self._models.items()},
            **{f"{key}_optimizer_state_dict": optimizer.state_dict() for key, optimizer in self.optimizers.items()},
            "iteration": self.num_learner_updates,
            "env_steps": self.num_env_steps,
        }
        saved.update(self.actor_learning_rate_multiplier.state_dict())
        if self.actor_ema is not None:
            saved.update(self.actor_ema.state_dict())
        return saved

    def get_default_load_cfg(self) -> dict:
        """Return the default selection for resuming this algorithm's own checkpoints, including the world model."""
        return dict.fromkeys((*self._models, "optimizer", "iteration"), True)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore a deployment actor or resume all configured models."""
        load_cfg = self.get_default_load_cfg() if load_cfg is None else load_cfg
        core = {name for name in ("actor", "critic", "world_model") if load_cfg.get(name)}
        if core not in ({"actor"}, self._models.keys()):
            raise ValueError("SHAC supports actor-only or full model loading.")
        if core == {"actor"} and any(load_cfg.get(key) for key in ("optimizer", "iteration")):
            raise ValueError("Actor-only loading cannot restore optimizer or iteration state.")
        if "world_model" in core and "world_model_state_dict" not in loaded_dict:
            raise ValueError("A full SHAC checkpoint must explicitly contain world_model.")

        for name, model in self._models.items():
            if name in core:
                model.load_state_dict(loaded_dict[f"{name}_state_dict"], strict=strict)
        if self.actor_ema is not None:
            self.actor_ema.load_state_dict(loaded_dict, strict)
        elif "actor_ema_state_dict" in loaded_dict:
            warnings.warn("Checkpoint contains an actor EMA; set actor_ema_halflife to deploy it.", stacklevel=2)

        if load_cfg.get("optimizer"):
            for name, optimizer in self.optimizers.items():
                optimizer.load_state_dict(loaded_dict[f"{name}_optimizer_state_dict"])
            self._configure_optimizer_capture()
        if load_cfg.get("iteration"):
            self.num_learner_updates = loaded_dict["iteration"]
            self.actor_learning_rate_multiplier.load_state_dict(loaded_dict)
            self.num_env_steps = loaded_dict.get("env_steps", 0)

        if self.learned_dynamics is not None:
            self.learned_dynamics.clear()
        self.action_history.zero_()
        self._transition = None
        self._restore_update_credit()
        self.actor_learning_rate_multiplier.reset_episode_returns()

        self._set_learning_rates()
        return bool(load_cfg.get("iteration"))

    def get_policy(self) -> MLPModel | SuperMimicActor:
        """Return the actor with its live normalization and optional averaged weights."""
        return self.actor if self.actor_ema is None else self.actor_ema.get_policy()

    def compile(self, mode: str | None = None) -> None:
        """Compile differentiable dynamics and stateless actor forwards using the standard mode restrictions."""
        if self.learned_dynamics is not None:
            self.learned_dynamics.model_forward = compile_model(self.learned_dynamics.world_model, mode)
        self._critic_forward = compile_model(self.critic, mode)
        self._actor_forward = compile_model(self.actor, mode)

    @staticmethod
    @torch.no_grad()
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SHAC:
        """Resolve the task and explicit network configs, inferring synthetic feature widths once."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        task_class, task_cfg = resolve_class(alg_cfg.pop("task_cfg"))
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        world_model_cfg = alg_cfg.pop("world_model_cfg", None)
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        if "critic" in cfg["obs_groups"]:
            raise ValueError("SHAC critic observations are synthetic; omit 'critic' from obs_groups.")
        allowed = ("actor", "state", "goal", "context", "labels", "prev_action", "prev_prev_action")
        groups = cfg["obs_groups"].copy()
        if "actor" not in groups and "actor" not in obs and "policy" in obs:
            groups["actor"] = ["policy"]
        obs_groups = resolve_obs_groups(obs, groups, ["state", "actor"], fallback_to_policy=False)
        if set(obs_groups) - set(allowed):
            raise ValueError("Unknown SHAC observation set.")
        cfg["obs_groups"] = obs_groups
        example = obs.to(device)
        if any(example[key].ndim != 2 for keys in obs_groups.values() for key in keys):
            raise ValueError("SHAC observation groups must be flat vectors.")

        task = task_class(env=env, device=device, **task_cfg)
        read_example_set = functools.partial(concatenate_observation_set, example, obs_groups)
        action_history = torch.zeros(env.num_envs, task.action_history_length, env.num_actions, device=device)
        for name, offset in (("prev_action", -1), ("prev_prev_action", -2)):
            if name in obs_groups:
                action = read_example_set(name)
                if action.shape[-1] != env.num_actions or task.action_history_length < -offset:
                    raise ValueError("Previous-action observation width/history is incompatible.")
                action_history[:, offset] = action
        if world_model_cfg is not None and read_example_set("labels").shape[-1] != task.num_labels:
            raise ValueError("Label observation width must equal task.num_labels.")

        state = task.representation.unflatten(read_example_set("state"))
        goal = read_example_set("goal")
        context = read_example_set("context")
        observations = task.actor_obs(state, goal, context, action_history)
        if "policy" not in observations or "history" in observations:
            raise ValueError("actor_obs must supply a policy frame and reserve history for SHAC.")
        history_length = alg_cfg.pop("history_length", 1)
        if history_length < 1:
            raise ValueError("history_length must be positive.")
        obs_history = observations["policy"][:, None].repeat(1, history_length, 1)
        inputs = prepare_actor_obs(observations, obs_history)
        for key in obs_groups["actor"]:
            if key not in inputs or inputs[key].shape != example[key].shape:
                raise ValueError(f"Imagined actor observation '{key}' must match its environment observation shape.")
        actor = actor_class(example, obs_groups, "actor", env.num_actions, **actor_cfg)
        if actor.is_recurrent:
            raise ValueError("SHAC requires a feed-forward actor.")
        critic_features = task.critic_obs(state, goal, context, action_history, obs_history)
        learned_dynamics = None
        if world_model_cfg is not None:
            learned_dynamics = LearnedDynamics(
                task,
                obs_groups,
                state,
                context,
                action_history,
                history_length,
                use_mixed_precision=alg_cfg.get("use_mixed_precision", False),
                world_size=cfg["multi_gpu"]["world_size"] if cfg["multi_gpu"] else 1,
                device=device,
                **world_model_cfg,
            )
        critic_obs = TensorDict({"critic": critic_features}, batch_size=[env.num_envs])
        critic = critic_class(critic_obs, {"critic": ["critic"]}, "critic", 1, **critic_cfg)

        learner = alg_class(
            actor,
            critic,
            task,
            obs_groups,
            action_history,
            learned_dynamics=learned_dynamics,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            max_episode_length=(
                getattr(env, "max_episode_length", None)
                if alg_cfg.get("actor_learning_rate_decay_return") is not None
                else None
            ),
            **alg_cfg,
        )
        learner.compile(cfg.get("torch_compile_mode"))
        return learner

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        """Synchronize learned model state while preserving each rank's local collection histories."""
        if not self.is_multi_gpu:
            return

        for model in self._models.values():
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)
        if self.actor_ema is not None:
            self.actor_ema.broadcast_parameters()

    def release_cuda_graphs(self) -> None:
        """Release graphs containing NCCL collectives before process-group teardown."""
        self._act_graphs.clear()
        self._process_env_step_graphs.clear()
        self._update_graphs.clear()

        if self.use_cuda_graph:
            torch.cuda.synchronize(self.device)

    def _get_graph_callable(self, function: Callable, graphs: dict[bool, CudaGraphCallable], phase: bool) -> Callable:
        """Select the eager function or its CUDA graph for the current training phase."""
        if not self.use_cuda_graph:
            return function
        if phase not in graphs:
            graphs[phase] = CudaGraphCallable(function, self.device)
        return graphs[phase]

    def _process_env_step(
        self,
        transition: TensorDict,
        obs: TensorDict,
        final: TensorDict,
        done: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Store the transition, update dynamics statistics and reset finished histories on the device."""
        if self.learned_dynamics is not None:
            self.learned_dynamics.observe(transition, obs, final, done)
        self.actor_learning_rate_multiplier.record_rewards(rewards, done)
        groups = self.obs_groups
        reset_action = (
            concatenate_observation_set(obs, groups, "prev_action")[:, None] if "prev_action" in groups else 0.0
        )
        shifted = torch.cat((transition["action_history"][:, 1:], transition["actions"][:, None]), dim=1)
        self.action_history.copy_(torch.where(done[:, None, None], reset_action, shifted))
        if "prev_prev_action" in groups:
            previous = concatenate_observation_set(obs, groups, "prev_prev_action")
            self.action_history[:, -2].copy_(torch.where(done[:, None], previous, self.action_history[:, -2]))
        return {}

    def _consume_update_credit(self) -> tuple[int, torch.Tensor | None]:
        """Consume accrued updates and prepare complete windows when every rank can train."""
        replay_ready = self.learned_dynamics is None or self.learned_dynamics.ready
        num_updates = max(int(self._update_credit + 1e-9), 0) if replay_ready else 0
        self._update_credit -= num_updates

        cumulative_counts = None
        if num_updates and self.learned_dynamics is not None:
            cumulative_counts = self.learned_dynamics.complete_window_counts()

        if self.is_multi_gpu or cumulative_counts is not None:
            ready_count = torch.tensor(num_updates, device=self.device)
            if cumulative_counts is not None:
                ready_count *= cumulative_counts[-1] > 0
            if self.is_multi_gpu:
                dist.all_reduce(ready_count, op=dist.ReduceOp.MIN)
            num_updates = int(ready_count)

        return num_updates, cumulative_counts

    def _restore_update_credit(self) -> None:
        """Restore the fractional update credit that collection would have accrued over ``env_steps``.

        The from-start schedule begins with ``1 - updates_per_step`` credit, so the first collected step yields one
        update; whole updates already ran, so only the fraction of the accrued credit remains.
        """
        if self.update_schedule == "after_warmup":
            self._update_credit = 0.0
            return

        accrued = 1.0 - self.updates_per_step + self.num_env_steps * self.updates_per_step
        self._update_credit = accrued - int(accrued + 1e-9) if self.num_env_steps else accrued

    def _run_learner_update(
        self, update_policy: bool, cumulative_counts: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Run one learner iteration and return the mean of each metric over its updates."""
        batches = []
        dynamics = self.learned_dynamics
        if dynamics is not None:
            for _ in range(dynamics.num_updates):
                loss, metrics = dynamics.compute_loss(cumulative_counts)
                metrics["world_model_grad_norm"] = self._step_model_optimizers(
                    loss, "world_model", dynamics.world_model
                )
                batches.append(metrics)

        if update_policy:
            batches.extend(self._update_actor_and_critic() for _ in range(self.num_policy_updates))

        return _mean_metrics(batches)

    def _update_actor_and_critic(self) -> dict[str, torch.Tensor]:
        """Backpropagate imagined returns through frozen dynamics and fit a TD(lambda) critic."""
        dynamics = self.learned_dynamics if self.learned_dynamics is not None else self.task
        with torch.no_grad():
            rollout = dynamics.start_rollout(self.policy_batch_size)

        imagined_rewards, termination_probs, critic_features, reach_probs = [], [], [], []
        survival = torch.ones(self.policy_batch_size, device=self.device)
        reachable = survival > 0

        with _freeze_parameters(*(model for name, model in self._models.items() if name != "actor")):
            for _ in range(self.num_imagined_steps):
                reach_probs.append(survival)
                critic_features.append(
                    self.task.critic_obs(
                        rollout.state, rollout.goal, rollout.context, rollout.action_history, rollout.obs_history
                    )
                )
                with self._autocast():
                    action = self._actor_forward(rollout.actor_obs).float()
                if self.use_reference_action:
                    action = action + self.task.reference_action(rollout.state, rollout.goal)

                rollout, reward, terminal = dynamics.step(rollout, action, survival)
                imagined_rewards.append(torch.where(reachable, reward, 0.0))
                terminal = torch.where(reachable, terminal, 1.0)
                termination_probs.append(terminal)
                survival = survival * (1 - terminal)
                reachable = survival > 0

            critic_features.append(
                self.task.critic_obs(
                    rollout.state, rollout.goal, rollout.context, rollout.action_history, rollout.obs_history
                )
            )
            rewards = torch.stack(imagined_rewards)
            terminals = torch.stack(termination_probs)
            reach = torch.stack(reach_probs)
            weights = reach * self.gamma ** torch.arange(self.num_imagined_steps, device=self.device)[:, None]
            value = self._predict_values(critic_features[-1])
            bootstrap = torch.where(reachable, self._cap_values(value), 0.0)
            objective = (weights * rewards).sum(dim=0) + self.gamma**self.num_imagined_steps * survival * bootstrap
            policy_loss = -objective.mean()

            gradient = self._step_model_optimizers(policy_loss, "actor", self.actor, clip_value=0.1)

        features = torch.stack(critic_features).detach()
        metrics = {
            "policy_loss": policy_loss.detach(),
            "policy_grad_norm": gradient,
            "imagined_survival": survival.mean().detach(),
            "imagined_reward": rewards.detach().sum() / (reach > 0).sum(),
        }
        if self.value_cap is not None:
            metrics["terminal_saturation"] = ((value.detach() > self.value_cap) & reachable).sum() / (
                reachable.sum().clamp_min(1)
            )
        metrics.update(self._update_critic(features, rewards.detach(), terminals.detach(), reach.detach()))
        return metrics

    def _predict_values(self, features: torch.Tensor) -> torch.Tensor:
        """Read scalar values through the standard synthetic critic observation set."""
        return self._critic_forward(TensorDict({"critic": features}, batch_size=features.shape[:-1])).squeeze(-1)

    def _cap_values(self, values: torch.Tensor) -> torch.Tensor:
        """Clamp values above ``value_cap`` when set; there is no lower bound."""
        return values if self.value_cap is None else values.clamp(max=self.value_cap)

    def _update_critic(
        self, features: torch.Tensor, rewards: torch.Tensor, terminals: torch.Tensor, reach: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Fit scalar values on detached imagined features with TD(lambda) targets.

        ``reach`` is the probability of reaching each imagined state without an earlier termination. The loss is
        weighted by it, and the normalizer is updated with the replay start states only, so states after a
        termination influence neither.
        """
        with torch.no_grad():
            replay_start_features = features[0]
            self.critic.update_normalization(
                TensorDict({"critic": replay_start_features}, batch_size=replay_start_features.shape[:1])
            )

            reachable = reach > 0
            final_reachable = reachable[-1:] & (terminals[-1:] < 1)
            reachable_states = torch.cat((reachable, final_reachable))
            features = torch.where(reachable_states[..., None], features, features[:1])
            values = self._predict_values(features)
            if self.critic_bootstrap_clipping:
                values = self._cap_values(values)

            targets = compute_lambda_returns(rewards, terminals, values, self.gamma, self.lam)
            targets = torch.where(reachable, self._cap_values(targets), 0.0).flatten()

        features = features[:-1].flatten(0, 1)
        if not self.critic_survival_weighting:
            reach = reachable.to(reach.dtype)
        reach = reach.flatten()
        num_samples = features.shape[0]

        losses = []
        permutation = None
        for _ in range(self.num_critic_epochs):
            if self.critic_shuffle_each_epoch:
                permutation = torch.rand(num_samples, device=self.device).argsort()
            elif permutation is None:
                num_batch_samples = num_samples // self.num_critic_mini_batches * self.num_critic_mini_batches
                permutation = torch.randperm(num_samples, device=self.device)[:num_batch_samples]

            for indices in permutation.chunk(self.num_critic_mini_batches):
                with self._autocast():
                    squared_errors = (self._predict_values(features[indices]) - targets[indices]).square()
                    loss = normalize_loss_across_ranks(
                        (reach[indices] * squared_errors).sum(), reach[indices].sum(), self.gpu_world_size, 1e-6
                    )
                self._step_model_optimizers(loss, "critic", self.critic)
                losses.append(loss.detach())

        return {"critic_loss": torch.stack(losses).mean()}

    def _step_model_optimizers(
        self, loss: torch.Tensor, model_name: str, module: nn.Module, clip_value: float | None = None
    ) -> torch.Tensor:
        """Apply finite gradients with optional value clipping.

        Fused optimizers skip nonfinite steps on the device; other optimizers check the gradient norm on the host.
        """
        adam = self.optimizers[model_name]
        muon = self.optimizers.get(f"{model_name}_muon")
        optimizers = (adam,) if muon is None else (adam, muon)
        module.zero_grad(set_to_none=True)
        loss.backward()
        if self.is_multi_gpu:
            reduce_gradients_in_buckets(module.parameters(), self.gpu_world_size, bucket_mb=25.0)

        grad_norm = nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
        if clip_value is not None:
            nn.utils.clip_grad_value_(module.parameters(), clip_value)
        finite = torch.isfinite(grad_norm)

        if adam.defaults.get("fused"):
            for optimizer in optimizers:
                optimizer.found_inf = (~finite).float()
                optimizer.step()
        elif finite:
            for optimizer in optimizers:
                optimizer.step()

        if model_name == "actor" and self.actor_ema is not None:
            self.actor_ema.update_parameters(finite)
        return grad_norm.detach()

    def _set_learning_rates(self) -> None:
        """Set all optimizer rates for the next learner iteration, clamping at the cosine floor."""
        step = (
            max(self.num_env_steps - 1, 0)
            if self.learning_rate_schedule_unit == "env_steps"
            else self.num_learner_updates
        )
        progress = min(step / self.num_schedule_updates, 1.0)
        factor = self.final_learning_rate_ratio + (1 - self.final_learning_rate_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

        rates = {name: base * factor for name, base in self._base_learning_rates.items()}
        rates["actor"] *= self.actor_learning_rate_multiplier(self.num_learner_updates)
        self.learning_rate = rates["critic"]
        self.actor_learning_rate = rates["actor"]

        for name, optimizer in self.optimizers.items():
            rate = rates[name.removesuffix("_muon")]
            for group in optimizer.param_groups:
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"].fill_(rate)
                else:
                    group["lr"] = rate

    def _configure_optimizer_capture(self) -> None:
        """Configure optimizers for the learner device and CUDA graphs, and discard stale graphs.

        Loaded optimizer states may come from another device, so their fused flags and Adam step counters are reset to
        this learner's configuration. Graphs need capturable optimizers with device learning rates.
        """
        for optimizer in self.optimizers.values():
            fused = bool(optimizer.defaults.get("fused"))
            for group in optimizer.param_groups:
                lr = float(group["lr"])
                group["lr"] = torch.tensor(lr, device=self.device) if self.use_cuda_graph else lr
                group["capturable"] = self.use_cuda_graph
                if "fused" in group:
                    group["fused"] = fused
                for param in group["params"]:
                    state = optimizer.state.get(param, {})
                    if "step" in state:
                        state["step"] = state["step"].to(param.device if fused or self.use_cuda_graph else "cpu")

        self._update_graphs.clear()


def _mean_metrics(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average each metric over the updates that reported it."""
    totals: dict[str, list[torch.Tensor]] = {}
    for batch in batches:
        for key, value in batch.items():
            totals.setdefault(key, []).append(value)
    return {key: torch.stack(values).float().mean() for key, values in totals.items()}


@contextmanager
def _freeze_parameters(*modules: nn.Module) -> Generator[None, None, None]:
    """Freeze parameters while preserving input derivatives and restoring flags after failures."""
    grad_flags = [(parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()]
    try:
        for parameter, _ in grad_flags:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in grad_flags:
            parameter.requires_grad_(enabled)


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
        self.decay_return_threshold = decay_return
        self.scheduled_decay_start_update = decay_start
        self.max_episode_length = (
            max_episode_length.to(device) if isinstance(max_episode_length, torch.Tensor) else max_episode_length
        )

        self.decay_start_update: int | None = None if multiplier > 1 else 0
        self.episode_returns = torch.zeros(num_envs, device=device)
        self.completed_return_stats = torch.zeros(2, device=device)

    def record_rewards(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Accumulate the normalized returns of completed episodes using only device operations."""
        if self.decay_return_threshold is None:
            return

        self.episode_returns.add_(rewards)
        normalized_returns = torch.where(dones, self.episode_returns / self.max_episode_length, 0.0)
        self.completed_return_stats.add_(torch.stack((normalized_returns.sum(), dones.sum())))
        self.episode_returns.masked_fill_(dones, 0.0)

    def update_decay_from_returns(self, num_learner_updates: int, is_multi_gpu: bool) -> None:
        """Start the decay once the mean completed episode return reaches its threshold."""
        if self.decay_return_threshold is None:
            return
        if is_multi_gpu:
            dist.all_reduce(self.completed_return_stats)
        normalized_return_sum, num_completed_episodes = self.completed_return_stats.tolist()
        self.completed_return_stats.zero_()
        if (
            self.decay_start_update is None
            and num_completed_episodes
            and normalized_return_sum / num_completed_episodes >= self.decay_return_threshold
        ):
            self.decay_start_update = num_learner_updates

    def reset_episode_returns(self) -> None:
        """Discard the accumulated returns of completed and in-progress episodes."""
        self.episode_returns.zero_()
        self.completed_return_stats.zero_()

    def __call__(self, num_learner_updates: int) -> float:
        """Return the multiplier, starting its decay at the latest scheduled iteration."""
        scheduled_start = self.scheduled_decay_start_update
        if self.decay_start_update is None and scheduled_start is not None and num_learner_updates >= scheduled_start:
            self.decay_start_update = scheduled_start
        if self.decay_start_update is None:
            return self.multiplier

        progress = min((num_learner_updates - self.decay_start_update) / self.decay_iterations, 1.0)
        return self.multiplier + (1 - self.multiplier) * progress

    def state_dict(self) -> dict:
        """Return the decay start under its checkpoint key while a multiplier is active."""
        return {"actor_learning_rate_decay_from": self.decay_start_update} if self.multiplier > 1 else {}

    def load_state_dict(self, loaded_dict: dict) -> None:
        """Restore the decay start of an active multiplier."""
        if self.multiplier > 1:
            self.decay_start_update = loaded_dict.get("actor_learning_rate_decay_from")


class ActorEMA:
    """Maintain a deployment actor with exponentially averaged parameters."""

    def __init__(self, actor: MLPModel | SuperMimicActor, halflife: int) -> None:
        """Start the average at the current actor parameters."""
        self.actor = actor
        self.halflife = halflife

        self.averaged_parameters = [parameter.detach().clone() for parameter in actor.parameters()]
        self.num_updates = torch.zeros((), dtype=torch.long, device=self.averaged_parameters[0].device)
        self._inference_policy: MLPModel | SuperMimicActor | None = None

    @torch.no_grad()
    def update_parameters(self, finite: torch.Tensor) -> None:
        """Average the actor after a successful step using the half-life and warm-start decay."""
        self.num_updates += finite.long()
        updates = self.num_updates.float()
        decay = torch.clamp((1 + updates) / (10 + updates), max=0.5 ** (1 / self.halflife))
        weight = (1 - decay) * finite.float()

        torch._foreach_lerp_(
            self.averaged_parameters, list(self.actor.parameters()), [weight] * len(self.averaged_parameters)
        )

    @torch.no_grad()
    def get_policy(self) -> MLPModel | SuperMimicActor:
        """Return the averaged parameters with the live actor's normalization buffers."""
        if self._inference_policy is None:
            with torch.inference_mode(False):
                self._inference_policy = copy.deepcopy(self.actor).requires_grad_(False)

        policy = self._inference_policy.eval()
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
    def broadcast_parameters(self) -> None:
        """Synchronize the average and its step count from rank 0."""
        for tensor in (*self.averaged_parameters, self.num_updates):
            dist.broadcast(tensor, src=0)
