from __future__ import annotations

from typing import Callable

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import robot_rl
from robot_rl.algorithms import ProbeAlg
from robot_rl.env import VecEnv
from robot_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticEstimator,
    EmpiricalNormalization,
    Probe,
    SAE,
)
from robot_rl.utils import store_code_state

from robot_rl.utils import (
    RobotRlProbeRunner,
    ProbeAlgorithmCfg,
    ProbeCfg,
    RobotRlActorCriticCfg,
)


class ProbeRunner:
    def __init__(
        self,
        env: VecEnv,
        train_cfg,
        log_dir=None,
        device="cpu",
    ) -> None:
        self.cfg: dict = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.probe_cfg = train_cfg["probe"]
        self.probe_type = "sae" if self.probe_cfg["class_name"] == "SAE" else "linear"
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        if self.probe_type == "linear":
            probe_obs = extras["observations"].get("probe")
            if probe_obs is None:
                if self.probe_type == "linear":
                    raise KeyError("No probe observations found in infos['observations'].")
                else:
                    probe_obs = obs
            num_probe_obs = probe_obs.shape[1]
            num_probe_obs_all = num_probe_obs
            self.probe_idx = [0, num_probe_obs]
            self.probe_obs_bool = self.cfg["probe_obs"]
            self.probe_next_obs_bool = self.cfg["probe_next_obs"]
            if self.probe_obs_bool:
                num_probe_obs_all += num_obs
                self.probe_idx.append(num_probe_obs_all)
            if self.probe_next_obs_bool:
                num_probe_obs_all += num_obs
                self.probe_idx.append(num_probe_obs_all)
        else:
            self.probe_idx = []
            num_probe_obs_all = 0
            self.probe_obs_bool = False
            self.probe_next_obs_bool = False
        if self.policy_cfg["class_name"] == "ActorCriticEstimator":
            num_est_obs = extras["observations"]["estimator"].shape[1]
            if self.alg_cfg["estimate_obs"]:
                num_est_obs += num_obs
            if self.alg_cfg["estimate_next_obs"]:
                num_est_obs += num_obs
            self.policy_cfg["num_estimates"] = num_est_obs
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstimator = policy_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        probe_class: Probe | SAE = eval(self.probe_cfg.pop("class_name"))
        policy_layers = []
        for l in self.policy_cfg["actor_hidden_dims"]:
            policy_layers.append(l)
            policy_layers.append(l)
        if not self.policy_cfg["oracle"]:
            policy_layers.append(self.env.num_actions)
        else:
            policy_layers.append(policy.num_estimates)
        probes = {
            str(l): probe_class(input_dim=policy_layers[l], output_dim=num_probe_obs_all, **self.probe_cfg)
            for l in self.cfg["layers"]
        }
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: ProbeAlg = alg_class(
            policy,
            probes,
            probe_type=self.probe_type,
            device=self.device,
            oracle=self.policy_cfg["oracle"],
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
        else:
            # no normalization
            self.obs_normalizer = torch.nn.Identity().to(self.device)
        # Probes always normalized for even scaling
        self.probe_obs_normalizer = EmpiricalNormalization(shape=[num_probe_obs_all], until=1.0e8).to(self.device)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [robot_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from robot_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from robot_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        probe_obs = extras["observations"].get("probe", obs).to(self.device)
        obs = obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        lenbuffer = deque(maxlen=100)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                actions = self.alg.act(obs)
                if self.policy_cfg["oracle"]:
                    estimates = self.alg.estimate(obs)  # for hooks
                latents = self.alg.extract_latents()
                next_obs, _, dones, infos = self.env.step(actions)
            next_obs = next_obs.to(self.device)
            next_obs = self.obs_normalizer(next_obs)
            dones_bool = dones.view(-1, 1) > 0
            if self.probe_type == "linear":
                probe_obs = infos["observations"]["probe"].to(self.device)
                probe_obs_all = probe_obs
                if self.probe_obs_bool:
                    probe_obs_all = torch.cat((probe_obs_all, obs), dim=1)
                if self.probe_next_obs_bool:
                    probe_obs_all = torch.cat((probe_obs_all, next_obs), dim=1)
                probe_obs_all = self.probe_obs_normalizer(probe_obs_all)
            else:
                probe_obs_all = None
            if self.log_dir is not None:
                if "episode" in infos:
                    ep_infos.append(infos["episode"])
                elif "log" in infos:
                    ep_infos.append(infos["log"])
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_episode_length[new_ids] = 0
            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            loss_dict, error_dict = self.alg.update(
                latents,
                dones_bool,
                probe_obs_all,
                self.probe_idx,
            )
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter and self.cfg.get("store_code_state", True):
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
            obs = next_obs

        self.save(os.path.join(self.log_dir, f"probe_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if locs["error_dict"]:
            for probe_layer, errors in locs["error_dict"].items():
                for error_name, error in errors.items():
                    self.writer.add_scalar(f"Probe_{probe_layer}/{error_name}", error.mean().item(), locs["it"])
                    for i in range(error.size(0)):
                        self.writer.add_scalar(f"Probe_{probe_layer}_{error_name}/{i:02d}", error[i].item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # callback for video logging
        if self.logger_type in ["wandb"]:
            self.writer.callback(locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        log_string = (
            f"""{"#" * width}\n"""
            f"""{str.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                locs["learn_time"]:.3f}s)\n"""
        )
        # -- Losses
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {
                self.tot_time / (locs["it"] + 1) * (locs["num_learning_iterations"] - locs["it"]):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "probes_state_dict": {name: probe.state_dict() for name, probe in self.alg.probes.items()},
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        saved_dict["probe_obs_norm_state_dict"] = self.probe_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)
        for name, probe in loaded_dict["probes_state_dict"].items():
            self.alg.probes[name].load_state_dict(probe)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        self.probe_obs_normalizer.load_state_dict(loaded_dict["probe_obs_norm_state_dict"])
        return loaded_dict["infos"]

    def load_actor(self, path, load_optimizer=False):
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # self.estimator_obs_normalizer.load_state_dict(loaded_dict["estimator_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None) -> Callable:
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def get_inference_estimator(self, device=None) -> Callable:
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        estimator = self.alg.policy.estimate
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            estimator = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return estimator

    def get_inference_probe(self, unnorm=False, device=None) -> Callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
            self.probe_obs_normalizer.to(device)
            for k, v in self.alg.probes.items():
                self.alg.probes[k] = v.to(device)

        def inference(obs, unnorm_) -> dict:
            _ = self.alg.act(obs)
            latents = self.alg.extract_latents()
            if unnorm_:
                return {k: self.probe_obs_normalizer.inverse(v(latents[k])) for k, v in self.alg.probes.items()}
            return {k: v(latents[k]) for k, v in self.alg.probes.items()}

        return lambda x: inference(x, unnorm)

    def train_mode(self):
        self.alg.policy.eval()
        for k, v in self.alg.probes.items():
            v.train()

    def eval_mode(self):
        self.alg.policy.eval()
        for k, v in self.alg.probes.items():
            v.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
